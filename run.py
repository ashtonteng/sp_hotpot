import operator
import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from model import Model #, NoCharModel, NoSelfModel
from sp_model import SPModel
# from normal_model import NormalModel, NoSelfModel, NoCharModel, NoSentModel
# from oracle_model import OracleModel, OracleModelV2
# from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from util import convert_tokens, evaluate
from util import get_buckets, DataIterator, IGNORE_INDEX
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F
from pytorch_pretrained_bert.tokenization import BertTokenizer
import process_data
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from hotpot_qa_model import BertForHotpotQA

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

nll_sum = nn.CrossEntropyLoss(reduction='sum', ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)

def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        # torch.cuda.set_device(1) # when only one device, set device here
        # device = torch.device("cuda", 1)
        torch.cuda.manual_seed_all(config.seed)

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'sp_model.py'])
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    logging("Building model...")
    train_buckets = get_buckets(config.train_record_file)
    dev_buckets = get_buckets(config.dev_record_file)

    def build_train_iterator():
        return DataIterator(config.cuda, train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, config.sent_limit)

    def build_dev_iterator():
        return DataIterator(config.cuda, dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, config.sent_limit)

    model = SPModel(config, word_mat, char_mat)

    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))

    if config.cuda:
        print("im using my gpus!")
        ori_model = model.cuda()
        #model.to(device)
        # model = nn.DataParallel(ori_model)
    else:
        ori_model = model

    lr = config.init_lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)  # switch to Adam optimizer
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    model.train()

    for epoch in range(10000):
        for data in build_train_iterator():
            context_idxs = data['context_idxs']
            ques_idxs = data['ques_idxs']
            context_char_idxs = data['context_char_idxs']
            ques_char_idxs = data['ques_char_idxs']
            context_lens = data['context_lens']
            # y1 = Variable(data['y1'])
            # y2 = Variable(data['y2'])
            # q_type = Variable(data['q_type'])
            is_support = data['is_support']
            start_mapping = data['start_mapping']
            end_mapping = data['end_mapping']
            all_mapping = data['all_mapping']

            predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens,
                                    start_mapping, end_mapping, all_mapping, return_yp=False)
            # logit1, logit2, predict_type, predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False)
            # loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
            # loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
            # loss = loss_1 + config.sp_lambda * loss_2
            loss = nll_average(predict_support.view(-1, 2), is_support.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch, global_step, lr, elapsed*1000/config.period, cur_loss))
                total_loss = 0
                start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()
                #torch.cuda.empty_cache()
                metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file, config)
                model.train()

                logging('-' * 89)
                logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f} | Precision {:.4f} | Recall {:.4f}'.format(global_step//config.checkpoint,
                    epoch, time.time()-eval_start_time, metrics['loss'], metrics['em'], metrics['f1'], metrics['prec'], metrics['recall']))
                logging('-' * 89)

                eval_start_time = time.time()

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        lr /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < config.init_lr * 1e-2:
                            stop_train = True
                            break
                        cur_patience = 0
        if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_F1))

def evaluate_batch(data_source, model, max_batches, eval_file, config):
    # answer_dict = {}
    sp_dict = {}
    total_loss, step_cnt = 0, 0
    iter = data_source
    sp_th = config.sp_threshold
    for step, data in enumerate(iter):
        if step >= max_batches and max_batches > 0: break

        context_idxs = Variable(data['context_idxs'])#, volatile=True)
        ques_idxs = Variable(data['ques_idxs'])#, volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'])#, volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'])#, volatile=True)
        context_lens = Variable(data['context_lens'])#, volatile=True)
        # y1 = Variable(data['y1'], volatile=True)
        # y2 = Variable(data['y2'], volatile=True)
        # q_type = Variable(data['q_type'], volatile=True)
        is_support = Variable(data['is_support'])#, volatile=True)
        start_mapping = Variable(data['start_mapping'])#, volatile=True)
        end_mapping = Variable(data['end_mapping'])#, volatile=True)
        all_mapping = Variable(data['all_mapping'])#, volatile=True)

        predict_support = model(context_idxs, ques_idxs, context_char_idxs,
                           ques_char_idxs, context_lens, start_mapping,
                           end_mapping, all_mapping, return_yp=True)
        loss = nll_sum(predict_support.view(-1, 2), is_support.view(-1))

        # logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        # loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) + config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1))

        # answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        # answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = data['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            sp_dict.update({cur_id: cur_sp_pred})

        total_loss += loss.item()  # previously loss.data[0]
        step_cnt += 1
    loss = total_loss / step_cnt
    metrics = evaluate(eval_file, sp_dict) # answer_dict)
    metrics['loss'] = loss

    return metrics


def predict(data_source, sp_model, eval_file, config, prediction_file, qa_model=None):
    torch.set_grad_enabled(False)
    sp_dict = dict()
    qa_sp_dict = dict()
    answer_dict = {}  # only used when qa_model != None
    # sp_logits_dict = dict() ###
    sp_th = config.sp_threshold
    qa_sp_th = config.qa_sp_threshold

    if qa_model != None:

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        hotpot_dict = process_data.hotpot_to_dict(config.hotpot_file)
        print("wrong series description")
        device2 = torch.device("cuda", 1)
    for step, data in enumerate(tqdm(data_source, desc="Iteration")):
        #torch.cuda.empty_cache()
        context_idxs = Variable(data['context_idxs'])#, volatile=True)
        ques_idxs = Variable(data['ques_idxs'])#, volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'])#, volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'])#, volatile=True)
        context_lens = Variable(data['context_lens'])#, volatile=True)
        start_mapping = Variable(data['start_mapping'])#, volatile=True)
        end_mapping = Variable(data['end_mapping'])#, volatile=True)
        all_mapping = Variable(data['all_mapping'])#, volatile=True)

        predict_support = sp_model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens,
                                          start_mapping, end_mapping, all_mapping, return_yp=True)
        # logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        # answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        # answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        sp_batch_dict = dict()
        qa_sp_batch_dict = dict()
        #sp_batch_logits_dict = dict() ###
        for i in range(predict_support_np.shape[0]): # for item in batch
            #cur_sp_pred_logits = [] ###
            cur_qa_sp_pred = []
            cur_sp_pred = []
            cur_id = data['ids'][i]
            max_sp = None
            max_sp_score = 0 
            for j in range(predict_support_np.shape[1]): # for each sentence in the paragraph
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                sp_score = predict_support_np[i, j] # our predicted score for this sentence
                sp = eval_file[cur_id]['sent2title_ids'][j]
                #cur_sp_pred_logits.append((sp, sp_score)) ###
                if sp_score > max_sp_score: # always keep max sp, even if not past threshold
                    max_sp_score = sp_score
                    max_sp = sp
                if predict_support_np[i, j] > qa_sp_th:
                    cur_qa_sp_pred.append(sp)
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(sp)
            if len(cur_qa_sp_pred) == 0:  # if none of the sentences made the cut, pick the max score one
                cur_qa_sp_pred = [max_sp]
            #sp_batch_logits_dict[cur_id] = cur_sp_pred_logits ###
            qa_sp_batch_dict[cur_id] = cur_qa_sp_pred
            sp_batch_dict[cur_id] = cur_sp_pred
        #sp_logits_dict.update(sp_batch_logits_dict) ###
        qa_sp_dict.update(qa_sp_dict)
        sp_dict.update(sp_batch_dict)
        if qa_model != None:
            #import pdb
            #pdb.set_trace()
            ################## TODO ############################# does hotpot dict has yes/no?
            squad_format_pred, supporting_fact_dict = process_data.pred_2_squad(hotpot_dict, qa_sp_batch_dict)
            pred_data = process_data.read_squad_examples(squad_format_pred, is_training=False, version_2_with_negative=True)
            eval_features = process_data.convert_examples_to_features(pred_data, tokenizer, is_training=False)            # TODO tokenizer, max_seq_length, doc_stride, max_query_length

            input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            example_index = torch.arange(input_ids.size(0), dtype=torch.long)
            #import pdb
            #pdb.set_trace()
            yns = torch.tensor([f.yns for f in eval_features], dtype=torch.long).view(config.batch_size)
            input_ids = input_ids.to(device2)
            input_mask = input_mask.to(device2)
            segment_ids = segment_ids.to(device2)
            yns = yns.to(device2)
            batch_start_logits, batch_end_logits, yes_no_span = qa_model(input_ids, segment_ids, input_mask)
            yes_no_list = yes_no_span.detach().cpu().tolist()
            for i, example_index in enumerate(example_index):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                yes_no_logits = yes_no_list[i]
                eval_feature = eval_features[example_index.item()]
                unique_id = eval_feature.unique_id
                yes_no_int = np.argmax(yes_no_logits)
                if yes_no_int == 0:
                    answer = "yes"
                elif yes_no_int == 1:
                    answer = "no"
                else:
                    start = max(enumerate(start_logits), key=operator.itemgetter(1))[0]
                    end = max(enumerate(end_logits), key=operator.itemgetter(1))[0]
                    answer = " ".join(supporting_fact_dict[unique_id].split()[start:end])
                # TODO double check start and end
                answer_dict[unique_id] = answer
    #import pickle
    #pickle.dump(sp_logits_dict, open("sp_logits_dict.pkl", "wb"))
    if config.integrate:
        prediction = {'answer': answer_dict, 'sp': sp_dict}
    else:
        prediction = {'sp': sp_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)

def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    if config.data_split == 'dev':
        with open(config.dev_eval_file, "r") as fh:
            dev_eval_file = json.load(fh)
    else:
        with open(config.test_eval_file, 'r') as fh:
            dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    if config.data_split == 'dev':
        dev_buckets = get_buckets(config.dev_record_file)
        para_limit = config.para_limit
        ques_limit = config.ques_limit
    elif config.data_split == 'test':
        para_limit = None
        ques_limit = None
        dev_buckets = get_buckets(config.test_record_file)

    def build_dev_iterator():
        return DataIterator(config.cuda, dev_buckets, config.batch_size, para_limit,
            ques_limit, config.char_limit, False, config.sent_limit)


    sp_model = SPModel(config, word_mat, char_mat)

    if config.integrate:
        #qa_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased", cache_dir="~/.pytorch_pretrained_bert")
        qa_model = BertForHotpotQA.from_pretrained("bert-base-uncased", cache_dir="~/hotpot/.pytorch_pretrained_bert")
        qa_model.load_state_dict(torch.load(config.qa_model_save))
    if config.cuda:
        # when only one device, set device here
        device = torch.device("cuda", 0)
        device2 = torch.device("cuda", 1)
        sp_saved_weights = torch.load(config.sp_model_save)
    else:
        sp_saved_weights = torch.load(config.sp_model_save, map_location="cpu")

    if config.cuda:
        print("im using my gpus!")
        ori_sp_model = sp_model.cuda()
        ori_sp_model.load_state_dict(sp_saved_weights)
        ori_sp_model.to(device)
        # model = nn.DataParallel(ori_model)
        sp_model = ori_sp_model
        if config.integrate:
            qa_model.to(device2)
    else:
        sp_model.load_state_dict(sp_saved_weights)

    sp_model.eval()

    if config.integrate:
        qa_model.eval()
        predict(build_dev_iterator(), sp_model, dev_eval_file, config, config.prediction_file, qa_model)
    else:
        predict(build_dev_iterator(), sp_model, dev_eval_file, config, config.prediction_file)
