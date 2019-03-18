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
import collections
import math

from pytorch_pretrained_bert.tokenization import BasicTokenizer

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

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    #logger.info("Writing predictions to: %s" % (output_prediction_file))
    #logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    
    print("*",len(all_results))

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        print(len(prelim_predictions))
        for pred in prelim_predictions:
            #if len(nbest) >= n_best_size:
            #    print(len(nbest))
                #break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
                
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            print("no nbest")
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        print(len(nbest))
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

    return all_predictions

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


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

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
    all_results = []
    pred_data_list = []
    eval_features_list = []
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
            pred_data_list += pred_data
            eval_features = process_data.convert_examples_to_features(pred_data, tokenizer, is_training=False)            # TODO tokenizer, max_seq_length, doc_stride, max_query_length
            eval_features_list += eval_features

            input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            example_index = torch.arange(input_ids.size(0), dtype=torch.long)
            #import pdb
            #pdb.set_trace()
            input_ids = input_ids.to(device2)
            input_mask = input_mask.to(device2)
            segment_ids = segment_ids.to(device2)
            batch_start_logits, batch_end_logits= qa_model(input_ids, segment_ids, input_mask)


            for i, example_index in enumerate(example_index):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = eval_feature.unique_id
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
            #for i, example_index in enumerate(example_index):
                #start_logits = batch_start_logits[i].detach().cpu().tolist()
                #end_logits = batch_end_logits[i].detach().cpu().tolist()
                #eval_feature = eval_features[example_index.item()]
                #unique_id = eval_feature.unique_id
 
                #start = max(enumerate(start_logits), key=operator.itemgetter(1))[0]
                #end = max(enumerate(end_logits), key=operator.itemgetter(1))[0]
                #answer = " ".join(supporting_fact_dict[unique_id].split()[start:end])
                # TODO double check start and end
                #answer_dict[unique_id] = answer
    len(all_results)
    answer_dict = write_predictions(pred_data_list, eval_features_list, all_results,
                          10, 30,True, None, None, None, True, False, 0.0)
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
        qa_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased", cache_dir="~/.pytorch_pretrained_bert")
        #qa_model = BertForHotpotQA.from_pretrained("bert-base-uncased", cache_dir="~/hotpot/.pytorch_pretrained_bert")
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
