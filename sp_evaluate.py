import sys
import ujson as json
import re
import string
from collections import Counter
import pickle


def f1_score(prediction, ground_truth):

    ZERO_METRIC = (0, 0, 0)

    # if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    #     return ZERO_METRIC
    # if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    #     return ZERO_METRIC

    prediction_tokens = [tuple(e) for e in prediction] #normalized_prediction.split()
    ground_truth_tokens = [tuple(e) for e in ground_truth] #normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction, gold):

    metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}

    for dp in gold:
        cur_id = dp['_id']
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
        else:
            update_sp(metrics, prediction['sp'][cur_id], dp['supporting_facts'])

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    return metrics

def plot_precision_recall(prediction, gold):
    ems = []
    f1s = []
    precisions = []
    recalls = []
    import numpy as np
    for sp_thresh in np.arange(0.0, 1.0, 0.01):
        metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
        sp_preds = [] # list of [title, idx], predictions for sp
        for dp in gold:
            cur_id = dp['_id']
            if cur_id in prediction['sp']:
                sp_logit_tuples = prediction['sp'][cur_id] # list of supporting facts, each is ([title, idx], score)
                for sp, score in sp_logit_tuples:
                    if score > sp_thresh:
                        sp_preds.append(sp)
                update_sp(metrics, sp_preds, dp['supporting_facts'])
        N = len(gold)
        for k in metrics.keys():
            metrics[k] /= N
        ems.append(metrics['sp_em'])
        f1s.append(metrics['sp_f1'])
        precisions.append(metrics['sp_prec'])
        recalls.append(metrics['sp_recall'])
    import pickle
    pickle.dump([ems, f1s, precisions, recalls], open("stats.pkl", "wb"))

if __name__ == '__main__':
    prediction_file, gold_file = sys.argv[1], sys.argv[2]
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    metrics = eval(prediction, gold)
    print(metrics)

