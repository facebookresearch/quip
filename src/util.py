# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""General utilities"""
PAD_TOKEN = 1

def make_batches(dataset, batch_size):
    batches = []
    cur_batch = []
    for i, ex in enumerate(dataset):
        cur_batch.append(ex)
        if len(cur_batch) == batch_size or i == len(dataset) - 1:
            batches.append(cur_batch)
            cur_batch = []
    return batches

def get_accuracy_f1_precision_recall(gold_labels, pred_labels):
    tp = sum(1 for y, p in zip(gold_labels, pred_labels) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(gold_labels, pred_labels) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(gold_labels, pred_labels) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(gold_labels, pred_labels) if y == 1 and p == 0)
    total = len(gold_labels)
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return acc, f1, precision, recall

def accuracy_f1_at_best_thresh(gold_labels, score_list):
    s_y_pairs = sorted(list(zip(score_list, gold_labels)), key=lambda x: -x[0])
    # Sort so highest score is first. 
    # This means that first threshold is labeling everything negative
    tp = 0
    fp = 0
    total_pos = sum(1 for y in gold_labels if y == 1)
    fn = total_pos
    best_err = fn
    best_f1 = 0
    for i, (s, y) in enumerate(s_y_pairs):
        if y == 1:
            tp += 1
            fn -=1
        else:
            fp += 1
        prec = tp / (i + 1)
        recall = tp / total_pos
        if prec == 0:
            f1 = 0
        else:
            f1 = 2 * prec * recall / (prec + recall)
        if fp + fn < best_err:
            best_err = fp + fn
        if f1 > best_f1:
            best_f1 = f1
    return (1 - best_err / len(gold_labels), best_f1)
