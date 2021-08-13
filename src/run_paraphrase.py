# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Paraphrase detection experiments."""
import argparse
import collections
import json
import numpy as np
import os
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys
import torch
from torch import nn
from tqdm import tqdm

from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

from util import get_accuracy_f1_precision_recall, accuracy_f1_at_best_thresh, make_batches, PAD_TOKEN

# QQP and MRPC are in the LM-BFF paper (their "test" sets are GLUE dev sets)
QQP_TRAIN_PATTERN = 'data/lm-bff/data/k-shot/QQP/16-{}/train.tsv'
QQP_DEV_PATTERN = 'data/lm-bff/data/k-shot/QQP/16-{}/dev.tsv'
QQP_TEST_FILE = 'data/lm-bff/data/k-shot/QQP/16-13/test.tsv'
MRPC_TRAIN_PATTERN = 'data/lm-bff/data/k-shot/MRPC/16-{}/train.tsv'
MRPC_DEV_PATTERN = 'data/lm-bff/data/k-shot/MRPC/16-{}/dev.tsv'
MRPC_TEST_FILE = 'data/lm-bff/data/k-shot/MRPC/16-13/test.tsv'
# Read PAWS from LM-BFF splits that were placed in the LM-BFF directory
PAWS_WIKI_TRAIN_PATTERN = 'data/lm-bff/data/k-shot/paws-wiki/16-{}/train.tsv'
PAWS_WIKI_DEV_PATTERN = 'data/lm-bff/data/k-shot/paws-wiki/16-{}/dev.tsv'
PAWS_WIKI_TEST_FILE = 'data/paws/wiki/test.tsv'
PAWS_QQP_TRAIN_PATTERN = 'data/lm-bff/data/k-shot/paws-qqp/16-{}/train.tsv'
PAWS_QQP_DEV_PATTERN = 'data/lm-bff/data/k-shot/paws-qqp/16-{}/dev.tsv'
PAWS_QQP_TEST_FILE = 'data/paws/qqp/dev_and_test.tsv'
KSHOT_SEEDS = [13, 21, 42, 87, 100]
FEATURE_LAYERS = list(range(17, 25))  # Layers used for classifier
FINAL_LAYER = 24  # Final layer of RoBERTa

class BertScoreModel(nn.Module):
    def __init__(self, roberta, layers, logit_scale):
        super(BertScoreModel, self).__init__()
        self.roberta_model = roberta.model
        self.layers = layers
        self.logit_scale = logit_scale
        self.output = nn.Linear(len(layers), 1)
        self.batchnorm = nn.BatchNorm1d(len(layers), momentum=None, eps=1e-8, affine=False)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x1, x2, len1, len2):
        """Compute binary prediction.

        x1: Tensor of shape (B, L1)
        x2: Tensor of shape (B, L2)
        len1: Tensor of shape (B,)
        len2: Tensor of shape (B,)
        """
        B = x1.shape[0]
        reps1 = self.roberta_model(x1, features_only=True, 
                                   return_all_hiddens=True)[1]['inner_states']  # Layer -> L, B, d
        reps2 = self.roberta_model(x2, features_only=True, 
                                   return_all_hiddens=True)[1]['inner_states']  
        feats = [[] for i in range(B)]
        for layer in self.layers:
            h1 = reps1[layer].permute(1, 0, 2)  # B, L1, d
            h2 = reps2[layer].permute(1, 0, 2)  # B, L2, d
            h1_normed = (h1 / torch.linalg.norm(h1, dim=2, keepdim=True))  # B, L1, d
            h2_normed = (h2 / torch.linalg.norm(h2, dim=2, keepdim=True))  # B, L2, d
            dots = torch.matmul(h1_normed, h2_normed.permute(0, 2, 1))  # B, L1, L2
            for i in range(B):
                cur_dots = dots[i,:len1[i],:len2[i]]
                s1 = torch.mean(torch.max(cur_dots, dim=0)[0])
                s2 = torch.mean(torch.max(cur_dots, dim=1)[0])
                score = 2 * s1 * s2 / (s1 + s2)
                feats[i].append(score)
        feat_mat = torch.stack([torch.stack(v) for v in feats])  # B, Layers
        feat_mat = self.batchnorm(feat_mat)
        out = self.logit_scale * self.output(feat_mat).squeeze(-1)
        return out

def read_qqp(filename, roberta):
    data = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            toks = line.strip().split('\t')
            uid, qid1, qid2, q1, q2, y = toks
            x1 = roberta.encode(q1)
            x2 = roberta.encode(q2)
            data.append(((x1, x2), int(y)))
    return data

def read_mrpc(filename, roberta):
    data = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            label, id1, id2, s1, s2 = line.strip().split('\t')
            x1 = roberta.encode(s1)
            x2 = roberta.encode(s2)
            y = int(label)
            data.append(((x1, x2), y))
    return data

def read_paws(filename, roberta):
    data = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            uid, s1, s2, y = line.strip().split('\t')
            x1 = roberta.encode(s1)
            x2 = roberta.encode(s2)
            data.append(((x1, x2), int(y)))
    return data

def read_data(dataset_name, roberta, train=False, kshot_seed=None):
    if dataset_name == 'qqp':
        if kshot_seed:
            if train:
                return read_qqp(QQP_TRAIN_PATTERN.format(kshot_seed), roberta)
            else:
                return read_qqp(QQP_DEV_PATTERN.format(kshot_seed), roberta)
        return read_qqp(QQP_TEST_FILE, roberta) 
    elif dataset_name == 'mrpc':
        if kshot_seed:
            if train:
                return read_mrpc(MRPC_TRAIN_PATTERN.format(kshot_seed), roberta)
            else:
                return read_mrpc(MRPC_DEV_PATTERN.format(kshot_seed), roberta)
        return read_mrpc(MRPC_TEST_FILE, roberta)
    elif dataset_name == 'paws-wiki':
        if kshot_seed:
            if train:
                return read_paws(PAWS_WIKI_TRAIN_PATTERN.format(kshot_seed), roberta)
            else:
                return read_paws(PAWS_WIKI_DEV_PATTERN.format(kshot_seed), roberta)
        return read_paws(PAWS_WIKI_TEST_FILE, roberta)
    elif dataset_name == 'paws-qqp':
        if kshot_seed:
            if train:
                return read_paws(PAWS_QQP_TRAIN_PATTERN.format(kshot_seed), roberta)
            else:
                return read_paws(PAWS_QQP_DEV_PATTERN.format(kshot_seed), roberta)
        return read_paws(PAWS_QQP_TEST_FILE, roberta)
    else:
        raise NotImplementedError
    return data


def compute_similarity(f1, f2, method):
    """Compute similarity between two contextualized representations.

    NOTE: Other methods besides bertscore aren't currently used but we did try
    them out.

    Args:
        f1: Tensor of shape (L_1, d)
        f2: Tensor of shape (L_2, d)
        method: Name of method
    Returns:
        Similarity score
    """
    if method == 'mean-cosine':
        r1 = torch.mean(f1, dim=0)  # d
        r2 = torch.mean(f2, dim=0)  # d
        return r1.dot(r2) / torch.sqrt(r1.dot(r1) * r2.dot(r2))
    elif method == 'bertscore' or method == 'bertscore-minmax':
        r1 = (f1.T / torch.norm(f1, dim=1)).T  # L_1, d
        r2 = (f2.T / torch.norm(f2, dim=1)).T  # L_2, d
        dots = torch.matmul(r1, r2.T)  # L_1, L_2
        if method == 'bertscore':
            s1 = torch.mean(torch.max(dots, dim=0)[0])
            s2 = torch.mean(torch.max(dots, dim=1)[0])
        elif method == 'bertscore-minmax':
            s1 = torch.min(torch.max(dots, dim=0)[0])
            s2 = torch.min(torch.max(dots, dim=1)[0])
        return 2 * s1 * s2 / (s1 + s2)
    raise NotImplementedError

def featurize_dataset(dataset, roberta, batch_size):
    feat_dicts = []
    batches = make_batches(dataset, batch_size)
    with torch.no_grad():
        for batch in tqdm(batches, desc='Computing BERTScore'):
            x0_batch = collate_tokens([x[0][0] for x in batch], pad_idx=PAD_TOKEN)
            x1_batch = collate_tokens([x[0][1] for x in batch], pad_idx=PAD_TOKEN)
            r0_batch = roberta.model(x0_batch.to(roberta.device),
                                    features_only=True, 
                                    return_all_hiddens=True)[1]['inner_states']  # Layer -> L, B, d
            r1_batch = roberta.model(x1_batch.to(roberta.device),
                                    features_only=True, 
                                    return_all_hiddens=True)[1]['inner_states']  # Layer -> L, B, d
            for i, ((x0, x1), y) in enumerate(batch):
                cur_feats = {}
                for layer in range(FINAL_LAYER + 1):
                    r0_i = r0_batch[layer][:len(x0), i, :].cpu()
                    r1_i = r1_batch[layer][:len(x1), i, :].cpu()
                    cur_feats[f'bertscore-{layer}'] = compute_similarity(r0_i, r1_i, 'bertscore').item()
                feat_dicts.append(cur_feats)
    return feat_dicts

def prune_features(feat_dicts):
    return [{k: v for k, v in d.items() if int(k.split('-')[1]) in FEATURE_LAYERS}
            for d in feat_dicts]

def train_skl(dataset, feat_dicts, skl_C=1.0):
    y_list = [y for (x, y) in dataset]
    vectorizer = DictVectorizer(sparse=False)
    X_raw = vectorizer.fit_transform(feat_dicts)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    model = LogisticRegression(C=skl_C, solver='liblinear')
    model.fit(X_scaled, y_list)
    return vectorizer, scaler, model

def predict_skl(features, vectorizer, scaler, model):
    X_raw = vectorizer.transform(features)
    X_scaled = scaler.transform(X_raw)
    pred_scores = model.decision_function(X_scaled)
    return pred_scores

def predict_zeroshot(dataset, feat_dicts):
    preds_per_layer = collections.defaultdict(list)  # Keep track of predictions from each layer
    for (((x1, x2), y), feats) in tqdm(zip(dataset, feat_dicts)):
        for layer in range(FINAL_LAYER + 1):
            preds_per_layer[f'bertscore, layer {layer}'].append(feats[f'bertscore-{layer}'])
    return preds_per_layer

def torchify_dataset(dataset, device):
    return [(x1.to(device), 
             x2.to(device), 
             torch.tensor(len(x1), device=device), 
             torch.tensor(len(x2), device=device),
             torch.tensor(y, device=device, dtype=torch.float32))
            for ((x1, x2), y) in dataset]

def collater(batch):
    batch = list(batch)
    x1 = collate_tokens([x[0] for x in batch], pad_idx=PAD_TOKEN)
    x2 = collate_tokens([x[1] for x in batch], pad_idx=PAD_TOKEN)
    len1 = torch.stack([x[2] for x in batch])
    len2 = torch.stack([x[3] for x in batch])
    y = torch.stack([x[4] for x in batch])
    return (x1, x2, len1, len2, y)

def train_ft(roberta, train_data, dev_data, logit_scale, batch_size, learning_rate, num_epochs):
    model = BertScoreModel(roberta, list(range(17, 25)), logit_scale)
    model.to(device=roberta.device)
    model.train()
    train_loader = torch.utils.data.DataLoader(
            torchify_dataset(train_data, roberta.device), 
            batch_size=batch_size, collate_fn=collater, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(
            torchify_dataset(dev_data, roberta.device), 
            batch_size=batch_size, collate_fn=collater, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 betas=(0.9, 0.98), eps=1e-6)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_num_correct = 0
        dev_loss = 0.0
        dev_num_correct = 0
        # Train
        for ex in train_loader:
            x1, x2, len1, len2, y = ex
            optimizer.zero_grad()
            out = model(x1, x2, len1, len2)
            loss = criterion(out, y)
            train_loss += loss.item()
            train_num_correct += sum(out * (2 * y - 1) > 0)
            loss.backward()
            optimizer.step()
        # Evaluate on dev
        model.eval()
        with torch.no_grad():
            for ex in dev_loader:
                x1, x2, len1, len2, y = ex
                out = model(x1, x2, len1, len2)
                loss = criterion(out, y)
                dev_loss += loss.item()
                dev_num_correct += sum(out * (2 * y - 1) > 0)
        train_acc = 100 * train_num_correct / len(train_data)
        dev_acc = 100 * dev_num_correct / len(dev_data)
        print(f'Epoch {epoch}: train loss={train_loss:.5f}, acc={train_acc:.2f}%; dev loss={dev_loss:.5f}, acc={dev_acc:.2f}%')
    return model

def predict_ft(dataset, model, device, batch_size):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
            torchify_dataset(dataset, device), 
            batch_size=batch_size, collate_fn=collater, shuffle=False)
    pred_scores = []
    with torch.no_grad():
        for ex in tqdm(data_loader):
            x1, x2, len1, len2, y = ex
            out = model(x1, x2, len1, len2)
            pred_scores.extend(out.tolist())
    return pred_scores

def evaluate(test_data, pred_scores, threshold=None):
    gold_labels = [y for x, y in test_data]
    stats = {
            'ap': average_precision_score(gold_labels, pred_scores),
            'auroc': roc_auc_score(gold_labels, pred_scores),
    }
    if threshold is not None:
        pred_labels = [int(x > threshold) for x in pred_scores]
        accuracy, f1, precision, recall = get_accuracy_f1_precision_recall(gold_labels, pred_labels)
        stats['acc'] = accuracy
        stats['f1'] = f1
        stats['precision'] = precision
        stats['recall'] = recall
    best_accuracy, best_f1 = accuracy_f1_at_best_thresh(gold_labels, pred_scores)
    stats['acc_best'] = best_accuracy
    stats['f1_best'] =  best_f1
    return stats

def average_results(results):
    out = {}
    for k in results[0].keys():
        arr = np.array([r[k] for r in results])
        out[f'{k}_mean'] = arr.mean()
        out[f'{k}_stderr'] = arr.std() / np.sqrt(len(arr))
    return out

def parse_args(args=None):
    parser = argparse.ArgumentParser('Run paraphrase detection with BERTScore experiments.')
    parser.add_argument('dataset', choices=['qqp', 'mrpc', 'paws-wiki', 'paws-qqp'], help='Which dataset to use')
    parser.add_argument('load_dir', help='Directory containing RoBERTa checkpoints')
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='Maximum batch size')
    # Fine-tuning 
    parser.add_argument('--train-ft', action='store_true', help='Train with fine-tuning.')
    parser.add_argument('--learning-rate', '-r', type=float, default=1e-5)
    parser.add_argument('--logit-scale', '-l', type=float, default=1e3)
    parser.add_argument('--num-epochs', '-T', type=int, default=20)
    parser.add_argument('--rng-seed', type=int, default=0)
    # scikit-learn training
    parser.add_argument('--train-skl', action='store_true', help='Train a scikit-learn model')
    parser.add_argument('--skl-C', type=float, default=1.0)
    return parser.parse_args(args=args)

def main(args):
    roberta = RobertaModel.from_pretrained(args.load_dir, 
                                           checkpoint_file='model.pt',
                                           dropout=0.0)
    # Load model
    roberta.to('cuda')
    roberta.eval()
    print('Finished loading model.')

    # Read test data
    test_data = read_data(args.dataset, roberta)
    print(f'{args.dataset}: Loaded {len(test_data)} test examples.')

    if args.train_ft:
        all_results = []
        for kshot_seed in KSHOT_SEEDS:
            random.seed(args.rng_seed)
            torch.manual_seed(args.rng_seed)
            train_data = read_data(args.dataset, roberta, train=True, kshot_seed=kshot_seed)
            print(f'Loaded {len(train_data)} training examples for kshot-seed {kshot_seed}.')
            dev_data = read_data(args.dataset, roberta, kshot_seed=kshot_seed)
            print(f'Loaded {len(dev_data)} dev examples.')
            model = train_ft(roberta, train_data, dev_data, args.logit_scale, 
                             args.batch_size, args.learning_rate, args.num_epochs)
            pred_scores = predict_ft(test_data, model, roberta.device, args.batch_size)
            stats = evaluate(test_data, pred_scores, threshold=0.0)
            all_results.append(stats)
            print(f'kshot-seed {kshot_seed}: {json.dumps(stats)}')
        print(f'Average: {json.dumps(average_results(all_results))}')
    elif args.train_skl:
        test_features = prune_features(featurize_dataset(test_data, roberta, args.batch_size))
        all_results = []
        for kshot_seed in KSHOT_SEEDS:
            train_data = read_data(args.dataset, roberta, train=True, kshot_seed=kshot_seed)
            print(f'Loaded {len(train_data)} training examples for kshot-seed {kshot_seed}.')
            train_features = prune_features(featurize_dataset(
                    train_data, roberta, args.batch_size))
            vectorizer, scaler, model = train_skl(train_data, train_features, skl_C=args.skl_C)
            pred_scores = predict_skl(test_features, vectorizer, scaler, model)
            stats = evaluate(test_data, pred_scores, threshold=0.0)
            all_results.append(stats)
            print(f'kshot-seed {kshot_seed}: {json.dumps(stats)}')
        print(f'Average: {json.dumps(average_results(all_results))}')
    else:
        feat_dicts = featurize_dataset(test_data, roberta, args.batch_size)
        preds_per_layer = predict_zeroshot(test_data, feat_dicts)
        for name, pred_scores in preds_per_layer.items():
            stats = evaluate(test_data, pred_scores)
            print(f'{name}: {json.dumps(stats)}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
