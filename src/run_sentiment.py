# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Zero-shot sentiment analysis experiments."""
import argparse
import csv
import json
import os
import random
from sklearn.metrics import average_precision_score, roc_auc_score
import sys
import torch
from tqdm import tqdm

from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

from biencoder_predict_qa import find_span 
from mock_classification_head import MockClassificationHead
from util import make_batches, PAD_TOKEN, get_accuracy_f1_precision_recall, accuracy_f1_at_best_thresh

SEP_TOKEN = 2

SST2_DEV_FILE = 'data/glue_data/SST-2/dev.tsv'
MR_TEST_FILE = 'data/lm-bff/data/original/mr/test.csv'
CR_TEST_FILE = 'data/lm-bff/data/original/cr/test.csv'

MOVIE_PROMPTS = [
        [('Why is it good?', 1), ('Why is it bad?', 0)],
        [('Why is this movie good?', 1), ('Why is this movie bad?', 0)],
        [('Why is it great?', 1), ('Why is it terrible?', 0)],
        [('What makes this movie good?', 1), ('What makes this movie bad?', 0)],
        [('What is the reason this movie is good?', 1), ('What is the reason this movie is bad?', 0)],
        [('What is the reason this movie is great?', 1),('What is the reason this movie is terrible?', 0)],
]
PRODUCT_PROMPTS = [
        [('Why is it good?', 1), ('Why is it bad?', 0)],
        [('Why is this product good?', 1), ('Why is this product bad?', 0)],
        [('Why is it great?', 1), ('Why is it terrible?', 0)],
        [('What makes this product good?', 1), ('What makes this product bad?', 0)],
        [('What is the reason this product is good?', 1), ('What is the reason this product is bad?', 0)],
        [('What is the reason this product is great?', 1),('What is the reason this product is terrible?', 0)],
]

QA_PROMPTS = {
        'sst2': MOVIE_PROMPTS,
        'mr': MOVIE_PROMPTS,
        'cr': PRODUCT_PROMPTS,
}

LM_PROMPT = (' It was', '.', [' terrible', ' great'])  # LM-BFF uses the same prompt for all these datasets
MLM_PROMPTS = {
        'sst2': LM_PROMPT,
        'cr': LM_PROMPT,
        'mr': LM_PROMPT
}
CALIBRATION_EXAMPLES = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I']

def read_data(dataset_name, roberta, num_examples=None):
    if dataset_name == 'sst2':
        data = []
        with open(SST2_DEV_FILE) as f:
            for i, line in enumerate(f):
                if i == 0: continue
                s, y = line.strip().split('\t')
                x = roberta.encode(s.strip())
                data.append((x, int(y)))
    elif dataset_name == 'mr':
        data = []
        with open(MR_TEST_FILE, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append((roberta.encode(row[1]), int(row[0])))
    elif dataset_name == 'cr':
        data = []
        with open(CR_TEST_FILE, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append((roberta.encode(row[1]), int(row[0])))
    else:
        raise NotImplementedError
    print(f'Read {len(data)} examples')
    if num_examples:
        random.shuffle(data)
        data = data[:num_examples]
        print(f'Truncated to {len(data)} examples')
    return data

def evaluate(gold_labels, pred_scores, threshold=0.0):
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


def parse_args(args=None):
    parser = argparse.ArgumentParser('Run QA reduction experiments.')
    parser.add_argument('dataset', choices=['sst2', 'mr', 'cr'], help='Which dataset to use')
    parser.add_argument('method', choices=['mlm', 'qa'], help='How to prompt')
    parser.add_argument('load_dir', help='Directory containing RoBERTa checkpoints')
    parser.add_argument('--prompt-index', '-p', default=4, type=int, help='Which prompt pair to use for QA (defaults to best prompt for SST-2)')
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='Maximum batch size')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--num-examples', '-n', type=int, default=None, help='Limit to some number of examples')
    parser.add_argument('--calibrate-lmbff', action='store_true', help='Try calibration with LM-BFF')
    return parser.parse_args(args=args)

def main(args):
    random.seed(0)

    # Load model, including start/end heads
    roberta = RobertaModel.from_pretrained(args.load_dir, checkpoint_file='model.pt')
    roberta.to('cuda')
    roberta.eval()
    if args.method == 'qa':
        heads = {}
        for name in ['start', 'end']:
            head = MockClassificationHead.load_from_file(
                    os.path.join(args.load_dir, f'model_qa_head_{name}.pt'),
                    do_token_level=True)
            head.to('cuda')
            heads[name] = head
    print('Finished loading model.')

    # Read data
    data = read_data(args.dataset, roberta, num_examples=args.num_examples)
    batch_data = make_batches(data, args.batch_size)
    print(f'Loaded {len(data)} examples ({len(batch_data)} batches).')

    with torch.no_grad():
        # Precompute prompt representations 
        calib_scores = []
        if args.method == 'qa':
            prompts = QA_PROMPTS[args.dataset][args.prompt_index]
            prompt_vecs = []
            for q, y in prompts:
                x = roberta.encode(q).unsqueeze(0)  # 1, L
                feats = roberta.model(x.to(roberta.device),
                                      features_only=True,
                                      return_all_hiddens=False)[0]  # 1, L, d
                cls_feats = feats[0,0,:]  # d
                start_vec = heads['start'](cls_feats)  # d
                end_vec = heads['end'](cls_feats)  # d
                prompt_vec = torch.stack([start_vec, end_vec], dim=1)
                prompt_vecs.append(prompt_vec)  # d, 2 

                # Contextual calibration
                cur_calib_scores = []
                for x_calib in CALIBRATION_EXAMPLES: 
                    toks_calib = roberta.encode(x_calib)
                    feats, _ = roberta.model(toks_calib.to(roberta.device).unsqueeze(0),
                                          features_only=True,
                                          return_all_hiddens=False)  # 1, L, d
                    word_scores = torch.matmul(feats[0,:,:], prompt_vec)  # L, 2
                    cur_calib_scores.append(torch.max(word_scores, dim=0)[0].sum().item())
                calib_scores.append(sum(cur_calib_scores) / len(cur_calib_scores))
            print('calibration:', calib_scores)

        else:
            prompt_start, prompt_end, prompt_options = MLM_PROMPTS[args.dataset]
            prompt_toks = roberta.task.source_dictionary.encode_line(
                    roberta.bpe.encode(prompt_start) + ' <mask> ' + roberta.bpe.encode(prompt_end))
            prompt_option_indices = [roberta.encode(x)[1] for x in prompt_options]

            # Contextual calibration
            if args.calibrate_lmbff:
                cur_calib_scores = [[], []]
                for x_calib in CALIBRATION_EXAMPLES:
                    toks_calib = torch.cat([roberta.encode(x_calib)[:-1], prompt_toks])
                    feats, _ = roberta.model(toks_calib.to(roberta.device).unsqueeze(0))
                    mask_idx = (toks_calib == roberta.task.mask_idx).nonzero(as_tuple=False)
                    logits = feats[0,mask_idx,:].squeeze()
                    for y, prompt_option_idx in enumerate(prompt_option_indices):
                        cur_calib_scores[y].append(logits[prompt_option_idx].item())
                calib_scores = [sum(x) / len(x) for x in cur_calib_scores]
            else:
                calib_scores = [0, 0]

            print(f'prompt_toks={prompt_toks}, options={prompt_option_indices}, calibration={calib_scores}')

        print('Preprocessed prompts.')

        # Score predictions
        gold_labels = []
        pred_labels = []
        pred_scores = []
        for batch in tqdm(batch_data):
            cur_pred_scores = [{} for b in batch]
            explanations = [{} for b in batch]
            if args.method == 'qa':
                x_batch = collate_tokens([x for (x, y) in batch], pad_idx=PAD_TOKEN)
                feats = roberta.model(x_batch.to(roberta.device),
                                      features_only=True,
                                      return_all_hiddens=False)[0]  # B, L, d
                for (q, y), prompt_vec, calib_score in zip(prompts, prompt_vecs, calib_scores):
                    prompt_mat = prompt_vec.unsqueeze(0).expand(len(batch), -1, -1)  # B, d, 2
                    word_scores = torch.matmul(feats, prompt_mat)  # B, L, 2
                    # Max across words, then sum across start + end vectors
                    agg_scores = torch.max(word_scores, dim=1)[0].sum(dim=1).tolist()  # B
                    for i in range(len(batch)):
                        cur_pred_scores[i][y] = agg_scores[i] - calib_score
                        if args.verbose:
                            start_idx, end_idx = find_span(word_scores[i,:,0].softmax(dim=0),
                                                           word_scores[i,:,1].softmax(dim=0),
                                                           max_ans_len=5)  # Find a short rationale
                            try:
                                explanations[i][y] = roberta.decode(batch[i][0][start_idx:end_idx+1])
                            except IndexError:
                                explanations[i][y] = ''
            else:  # args.method == 'mlm'
                xs_with_prompt = [torch.cat([x[:-1], prompt_toks]) for (x, y) in batch]  # Strip the old EOS
                x_batch = collate_tokens(xs_with_prompt, pad_idx=PAD_TOKEN)
                feats, _ = roberta.model(x_batch.to(roberta.device))  # B, L, V
                for i, x_with_prompt in enumerate(xs_with_prompt):
                    mask_idx = (x_with_prompt == roberta.task.mask_idx).nonzero(as_tuple=False)
                    logits = feats[i,mask_idx,:].squeeze()
                    for y, prompt_option_idx in enumerate(prompt_option_indices):
                        cur_pred_scores[i][y] = logits[prompt_option_idx].item() - calib_scores[y]
            for i, (x, y) in enumerate(batch):
                gold_labels.append(y)
                pred_scores.append(cur_pred_scores[i][1] - cur_pred_scores[i][0])
                y_pred, max_score = max(cur_pred_scores[i].items(), key=lambda p: p[1])
                pred_labels.append(y_pred)
                if args.verbose:
                    log_obj = {
                            'x': roberta.decode(x),
                            'y': y,
                            'pred': y_pred,
                            'scores': cur_pred_scores[i],
                            'explanation': explanations[i][y] if explanations[i] else ''
                    }
                    print(json.dumps(log_obj))

    # Print stats
    num_correct = sum(1 for y, pred in zip(gold_labels, pred_labels) if y == pred)
    print(f'Accuracy: {num_correct}/{len(gold_labels)} = {100 * num_correct / len(gold_labels):.2f}%')
    fp = sum(1 for y, pred in zip(gold_labels, pred_labels) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(gold_labels, pred_labels) if y == 1 and pred == 0)
    print(f'fp={fp}, fn={fn}')
    print(evaluate(gold_labels, pred_scores))


if __name__ == '__main__':
    args = parse_args()
    main(args)
