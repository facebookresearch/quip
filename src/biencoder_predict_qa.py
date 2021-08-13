# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Bi-encoder test-time code for SQuAD and MRQA."""
import argparse
import collections
import glob
import json
import os
import subprocess
import time
from tqdm import tqdm
import torch
from torch.nn import functional as F

from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

from mock_classification_head import MockClassificationHead

SQUAD_DEV_FILE = 'data/squad/dev-v1.1.json'
MRQA_DEV_DIRS = ['data/mrqa/data/dev', 'data/mrqa/data/ood-dev']
MAX_ANS_LEN = 17

def read_squad(squad_file, roberta, args):
    with open(squad_file) as f:
        squad_data = json.load(f)

    # Get examples to prepare to batch them
    examples = []
    truncated_questions = 0
    for a in squad_data['data']:
        for p in a['paragraphs']:
            paragraph_bpe = roberta.encode(p['context'])
            for q in p['qas']:
                question_bpe = roberta.encode(q['question'])
                if len(question_bpe) > args.question_max_len:
                    question_bpe = question_bpe[:args.question_max_len]
                    truncated_questions += 1
                examples.append((q['id'], paragraph_bpe, question_bpe))
    print(f'Truncated {truncated_questions}/{len(examples)} questions')
    return examples

def read_mrqa_all(roberta, args):
    MAX_PARAGRAPH_LEN = min(
            args.context_max_len + (args.batch_size - 1) * args.context_stride,
            args.context_max_len + (args.max_tokens // args.context_max_len - 1) * args.context_stride)
    print(f'Max paragraph len set to {MAX_PARAGRAPH_LEN}')
    examples = []
    filenames = []
    for dirname in MRQA_DEV_DIRS:
        for filename in sorted(glob.glob(os.path.join(dirname, '*.jsonl'))):
            filenames.append(filename)
            cur_paragraphs = 0
            cur_truncated_paragraphs = 0
            cur_questions = 0
            cur_truncated_questions = 0
            for line in open(filename):
                line = json.loads(line.strip())
                if 'header' in line:
                    continue
                paragraph_bpe = roberta.encode(line['context'].strip())
                cur_paragraphs += 1
                if len(paragraph_bpe) > MAX_PARAGRAPH_LEN:
                    paragraph_bpe = paragraph_bpe[:MAX_PARAGRAPH_LEN]
                    cur_truncated_paragraphs += 1
                for qa in line['qas']:
                    cur_questions += 1
                    question_bpe = roberta.encode(qa['question'].strip())
                    if len(question_bpe) > args.question_max_len:
                        question_bpe = question_bpe[:args.question_max_len]
                        cur_truncated_questions += 1
                    examples.append((qa['qid'], paragraph_bpe, question_bpe))
            print(f'{filename}: Truncated {cur_truncated_paragraphs}/{cur_paragraphs} paragraphs')
            print(f'{filename}: Truncated {cur_truncated_questions}/{cur_questions} questions')
    print(f'Max BPE length: {max(len(x[1]) for x in examples)}')
    return filenames, examples


def find_span(start_probs, end_probs, max_ans_len=MAX_ANS_LEN):
    best_score = 0.0
    best_start = None
    best_end = None
    for i in range(len(start_probs)):
        start_score = start_probs[i]
        for j in range(i, min(i + max_ans_len, len(end_probs))):
            end_score = end_probs[j]
            cur_score = start_score * end_score
            if cur_score > best_score:
                best_score = cur_score
                best_start = i
                best_end = j
    return best_start, best_end

def run_prediction(roberta, heads, examples, args):
    C_MAX = args.context_max_len
    STRIDE = args.context_stride 
    assert C_MAX % 2 == 0
    assert STRIDE % 2 == 0
    # Run prediction in batch
    pred_json = {}
    i = 0
    with tqdm(total=len(examples)) as pbar:
        while i < len(examples):
            # Greedily grab the next batch until batch_size and max_tokens are full
            cur_batch = []
            cur_tokens = 0
            qid_to_batch_index = collections.defaultdict(list)
            qid_to_full_bpe = {}
            while i < len(examples) and len(cur_batch) < args.batch_size:
                next_example = examples[i]
                qid, p_bpe, q_bpe = examples[i]
                if len(p_bpe) > C_MAX:
                    p_bpe_list = []
                    for j in range(0, len(p_bpe) - C_MAX + STRIDE, STRIDE):
                        cur_endpoint = min(len(p_bpe), j + C_MAX)
                        p_bpe_list.append(p_bpe[j:cur_endpoint])
                    cur_tokens = max(cur_tokens, C_MAX, len(q_bpe))
                else:
                    p_bpe_list = [p_bpe]
                    cur_tokens = max(cur_tokens, len(p_bpe), len(q_bpe))
                if (len(cur_batch) + len(p_bpe_list)) * cur_tokens > args.max_tokens:
                    # Wait until next batch to process this
                    break
                # Add to the batch now
                for p_bpe_chunk in p_bpe_list:
                    qid_to_batch_index[qid].append(len(cur_batch))
                    cur_batch.append((qid, p_bpe_chunk, q_bpe))
                qid_to_full_bpe[qid] = p_bpe
                i += 1

            # Make prediction
            with torch.no_grad():
                p_tokens = collate_tokens([b[1] for b in cur_batch], pad_idx=1)
                q_tokens = collate_tokens([b[2] for b in cur_batch], pad_idx=1)
                p_features = roberta.model(p_tokens.to(device=roberta.device),
                                           features_only=True,
                                           return_all_hiddens=False)[0]  # B, L, d
                q_features = roberta.model(q_tokens.to(device=roberta.device),
                                           features_only=True,
                                           return_all_hiddens=False)[0]  # B, L, d
                q_cls = q_features[:,0,:]  # B, d
                start_vec = heads['start'](q_cls)  # B, d
                end_vec = heads['end'](q_cls)  # B, d
                start_logits = torch.matmul(p_features, start_vec.unsqueeze(-1))[:,:,0]
                end_logits = torch.matmul(p_features, end_vec.unsqueeze(-1))[:,:,0]

                for qid, batch_idxs in qid_to_batch_index.items():
                    if len(batch_idxs) == 1:
                        # Easy case
                        cur_start_logits = start_logits[batch_idxs[0],:]
                        cur_end_logits = end_logits[batch_idxs[0],:]
                    else:
                        cur_start_list = []
                        cur_end_list = []
                        for j in batch_idxs:
                            if j == batch_idxs[0]: # Case 1: first index
                                cur_start_list.append(start_logits[j,:C_MAX//2 + STRIDE//2])
                                cur_end_list.append(end_logits[j,:C_MAX//2 + STRIDE//2])
                            elif j == batch_idxs[-1]: # Case 2: last index (will truncate later)
                                cur_start_list.append(start_logits[j,C_MAX//2 - STRIDE//2:])
                                cur_end_list.append(end_logits[j,C_MAX//2 - STRIDE//2:])
                            else: # Case 3: middle index
                                cur_start_list.append(start_logits[j, C_MAX//2 - STRIDE//2: C_MAX//2 + STRIDE//2])
                                cur_end_list.append(end_logits[j, C_MAX//2 - STRIDE//2: C_MAX//2 + STRIDE//2])
                        cur_start_logits = torch.cat(cur_start_list)[:len(qid_to_full_bpe[qid])]
                        cur_end_logits = torch.cat(cur_end_list)[:len(qid_to_full_bpe[qid])]
                    start_probs = F.softmax(cur_start_logits, dim=0).tolist()
                    end_probs = F.softmax(cur_end_logits, dim=0).tolist()
                    best_start, best_end = find_span(start_probs, end_probs)
                    pred_str = roberta.decode(qid_to_full_bpe[qid][best_start:best_end + 1])
                    pred_json[qid] = pred_str
                    pbar.update(1)

    with open(args.output_file, 'w') as f:
        json.dump(pred_json, f)


def main():
    parser = argparse.ArgumentParser('Generate predictions on SQuAD.')
    parser.add_argument('dataset_name', choices=['squad', 'mrqa'])
    parser.add_argument('output_file', help='Where to write predictions JSON')
    parser.add_argument('load_dir', help='Directory containing trained model checkpoint')
    parser.add_argument('--batch-size', '-b', default=32, type=int, help='Maximum batch size')
    parser.add_argument('--max-tokens', '-m', default=8800, type=int, help='Maximum tokens per batch')
    parser.add_argument('--context-max-len', type=int, default=456)
    parser.add_argument('--context-stride', type=int, default=328)  # Stride = distance between windows in sliding window; this makes the overlap region 456 - 328 = 128
    parser.add_argument('--question-max-len', type=int, default=456)  # Longer for MRQA
    args = parser.parse_args()

    roberta = RobertaModel.from_pretrained(args.load_dir, checkpoint_file='model.pt')
    roberta.to('cuda')
    roberta.eval()
    heads = {}
    for name in ['start', 'end']:
        head = MockClassificationHead.load_from_file(
                os.path.join(args.load_dir, f'model_qa_head_{name}.pt'),
                do_token_level=True)
        head.to('cuda')
        heads[name] = head
    print('Finished loading model.')

    print(f'Reading dataset {args.dataset_name}')
    if args.dataset_name == 'squad':
        examples = read_squad(SQUAD_DEV_FILE, roberta, args)
        run_prediction(roberta, heads, examples, args)
        subprocess.check_call(['python', 'data/squad/evaluate-v2.0.py', SQUAD_DEV_FILE, 
                               args.output_file])
    else:  # MRQA
        mrqa_filenames, examples = read_mrqa_all(roberta, args)
        run_prediction(roberta, heads, examples, args)
        for filename in mrqa_filenames:
            print(f'Evaluating on {filename}')
            subprocess.check_call(['python', 'data/mrqa/mrqa_official_eval.py', filename, args.output_file])

if __name__ == '__main__':
    main()

