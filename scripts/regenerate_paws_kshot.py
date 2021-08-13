# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Re-generate our PAWS k-shot data files."""
import os

def main():
    for dataset in ['qqp', 'wiki']:
        id_to_example = {}
        with open(os.path.join('data/paws', dataset, 'train.tsv')) as f:
            for i, line in enumerate(f):
                if i == 0:
                    header = line.strip()
                else:
                    ex_id = line.split('\t')[0]
                    id_to_example[ex_id] = line.strip()
        for split in ['train', 'dev']:
            for seed in [13, 21, 42, 87, 100]:
                id_list_fn = os.path.join('paws_kshot_splits', 
                                          f'paws-{dataset}', 
                                          f'16-{seed}',
                                          f'{split}_ids.tsv')
                cur_out_lines = [header]
                with open(id_list_fn) as f:
                    for line in f:
                        cur_out_lines.append(id_to_example[line.strip()])
                cur_out_dir = os.path.join('data/lm-bff/data/k-shot', 
                                           f'paws-{dataset}', 
                                           f'16-{seed}')
                if split == 'train':
                    os.makedirs(cur_out_dir)
                with open(os.path.join(cur_out_dir, f'{split}.tsv'), 'w') as f:
                    for line in cur_out_lines:
                        print(line, file=f)

if __name__ == '__main__':
    main()
