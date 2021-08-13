#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
set -eu -o pipefail
mkdir data
cd data

# GLUE
wget https://gist.githubusercontent.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3/raw/db67cdf22eb5bd7efe376205e8a95028942e263d/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all

# LM-BFF
git clone https://github.com/princeton-nlp/LM-BFF.git lm-bff
cd lm-bff/data
bash download_dataset.sh
cd ..
python tools/generate_k_shot_data.py
cd data/k-shot
md5sum -c checksum
cd ../../..

# SQuAD
mkdir squad
cd squad
wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json
wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json
wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/evaluate-v2.0.py
cd ..

# MRQA
git clone https://github.com/mrqa/MRQA-Shared-Task-2019.git mrqa
cd mrqa
./download_in_domain_dev.sh data/dev
./download_out_of_domain_dev.sh data/ood-dev
cd data/dev
gunzip *
cd ../ood-dev
gunzip *
cd ../../..

# PAWS
git clone https://github.com/google-research-datasets/paws
cd paws
# paws-wiki
wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz
tar -zxvf paws_wiki_labeled_final.tar.gz
mv final wiki
# paws-qqp
wget https://storage.googleapis.com/paws/english/paws_qqp.tar.gz
tar -zxvf paws_qqp.tar.gz
wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
export ORIGINAL_QQP_FILE="quora_duplicate_questions.tsv"
export PAWS_QQP_DIR="paws_qqp/"
export PAWS_QQP_OUTPUT_DIR="qqp"
mkdir $PAWS_QQP_OUTPUT_DIR
python2.7 qqp_generate_data.py \
  --original_qqp_input="${ORIGINAL_QQP_FILE}" \
  --paws_input="${PAWS_QQP_DIR}/train.tsv" \
  --paws_output="${PAWS_QQP_OUTPUT_DIR}/train.tsv"
python2.7 qqp_generate_data.py \
  --original_qqp_input="${ORIGINAL_QQP_FILE}" \
  --paws_input="${PAWS_QQP_DIR}/dev_and_test.tsv" \
  --paws_output="${PAWS_QQP_OUTPUT_DIR}/dev_and_test.tsv"
rm *.tar.gz
cd ../..
# Regenerate k-shot splits
python scripts/regenerate_paws_kshot.py
