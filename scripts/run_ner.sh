#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
set -eu -o pipefail
if [ $# == 0 ]
then
    echo "Usage: $0 [conll2003|wnut_17] <out_dir> <model_name_or_path> [flags]"
    exit 1
fi
DATASET="$1"
OUT_DIR="$2"
MODEL="$3"
shift
shift
shift
FLAGS="$@"

python src/hf/run_ner.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --tokenizer_name roberta-large \
    --do_train --do_eval --do_predict \
    --save_steps 2000 \
    --output_dir ${OUT_DIR} $FLAGS

