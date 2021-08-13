#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
set -eu -o pipefail
# Sweep NER results across 5 random seeds for few-shot
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
mkdir -p $OUT_DIR
date=$(date +"%Y-%M-%d %H:%M:%S")
echo "$date $DATASET $OUT_DIR $MODEL, flags=$FLAGS"
for seed in 0 1 2 3 4
do
    bash scripts/run_ner.sh 5shot-${DATASET} ${OUT_DIR}/outdir_${seed} ${MODEL} ${FLAGS} --few_shot_seed ${seed} --seed ${seed} 2> ${OUT_DIR}/err_${seed}.txt > ${OUT_DIR}/out_${seed}.txt
done
grep eval_f1 $OUT_DIR/outdir_*/test_results.txt | cut -d ' ' -f3 | tr '\n' ' '
echo
