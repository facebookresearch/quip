#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
set -eu -o pipefail
mkdir models
cd models
wget https://dl.fbaipublicfiles.com/quip/quip.tar.gz
tar -zxvf quip.tar.gz
rm quip.tar.gz
wget https://dl.fbaipublicfiles.com/quip/quip-hf.tar.gz
tar -zxvf quip-hf.tar.gz
rm quip-hf.tar.gz
