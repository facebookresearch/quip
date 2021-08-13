# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Replacement for RobertaClassificationHead.

Easier to deal with this separately because normal RoBERTa classification heads
are configured based on tasks.

Note: RoBERTa uses tanh for pooler activation function (which is used for classification heads) by default.
"""
import torch
from torch import nn

class MockClassificationHead(nn.Module):
    """Head for sentence-level or token-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, do_token_level=False):
        super().__init__()
        self.do_token_level = do_token_level
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = torch.tanh
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        if self.do_token_level:
            x = features  # score every token
        else:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.out_proj(x)
        return x

    @classmethod
    def load_from_file(cls, filename, do_token_level=False):
        params = torch.load(filename)
        input_dim, inner_dim = params['dense.weight'].shape
        num_classes = params['out_proj.weight'].shape[1]
        model = cls(input_dim, inner_dim, num_classes, do_token_level=do_token_level)
        model.load_state_dict(params)
        return model
