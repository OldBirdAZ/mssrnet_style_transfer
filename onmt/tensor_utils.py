# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import torch
from onmt.utils.misc import sequence_mask


def convert_to_onehot(labels, vocab_size, device):
    """
    :param labels: [N]
    :return:
    """
    batch_size = labels.size(0)
    labels_ids = labels.unsqueeze(1)
    empty = torch.zeros(batch_size, vocab_size).to(device)
    one_hots = empty.scatter_(1, labels_ids, 1)
    return one_hots


def word_drop(real_src, real_src_lengths, drop_prob, unk_idx):
    if not drop_prob or drop_prob <= 1e-4:
        return real_src
    x_valid_mask = sequence_mask(real_src_lengths).transpose(0, 1).unsqueeze(2)
    noise = torch.rand(real_src.size(), dtype=torch.float).to(real_src_lengths.device)
    noise_mask = noise < drop_prob
    noised_real_src = real_src.masked_fill(noise_mask & x_valid_mask, unk_idx)
    return noised_real_src

