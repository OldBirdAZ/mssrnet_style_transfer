# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from torch.utils.data import Dataset
import torch
import json


class FtClsDataset(Dataset):
    def __init__(self, tgt_field, data_path, src_seq_length):
        super(FtClsDataset, self).__init__()
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self.src_seq_length = src_seq_length

        datas = []
        with open(data_path) as df:
            for line in df:
                example = json.loads(line)
                example['pred_probs'] = [
                    example['pred_label_0'],
                    example['pred_label_1']
                ]
                example['x_ids'] = self.convert_w2ids(example['text'])
                example['x_len'] = len(example['x_ids'])
                x_ids_with_eos = list(example['x_ids'])
                x_ids_with_eos.append(self._tgt_eos_idx)
                example['x_ids_with_eos'] = x_ids_with_eos
                datas.append(example)
        self._datas = datas

    def convert_w2ids(self, sent):
        ids = []
        toks = sent.split()[:self.src_seq_length - 1]
        for tok in toks:
            ids.append(self._tgt_vocab.stoi[tok] if tok in self._tgt_vocab.stoi else self._tgt_unk_idx)
        return ids

    def __len__(self):
        return len(self._datas)

    def __getitem__(self, item_idx):
        example = self._datas[item_idx]
        x_ids = example['x_ids']
        shift_len = self.src_seq_length - len(x_ids)
        if shift_len > 0:
            x_ids = x_ids + [self._tgt_pad_idx] * shift_len
        x_ids_with_eos = example['x_ids_with_eos']
        shift_len = shift_len - 1
        if shift_len > 0:
            x_ids_with_eos = x_ids_with_eos + [self._tgt_pad_idx] * shift_len

        result = {
            'x_ids': torch.LongTensor(x_ids),
            'x_ids_with_eos': torch.LongTensor(x_ids_with_eos),
            'x_len': torch.LongTensor([example['x_len']]),
            'probs': torch.FloatTensor(example['pred_probs'])
        }
        return result



