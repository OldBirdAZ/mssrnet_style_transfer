# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from torch.nn import NLLLoss


class VocabLossFn:
    def __init__(self, token_mode=False, ignore_index=None, all_tokens_aware=False):
        if all_tokens_aware or ignore_index is None:
            self.vocab_loss_fn = NLLLoss(reduction='mean' if token_mode else 'sum')
        else:
            self.vocab_loss_fn = NLLLoss(ignore_index=ignore_index, reduction='mean' if token_mode else 'sum')
        self.token_mode = token_mode

    def calculate_loss(self, log_softmax_logits, tgt_ids, batch_size):
        """
        :param log_softmax_logits: [N, vocab_size]
        :param tgt_ids: [N]
        :param batch_size: float or int
        :return: Tensor, loss tensor
        """
        loss = self.vocab_loss_fn(log_softmax_logits, tgt_ids)
        if self.token_mode:
            return loss
        else:
            return loss / batch_size
    pass


class UniformValChangeScheduler:
    def __init__(self, start_var, min_val, max_val, interval_step, interval_gap, continuous_mode=True):
        self.start_var = start_var
        self.min_val = min_val
        self.max_val = max_val
        self.interval_step = interval_step
        self.interval_gap = interval_gap
        self.continuous_mode = continuous_mode

        self._internal_step = 0
        self._cur_val = start_var

    def update(self):
        self._internal_step += 1
        move = self._internal_step / self.interval_step
        if not self.continuous_mode:
            move = int(move)
        self._cur_val = self.start_var + self.interval_gap * move
        if self.interval_gap > 0:
            self._cur_val = min(self._cur_val, self.max_val)
        else:
            self._cur_val = max(self._cur_val, self.min_val)

    def get_val(self):
        return self._cur_val

    def get_val_for_step(self, cur_step):
        move = cur_step / self.interval_step
        if not self.continuous_mode:
            move = int(move)
        _cur_val = self.start_var + self.interval_gap * move
        if self.interval_gap > 0:
            _cur_val = min(_cur_val, self.max_val)
        else:
            _cur_val = max(_cur_val, self.min_val)
        return _cur_val


