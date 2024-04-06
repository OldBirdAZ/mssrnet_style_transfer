# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from onmt.utils.optimizers import make_learning_rate_decay_fn


class LrAdapter(object):
    def __init__(self, opt):
        self._decay_step = 1
        self._learning_rate_decay_fn = make_learning_rate_decay_fn(opt)
        self._learning_rate = opt.learning_rate

    def get_new_lr_(self, step):
        scale = self._learning_rate_decay_fn(step)
        return scale * self._learning_rate

    def step_and_get_lr(self):
        lr = self.get_new_lr_(self._decay_step)
        self._decay_step += 1
        return lr

    def get_cur_step_lr(self):
        return self.get_new_lr_(self._decay_step)

    def apply_to_optimizer(self, optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr




