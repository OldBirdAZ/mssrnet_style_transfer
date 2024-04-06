# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from torch.utils.tensorboard import SummaryWriter
import os


class TensorBoardWrapper:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.summ_writer = SummaryWriter(save_dir)

    def put_scalar(self, name, value, step):
        self.summ_writer.add_scalar(name, value, step)
        self.summ_writer.flush()

    def put_histogram(self, name, value, step):
        self.summ_writer.add_histogram(name, value, step)
        self.summ_writer.flush()

    def close(self):
        self.summ_writer.close()

