# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function


class InfoLogger:
    def __init__(self, *keys):
        self.keys = list(keys)
        self.step = 0.0
        for k in keys:
            setattr(self, k, 0.0)
        self.default_template = ' {}: {}'

    def update_template(self, num_decimal):
        self.default_template = ' {}: {' + ':.' + str(int(num_decimal)) + 'f}'

    def add_item(self, key, init_val=0.0):
        if hasattr(self, key):
            print('** ** ** Already contains: {}'.format(key))
        else:
            setattr(self, key, init_val)
            self.keys.append(key)

    def update(self, *values):
        for k, v in zip(self.keys, values):
            setattr(self, k, getattr(self, k) + v)
        self.step += 1
        pass

    def clear(self):
        for k in self.keys:
            setattr(self, k, 0.0)
        pass
        self.step = 0.0

    def get_metric(self):
        msgs = []
        flag = 0 >= self.step
        for k in self.keys:
            msgs.append(self.default_template.format(k, 0.0 if flag else (getattr(self, k) / self.step)))
        return ','.join(msgs)
    pass

