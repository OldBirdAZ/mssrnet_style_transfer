# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function
import random
import torch


class ClsProxy:
    def __init__(self, cls_restore_path, device):
        all_cls_paths = cls_restore_path.split(',')
        self.classifiers = []
        for cls_path in all_cls_paths:
            classifier = torch.load(cls_path, map_location=device)
            classifier.to(device)
            classifier.eval()
            self.classifiers.append(classifier)
        self.n_cls = len(self.classifiers)

    def fetch_classifier(self):
        idx = random.randint(0, self.n_cls - 1)
        return self.classifiers[idx]


