# -*- coding:utf-8 -*-
"""
notes:
    1. code used in kaggle
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import sys
import json
import random

template = '__label__{} {}'


def handle(labels_map, src_path, tgt_path):
    with open(src_path) as sf, open(tgt_path, 'wt', encoding='utf-8') as tf:
        for line in sf:
            line = line.strip()
            toks = line.split()
            label = labels_map[toks[0]]
            body = ' '.join(toks[1:])
            tf.write('{}\n'.format(template.format(label, body)))
        tf.flush()


if __name__ == '__main__':
    train_path = sys.argv[1]
    valid_path = sys.argv[2]
    save_train_path = sys.argv[3]
    save_valid_path = sys.argv[4]
    label_tokens = sys.argv[5]

    label_dict = {k: label_idx for label_idx, k in enumerate(label_tokens.split(','))}
    handle(label_dict, train_path, save_train_path)
    handle(label_dict, valid_path, save_valid_path)




"""

# -*- coding:utf-8 -*-
import os
template = '__label__{} {}'


def handle(labels_map, src_path, tgt_path):
    with open(src_path) as sf, open(tgt_path, 'wt', encoding='utf-8') as tf:
        for line in sf:
            line = line.strip()
            toks = line.split()
            label = labels_map[toks[0]]
            body = ' '.join(toks[1:])
            tf.write('{}\n'.format(template.format(label, body)))
        tf.flush()


if __name__ == '__main__':
#     train_path = sys.argv[1]
#     valid_path = sys.argv[2]
#     save_train_path = sys.argv[3]
#     save_valid_path = sys.argv[4]
#     label_tokens = sys.argv[5]
    label_tokens = "<sl0>,<sl1>"
    label_dict = {k: label_idx for label_idx,k in enumerate(label_tokens.split(','))}
    
    input_dir = "/kaggle/input/style-dataset/raw/yelp"
    save_dir = "/kaggle/working/yelp"
    train_path = os.path.join(input_dir, "train.src")
    valid_path = os.path.join(input_dir, "dev.src")
    test_path = os.path.join(input_dir, "test.src")
    save_train_path = os.path.join(save_dir, "train.txt")
    save_valid_path = os.path.join(save_dir, "valid.txt")
    save_test_path = os.path.join(save_dir, "test.txt")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    handle(label_dict, train_path, save_train_path)
    handle(label_dict, valid_path, save_valid_path)
    handle(label_dict, test_path, save_test_path)
    
    print("=" * 80)
    input_dir = "/kaggle/input/style-dataset/raw/imdb"
    save_dir = "/kaggle/working/imdb"
    train_path = os.path.join(input_dir, "train.src")
    valid_path = os.path.join(input_dir, "dev.src")
    test_path = os.path.join(input_dir, "test.src")
    save_train_path = os.path.join(save_dir, "train.txt")
    save_valid_path = os.path.join(save_dir, "valid.txt")
    save_test_path = os.path.join(save_dir, "test.txt")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    handle(label_dict, train_path, save_train_path)
    handle(label_dict, valid_path, save_valid_path)
    handle(label_dict, test_path, save_test_path)
    
    

"""
