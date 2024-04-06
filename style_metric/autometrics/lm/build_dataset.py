# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import sys
import random


def handle_datas(pos_path, neg_path, save_path):
    print('Reading datas ...')
    datas = []
    with open(pos_path, encoding='utf-8') as sf:
        for line in sf:
            datas.append(
                line.strip()
            )
    with open(neg_path, encoding='utf-8') as sf:
        for line in sf:
            datas.append(
                line.strip()
            )
    print('Loaded.')
    random.shuffle(datas)
    random.shuffle(datas)
    random.shuffle(datas)
    random.shuffle(datas)
    random.shuffle(datas)
    with open(save_path, 'wt', encoding='utf-8') as of:
        for line in datas:
            of.write('{}\n'.format(line))
        of.flush()
    print('Finished.')


if __name__ == '__main__':
    src_pos_path = sys.argv[1]
    src_neg_path = sys.argv[2]
    tgt_save_path = sys.argv[3]

    handle_datas(src_pos_path, src_neg_path, tgt_save_path)


"""

"""
