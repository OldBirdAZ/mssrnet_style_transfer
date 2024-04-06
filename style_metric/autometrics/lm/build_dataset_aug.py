# -*- coding:utf-8 -*-
"""
notes:
    1. augment version: support for multiple files
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import sys
import random


def handle_datas(src_files, save_path):
    print('Reading datas ...')
    datas = []
    for src_file in src_files:
        with open(src_file, encoding='utf-8') as sf:
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
    src_path = str(sys.argv[1])
    tgt_path = str(sys.argv[2])
    random.seed(123)

    handle_datas(src_path.split(','), tgt_path)


"""

"""
