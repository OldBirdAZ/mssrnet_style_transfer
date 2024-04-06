# -*- coding:utf-8 -*-
"""
notes:
    1. augment version: support for multiple classes (multiple styles)
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import sys
import random
"""
pos <--> 1
neg <--> 0
"""
template = '__label__{} {}'


def handle_datas(in_paths, save_path):
    print('Reading datas ...')
    all_data = []
    for label, in_path in enumerate(in_paths):
        with open(in_path, encoding='utf-8') as sf:
            for line in sf:
                line = line.strip()
                if len(line) > 0:
                    all_data.append(template.format(label, line))
    random.shuffle(all_data)
    random.shuffle(all_data)
    random.shuffle(all_data)
    random.shuffle(all_data)

    with open(save_path, 'wt', encoding='utf-8') as of:
        for line in all_data:
            of.write('{}\n'.format(line))
        of.flush()
    print('Finished.')


if __name__ == '__main__':
    src_path = str(sys.argv[1])
    tgt_path = str(sys.argv[2])

    handle_datas(src_path.split(','), tgt_path)




