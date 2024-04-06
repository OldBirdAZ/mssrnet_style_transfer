# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import sys
LABEL_TEMPLATE = '<sl{}>'
import random


def handle(in_paths, out_path):
    with open(out_path+'.src', 'wt', encoding='utf-8') as outf_src, open(out_path+'.tgt', 'wt', encoding='utf-8') as outf_tgt:
        all_datas = []
        for label, in_path in enumerate(in_paths):
            with open(in_path) as inf:
                for line in inf:
                    line = line.strip()
                    if len(line) > 0:
                        all_datas.append((label, line))
        random.shuffle(all_datas)
        for label, line in all_datas:
            outf_src.write('{}\n'.format(LABEL_TEMPLATE.format(label) + ' ' + line))
            outf_tgt.write('{}\n'.format(line))
        outf_src.flush()
        outf_tgt.flush()


if __name__ == '__main__':
    src_path = str(sys.argv[1])
    tgt_path = str(sys.argv[2])
    handle(src_path.split(','), tgt_path)







