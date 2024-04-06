# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from fasttext import train_supervised
import os
import argparse
import streamtologger


def set_model_args(parser):
    # about model parameters
    parser.add_argument('--n_labels', default=2, type=int, help='')
    parser.add_argument('--emb_dim', default=100, type=int, help='')
    parser.add_argument('--wordNgrams', default=2, type=int, help='')
    parser.add_argument('--ws', default=5, type=int, help='')
    parser.add_argument('--minCount', default=5, type=int, help='')
    unsupervised_default = {
        'model': "skipgram",
        'lr': 0.05,
        'dim': 100,
        'ws': 5,
        'epoch': 5,
        'minCount': 5,
        'minCountLabel': 0,
        'minn': 3,
        'maxn': 6,
        'neg': 5,
        'wordNgrams': 1,
        'loss': "ns",
        'bucket': 2000000,
        # 'thread': multiprocessing.cpu_count() - 1,
        'lrUpdateRate': 100,
        't': 1e-4,
        'label': "__label__",
        'verbose': 2,
        'pretrainedVectors': "",
    }
    pass


def set_train_args():
    parser = argparse.ArgumentParser(description='Train Config')
    # Where to find data
    parser.add_argument(
        '--data_base_path',
        default=None,
        help='base Path of data.', required=True)

    parser.add_argument('--workdir', default='../outputs/m1', help='')
    parser.add_argument('--lr', default=0.05, type=float, help='')
    parser.add_argument('--n_epochs', default=100, type=int, help='')
    set_model_args(parser)

    args = parser.parse_args()
    save_path = os.path.join(args.workdir, 'saved')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.save_path = save_path

    log_file_path = os.path.join(args.workdir, 'log_train.txt')
    streamtologger.redirect(target=log_file_path, append=True,
                            header_format="[{timestamp:%Y-%m-%d %H:%M:%S} - {level:5}] ")
    return args


def main():
    model = train_supervised(
        input=options.data_base_path,
        dim=options.emb_dim,
        ws=options.ws,
        epoch=options.n_epochs, lr=options.lr,
        wordNgrams=options.wordNgrams, minCount=options.minCount, loss="softmax")

    model.save_model(os.path.join(options.save_path, 'model.bin'))


if __name__ == '__main__':
    options = set_train_args()
    main()

"""

"""


