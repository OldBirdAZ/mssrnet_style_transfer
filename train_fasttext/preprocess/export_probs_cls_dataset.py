# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from fasttext import load_model
import sys
import json
import os

LT = '__label__'
nLT = len(LT)


def _convert_label(pred_label):
    return int(pred_label[nLT:])


def handle(src_path, tgt_path):
    with open(src_path, encoding='utf-8') as sf, open(tgt_path, 'wt', encoding='utf-8') as tf:
        for line in sf:
            toks = line.strip().split()
            ground_truth = toks[0]
            text = ' '.join(toks[1:])
            pred_labels, pred_probs = model.predict(text)
            pred_label = _convert_label(pred_labels[0])
            gt_label = _convert_label(ground_truth)
            pred_prob = pred_probs[0]
            pred_prob = 1.0 if pred_prob > 1.0 else pred_prob
            example = {
                'text': text,
                'label': gt_label,
                'pred_label_{}'.format(pred_label): pred_prob,
                'pred_label_{}'.format(int(1 - pred_label)): 1.0 - pred_prob
            }
            tf.write('{}\n'.format(json.dumps(example)))
        tf.flush()
    print('#### FINISHED:: {}'.format(src_path))


if __name__ == '__main__':
    model_path = sys.argv[1]
    src_dir = sys.argv[2]
    tgt_dir = sys.argv[3]
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    model = load_model(model_path)
    handle(os.path.join(src_dir, 'train.txt'), os.path.join(tgt_dir, 'train.json'))
    handle(os.path.join(src_dir, 'valid.txt'), os.path.join(tgt_dir, 'valid.json'))
    handle(os.path.join(src_dir, 'test.txt'), os.path.join(tgt_dir, 'test.json'))



