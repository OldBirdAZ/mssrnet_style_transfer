# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

from fasttext import load_model
import sys

LT = '__label__'
nLT = len(LT)


def _convert_label(pred_label):
    return int(pred_label[nLT:])


def infer():
    val_steps = 0.0
    preds = {}
    with open(data_path, encoding='utf-8') as sf:
        for line in sf:
            pred_labels, pred_probs = model.predict(line.strip())
            pred_label = _convert_label(pred_labels[0])
            val_steps += 1
            if pred_label not in preds:
                preds[pred_label] = 0
            preds[pred_label] += 1
    print('#### TEST RESULTS')
    # print('total num: {}'.format(val_steps))
    # print(preds)
    print(preds[label_id] / val_steps)


if __name__ == '__main__':
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    label_id = int(sys.argv[3])

    model = load_model(model_path)
    infer()




