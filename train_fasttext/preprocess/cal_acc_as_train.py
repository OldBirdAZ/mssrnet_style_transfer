# -*- coding:utf-8 -*-
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
    n_total = 0.0
    n_correct = 0.0
    with open(data_path, encoding='utf-8') as sf:
        for line in sf:
            toks = line.strip().split()
            ground_truth = toks[0]
            text = ' '.join(toks[1:])
            pred_labels, pred_probs = model.predict(text)
            if ground_truth == pred_labels[0]:
                n_correct += 1
            n_total += 1
    print('#### TEST RESULTS')
    # print('total num: {}'.format(val_steps))
    # print(preds)
    print(n_correct / n_total)


if __name__ == '__main__':
    model_path = sys.argv[1]
    data_path = sys.argv[2]

    model = load_model(model_path)
    infer()




