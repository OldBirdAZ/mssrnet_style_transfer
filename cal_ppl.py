import kenlm
import math
import sys
BASE = 10


class PPLEvaluator(object):
    def __init__(self, ppl_model_path):
        self.ppl_model = kenlm.Model(ppl_model_path)

    def cal_ppl(self, texts_transferred):
        """
        :param texts_transferred: list of sentences
        :return:
        """
        total_sum = 0
        length = 0
        for i, line in enumerate(texts_transferred):
            length += len(line.split())
            score = self.ppl_model.score(line)
            total_sum += score
        if 0 == length:
            return 0.0
        return math.pow(BASE, -total_sum / length)


if __name__ == '__main__':
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    datas = []
    with open(data_path, encoding='utf-8') as sf:
        for sent in sf:
            sent = sent.strip()
            datas.append(sent)
    evaluator = PPLEvaluator(model_path)
    print(evaluator.cal_ppl(datas))




