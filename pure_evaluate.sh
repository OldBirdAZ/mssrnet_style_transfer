#!/usr/bin/env bash

cls_path=${1}
ppl_path=${2}
pos_path=${3}
neg_path=${4}
pos2neg_path=${5}
neg2pos_path=${6}
pos_ref=${7:-"NOSET"}
neg_ref=${8:-"NOSET"}

python cal_acc.py "${cls_path}" "${pos2neg_path}" 0
python cal_acc.py "${cls_path}" "${neg2pos_path}" 1
python cal_ppl.py "${ppl_path}" "${pos2neg_path}"
python cal_ppl.py "${ppl_path}" "${neg2pos_path}"

perl tools/multi-bleu.perl "${pos_path}" < "${pos2neg_path}"
perl tools/multi-bleu.perl "${neg_path}" < "${neg2pos_path}"

if [ "${pos_ref}" != "NOSET" ];then
    perl tools/multi-bleu.perl "${pos_ref}" < "${pos2neg_path}";
    perl tools/multi-bleu.perl "${neg_ref}" < "${neg2pos_path}";
fi

