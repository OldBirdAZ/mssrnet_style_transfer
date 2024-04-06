#!/usr/bin/env bash

cls_path=${1}
ppl_path=${2}
p2n_test_path=${3}  # output of pos2neg path
n2p_test_path=${4}  # output of neg2pos path
pos_path=${5}
neg_path=${6}
p2n_ref=${7:-"NOSET"}
n2p_ref=${8:-"NOSET"}


echo "=========================================================================";
python cal_acc.py "${cls_path}" ${p2n_test_path} 0
python cal_acc.py "${cls_path}" ${n2p_test_path} 1
python cal_ppl.py "${ppl_path}" ${p2n_test_path}
python cal_ppl.py "${ppl_path}" ${n2p_test_path}
perl tools/multi-bleu.perl "${pos_path}" < ${p2n_test_path}
perl tools/multi-bleu.perl "${neg_path}" < ${n2p_test_path}
if [ "${p2n_ref}" != "NOSET" ];then
    echo "## ground truth bleu";
    perl tools/multi-bleu.perl "${p2n_ref}" < ${p2n_test_path};
    perl tools/multi-bleu.perl "${n2p_ref}" < ${n2p_test_path};
fi
echo "========================================================================="




