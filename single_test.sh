#!/usr/bin/env bash

model_dir=${1}
save_dir=${2}
cls_path=${3}
ppl_path=${4}
pos_path=${5}
neg_path=${6}
cur_step=${7:-5000}
beam=${8:-10}
gpu=${9:-0}
pos_ref=${10:-"NOSET"}
neg_ref=${11:-"NOSET"}

mkdir -p ${save_dir}

echo "=========================================================================";
echo "CKPT :: ${cur_step}";
python translate_wh.py \
-src "${pos_path}" \
-tgt "${pos_path}" \
-output "${save_dir}/test_out_p2n_b${beam}_${cur_step}.txt" \
-desired_style 0 \
-model_path "${model_dir}/md_${cur_step}.pt" \
-vocab_opt_path "${model_dir}/vocab_model_opt.pt" \
-replace_unk -gpu "${gpu}" -beam_size "${beam}";
python translate_wh.py \
-src "${neg_path}" \
-tgt "${neg_path}" \
-output "${save_dir}/test_out_n2p_b${beam}_${cur_step}.txt" \
-desired_style 1 \
-model_path "${model_dir}/md_${cur_step}.pt" \
-vocab_opt_path "${model_dir}/vocab_model_opt.pt" \
-replace_unk -gpu "${gpu}" -beam_size "${beam}";
echo "========================================================================="


echo "=========================================================================";
echo "CKPT :: ${cur_step}";
python cal_acc.py "${cls_path}" "${save_dir}/test_out_p2n_b${beam}_${cur_step}.txt" 0
python cal_acc.py "${cls_path}" "${save_dir}/test_out_n2p_b${beam}_${cur_step}.txt" 1
python cal_ppl.py "${ppl_path}" "${save_dir}/test_out_p2n_b${beam}_${cur_step}.txt"
python cal_ppl.py "${ppl_path}" "${save_dir}/test_out_n2p_b${beam}_${cur_step}.txt"
perl tools/multi-bleu.perl "${pos_path}" < "${save_dir}/test_out_p2n_b${beam}_${cur_step}.txt"
perl tools/multi-bleu.perl "${neg_path}" < "${save_dir}/test_out_n2p_b${beam}_${cur_step}.txt"
if [ "${pos_ref}" != "NOSET" ];then
    echo "## ground truth bleu";
    perl tools/multi-bleu.perl "${pos_ref}" < "${save_dir}/test_out_p2n_b${beam}_${cur_step}.txt";
    perl tools/multi-bleu.perl "${neg_ref}" < "${save_dir}/test_out_n2p_b${beam}_${cur_step}.txt";
fi
echo "========================================================================="




