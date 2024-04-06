# MSSRNet
The official implementation of paper, [MSSRNet:Manipulating Sequential Style Representation for Unsupervised Text Style Transfer](https://arxiv.org/pdf/2306.07994.pdf)

## Environment
You are recommended to use the requirement of packages listed in running-env/style_env.yaml. 
If you use conda, you can directly use our env config:
```shell
conda env create -n "YOUR-ENV-NAME" -f running-env/style_env.yaml
```


## Data Process
```shell
python build_opennmt_format.py \
YOUR-DATA-PATH/train.style-0,YOUR-DATA-PATH/train.style-1 \
YOUR-DATA-RESAVE-PATH/train

python build_opennmt_format.py \
YOUR-DATA-PATH/valid.style-0,YOUR-DATA-PATH/valid.style-1 \
YOUR-DATA-RESAVE-PATH/valid

python build_opennmt_format.py \
YOUR-DATA-PATH/test.style-0,YOUR-DATA-PATH/test.style-1 \
YOUR-DATA-RESAVE-PATH/test

python preprocess.py \
-share_vocab \
-train_src YOUR-DATA-RESAVE-PATH/train.src \
-train_tgt YOUR-DATA-RESAVE-PATH/train.tgt \
-valid_src YOUR-DATA-RESAVE-PATH/valid.src \
-valid_tgt YOUR-DATA-RESAVE-PATH/valid.tgt \
-save_data YOUR-DATASET-PATH/yelp/data
```


## Train
### Train Teacher Model
```shell
python train_teacher_model.py \
-data YOUR-DATASET-PATH/yelp/data \
-save_model YOUR-SAVE-PATH/teacher-model/model \
-n_layers_cls 3 -n_heads_cls 8 -dim_each_head_cls 32 \
-encoder_type transformer -decoder_type transformer -position_encoding \
-train_steps 200000 -dropout 0.1 \
-batch_size 16 -batch_type sents -normalization tokens  -accum_count 1 \
-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
-max_grad_norm 0 -param_init 0  -param_init_glorot \
-world_size 1 -gpu_ranks 0 \
-report_every 1000 -save_checkpoint_steps 10000 
```

### Train Style-Transfer Model
```shell
python train_wgan_wh.py \
-data YOUR-DATASET-PATH/yelp/data \
-save_model YOUR-SAVE-PATH/style-transfer-model/model \
-cls_restore_path YOUR-SAVE-PATH/teacher-model/model-m1.pt \
-n_layers_cls 3 -n_heads_cls 8 -dim_each_head_cls 32 \
-n_layers_sgenerator 3 -n_heads_sgenerator 8 -dim_each_head_sgenerator 32 \
-n_layers_tdiscr 1 -n_heads_tdiscr 8 -dim_each_head_tdiscr 32 \
-n_layers_sdiscr 3 -n_heads_sdiscr 8 -dim_each_head_sdiscr 64 \
-layers 12 -rnn_size 512 -word_vec_size 512 -transformer_ff 1024 -heads 8  \
-encoder_type transformer -decoder_type transformer -position_encoding \
-train_steps 160000 -dropout 0.1 \
-batch_size 80 -batch_type sents -normalization tokens  -accum_count 1 \
-optim adam -adam_beta1 0.5 -adam_beta2 0.9 -decay_method noam -warmup_steps 8000 -learning_rate 0.00004 \
-param_init 0  -param_init_glorot \
-world_size 1 -gpu_ranks 0 \
-report_every 200 -init_steps 2000 -save_checkpoint_steps 5000 -valid_steps 5000 \
-learn_disc_steps 5 -main_steps 1 -reconstruct_steps 25 -sr_noise_prob 0.0 \
-max_grad_norm=100 -use_cls_feedback -optimizer_class 'Adam' \
-seed 123 -use_gp_norm_seq=True -convert_gain2loss=False \
-dynamic_dec_type="use_ori_train_dec" -trans_beam_size 1 -without_disc_init \
-reconst_optimizer_class='Adam' \
-force_cycle_len=True \
-print_args=True \
--reconstruct_every_step=False \
--vloss_token_mode=False \
--sent_stop_aware \
--discr_sent_stop_aware=False --force_ap_text_stop=False \
--force_positive_gain_mode_s=sigmoid --force_positive_gain_mode_t=sigmoid \
--alpha_style_ap=1 --alpha_text_ap=1.0 --sr_loss_ratio=1.0 \
--ap_gen_self_style_text=True 
```
You can change the hyperparameter for your own dataset.


## Evaluation and Test
```shell
./batch_test.sh \
YOUR-TRAINED-MODEL-WORKING-DIR \
YOUR-TRAINED-MODEL-WORKING-DIR/test \
YOUR-PATH/autometrics/all_metrics/yelp_acc.bin \
YOUR-PATH/autometrics/all_metrics/yelp_ppl.bin \
YOUR-PATH/yelp/test.pos \
YOUR-PATH/yelp/test.neg \
32 \
5000 \
5 \
0 \
YOUR-PATH/yelp/reference.pos \
YOUR-PATH/yelp/reference.neg \
> YOUR-TRAINED-MODEL-WORKING-DIR/test/test-log.txt 2>&1
```
Here we use the fasttext and kenLM to automatically evaluate the performance of the trained model in terms of transfer accuracy and the fluency of the generated text.

You can train such evaluation model using the code: [style_metric](style_metric)



## Cite Our Paper
```
@inproceedings{yang2023mssrnet,
  title={MSSRNet: Manipulating Sequential Style Representation for Unsupervised Text Style Transfer},
  author={Yang, Yazheng and Zhao, Zhou and Liu, Qi},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={3022--3032},
  year={2023}
}
```



