"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import torch

import onmt.utils
from onmt.utils.logging import logger
from torch.nn import functional as F
import random
from torch.nn.utils import clip_grad_norm_
from torch import nn
from onmt.info_logging import InfoLogger
from onmt.auto_occupy import auto_occupy_gpu_error_friendly2
from onmt.tensor_utils import word_drop
from torch import optim
from onmt.lr_adapter import LrAdapter
from torch.autograd import grad as grad_fn
from onmt.utils.tensor_board_wrapper import TensorBoardWrapper
import os
from onmt.train_util import VocabLossFn
from onmt.train_util import UniformValChangeScheduler


def build_trainer(
        opt, device_id,
        model, style_generator, style_discriminator, text_discriminator, cls_proxy,
        fields):
    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    tgt_field = dict(fields)["tgt"].base_field
    tgt_vocab = tgt_field.vocab
    eos_idx = tgt_vocab.stoi[tgt_field.eos_token]
    pad_idx = tgt_vocab.stoi[tgt_field.pad_token]
    bos_idx = tgt_vocab.stoi[tgt_field.init_token]
    unk_idx = tgt_vocab.stoi[tgt_field.unk_token]
    tgt_vocab_len = len(tgt_vocab)

    trainer = Trainer(
        model, style_generator, style_discriminator, text_discriminator, cls_proxy,
        pad_idx, unk_idx, bos_idx, eos_idx, tgt_vocab,
        opt, fields, train_loss,
        trunc_size=trunc_size, shard_size=shard_size, norm_method=norm_method,
        accum_count=accum_count, accum_steps=accum_steps,
        with_align=True if opt.lambda_align > 0 else False,
        model_dtype=opt.model_dtype,
        earlystopper=earlystopper,
        dropout=dropout,
        dropout_steps=dropout_steps,
        vocab_size=tgt_vocab_len
    )
    return trainer


class Trainer(object):
    def __init__(self,
                 model,
                 style_generator,
                 style_discriminator, text_discriminator,
                 cls_proxy,
                 pad_idx, unk_idx, bos_idx, eos_idx, vocab,
                 opt, fields, train_loss,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 with_align=False,
                 model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0],
                 vocab_size=None
                 ):
        # Basic attributes.
        self.model = model
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.with_align = with_align
        self.moving_average = None
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.train_loss = train_loss
        self.sr_noise_prob = opt.sr_noise_prob
        self.vocab_size = vocab_size
        self.use_recycle = opt.use_recycle
        self.no_recycle_to_fw = opt.no_recycle_to_fw
        self.cycle_loss_ratio = opt.cycle_loss_ratio
        self.sr_loss_ratio = opt.sr_loss_ratio
        self.cls_loss_ratio = opt.cls_loss_ratio
        self.style_train_steps = opt.style_train_steps
        occupy_ram = opt.occupy_ram
        if occupy_ram:
            auto_occupy_gpu_error_friendly2(0.9)

        self.opt = opt
        num_styles = opt.num_styles
        self.num_styles = num_styles
        self.style_generator = style_generator
        self.style_discriminator = style_discriminator
        self.text_discriminator = text_discriminator
        self.force_positive_gain = opt.force_positive_gain
        self.force_positive_gain_mode_t = opt.force_positive_gain_mode_t
        self.force_positive_gain_mode_s = opt.force_positive_gain_mode_s
        self.force_positive_gain_ratio = opt.force_positive_gain_ratio

        # shadow
        import copy
        self.shadow_model = copy.deepcopy(self.model)
        self.shadow_style_generator = copy.deepcopy(self.style_generator)
        self.shadow_model.eval()
        self.shadow_style_generator.eval()

        self.cls_proxy = cls_proxy
        self.use_gp_norm_seq = opt.use_gp_norm_seq
        self.pad_idx, self.unk_idx, self.bos_idx, self.eos_idx = pad_idx, unk_idx, bos_idx, eos_idx
        # self.dot_idx = vocab.stoi['.']
        # self.excalmatory_idx = vocab.stoi['!']
        # self.interrogation_idx = vocab.stoi['?']
        logger.info("Building TrainTranslator ... ")
        from onmt.train_and_translate import TrainTranslator
        self.train_translator = TrainTranslator(
            model=model, style_generator=style_generator,
            fields=fields,
            opt=opt
        )
        self.report_every = self.opt.report_every
        self.max_extra_dec_steps = 10

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        self.init_steps = opt.init_steps
        self.learn_disc_steps = opt.learn_disc_steps
        self.main_steps = opt.main_steps
        # Set model in training mode.
        self.model.train()
        self.style_generator.train()
        self.style_discriminator.train()
        self.text_discriminator.train()

        self.sent_stop_aware = opt.sent_stop_aware
        self.vloss_token_mode = opt.vloss_token_mode
        self.vocab_loss_fn = VocabLossFn(token_mode=self.vloss_token_mode, ignore_index=pad_idx, all_tokens_aware=self.sent_stop_aware)
        # self.vocab_loss_fn = nn.NLLLoss(ignore_index=pad_idx)
        self.label_loss_fn = nn.NLLLoss()
        self.total_train_steps = 0
        if self.opt.restore_from is not None:
            assert self.opt.restore_step is not None, "must set restore_step while restore from ckpt!"
            self.total_train_steps = self.opt.restore_step
        self.total_train_reconstruct_steps = 0
        self.total_train_recycle_steps = 0
        self.reconstruct_interval_steps = opt.reconstruct_steps
        self.recycle_steps = opt.recycle_steps
        self.use_cls_feedback = opt.use_cls_feedback
        self.sent_stop_aware = opt.sent_stop_aware
        # if self.sent_stop_aware:
        #     self.vocab_loss_fn = nn.NLLLoss()
        self.without_disc_init = opt.without_disc_init
        self.without_stl_penalty = opt.without_stl_penalty
        self.convert_gain2loss = opt.convert_gain2loss
        self.force_cycle_len = opt.force_cycle_len

        OptimizerClass = getattr(optim, opt.optimizer_class, 'SGD')
        assert issubclass(OptimizerClass, optim.Optimizer)
        if 'Adam' == opt.optimizer_class:
            self.style_generator_optim = optim.Adam(
                self.style_generator.parameters(),
                lr=opt.learning_rate,
                betas=(opt.adam_beta1, opt.adam_beta2)
            )
            self.style_discriminator_optim = optim.Adam(
                self.style_discriminator.parameters(),
                lr=opt.learning_rate,
                betas=(opt.adam_beta1, opt.adam_beta2)
            )
            self.text_discriminator_optim = optim.Adam(
                self.text_discriminator.parameters(),
                lr=opt.learning_rate,
                betas=(opt.adam_beta1, opt.adam_beta2)
            )
            self.model_optim = optim.Adam(
                self.model.parameters(),
                lr=opt.learning_rate,
                betas=(opt.adam_beta1, opt.adam_beta2)
            )
        else:
            self.style_generator_optim = OptimizerClass(
                self.style_generator.parameters(),
                lr=opt.learning_rate
            )
            self.style_discriminator_optim = OptimizerClass(
                self.style_discriminator.parameters(),
                lr=opt.learning_rate
            )
            self.text_discriminator_optim = OptimizerClass(
                self.text_discriminator.parameters(),
                lr=opt.learning_rate
            )
            self.model_optim = OptimizerClass(
                self.model.parameters(),
                lr=opt.learning_rate
            )
        if opt.reconst_optimizer_class == opt.optimizer_class:
            self.reconst_model_optim = self.model_optim
            self.reconst_style_generator_optim = self.style_generator_optim
        else:
            OptimizerClass = getattr(optim, opt.reconst_optimizer_class, 'SGD')
            assert issubclass(OptimizerClass, optim.Optimizer)
            if 'Adam' == opt.reconst_optimizer_class:
                self.reconst_style_generator_optim = optim.Adam(
                    self.style_generator.parameters(),
                    lr=opt.learning_rate,
                    betas=(opt.adam_beta1, opt.adam_beta2)
                )
                self.reconst_model_optim = optim.Adam(
                    self.model.parameters(),
                    lr=opt.learning_rate,
                    betas=(opt.adam_beta1, opt.adam_beta2)
                )
            else:
                self.reconst_style_generator_optim = OptimizerClass(
                    self.style_generator.parameters(),
                    lr=opt.learning_rate
                )
                self.reconst_model_optim = OptimizerClass(
                    self.model.parameters(),
                    lr=opt.learning_rate
                )
        # print("*" * 100)
        # for name, param in self.model.named_parameters():
        #     print(name)
        # print("*" * 100)
        # print("-" * 100)
        # for name, param in self.style_generator.named_parameters():
        #     print(name)
        # print("-" * 100)
        self.force_ap_text_stop = opt.force_ap_text_stop
        if self.force_ap_text_stop:
            self.force_apts_loss_fn = VocabLossFn(token_mode=True, ignore_index=None)
        self.ap_gen_self_style_text = opt.ap_gen_self_style_text

        self.max_grad_norm = opt.max_grad_norm
        self.init_logger = InfoLogger(
            'INIT_SL', 'INIT_VL'
        )
        self.init_logger.update_template(4)

        # *_cost: smaller is better, *_Wgap: bigger is better (for discriminator)
        disc_log_items = [
            'TR_gain', 'TF_gain', 'T_gp', 'T_cost', 'T_Wgap',
            'SR_gain', 'SF_gain', 'S_gp', 'S_cost', 'S_Wgap'
        ]
        if self.convert_gain2loss:
            disc_log_items = [x_.replace('gain', 'loss') for x_ in disc_log_items]
        self.disc_logger = InfoLogger(*disc_log_items)
        self.disc_logger.update_template(4)

        main_log_items = [
            'AT_gain',
            'AT_acc',
            'AS_gain',
            # 'ASC_gain'
        ]
        if self.convert_gain2loss:
            main_log_items = [x_.replace('gain', 'loss') for x_ in main_log_items]
        if self.force_ap_text_stop:
            main_log_items.append('AT_stop_loss')
        if self.ap_gen_self_style_text:
            main_log_items.append('T_selfstyle_vloss')
        self.main_logger = InfoLogger(
            *main_log_items
        )
        self.main_logger.update_template(4)

        reconstruct_log_items = [
            'T_CONSTRUCT_R_VL', 'T_CONSTRUCT_F_VL'
        ]
        if self.use_cls_feedback:
            reconstruct_log_items.append('T_CLS_loss')
            reconstruct_log_items.append('T_CLS_acc')
            reconstruct_log_items.append('S_CLS_loss')
            reconstruct_log_items.append('S_CLS_acc')
            reconstruct_log_items.append('SCur_CLS_loss')
            reconstruct_log_items.append('SCur_CLS_acc')
        if not self.without_stl_penalty:
            reconstruct_log_items.append('S_GAP_loss')
        self.reconstruct_logger = InfoLogger(
            *reconstruct_log_items
        )
        self.reconstruct_logger.update_template(4)

        self.recycle_logger = InfoLogger(
            'RECYCLE_LOSS'
        )
        self.recycle_logger.update_template(4)

        self.save_path = opt.save_model
        self.lr_adapter = LrAdapter(opt)
        self.dynamic_lr = opt.dynamic_lr
        self.lr = opt.learning_rate
        self.without_bp2stlgen = opt.without_bp2stlgen
        self.n_show_gen_texts = opt.n_show_gen_texts

        gpu_id = None
        from onmt.utils.misc import use_gpu
        gpu = use_gpu(opt)
        device = torch.device("cpu")
        if gpu and gpu_id is not None:
            device = torch.device("cuda", gpu_id)
        elif gpu and not gpu_id:
            device = torch.device("cuda")
        elif not gpu:
            device = torch.device("cpu")
        self.one_tensor = torch.ones([]).to(device)
        self.mone_tensor = -1 * torch.ones([]).to(device)
        if self.convert_gain2loss:
            logger.info("**** Converting gain mode to loss mode ****")
            ttemp = self.one_tensor
            self.one_tensor = self.mone_tensor
            self.mone_tensor = ttemp
        self.peak_main_steps = opt.peak_main_steps
        self.start_peak_step = opt.start_peak_step
        self.style_ap_scale = opt.style_ap_scale
        self.text_ap_scale = opt.text_ap_scale
        self.style_ap_mone_tensor = self.mone_tensor * self.style_ap_scale
        self.text_ap_mone_tensor = self.mone_tensor * self.text_ap_scale
        save_model_path = os.path.abspath(opt.save_model)
        model_dirname = os.path.dirname(save_model_path)
        tensorboard_dir = os.path.join(model_dirname, 'tensorboard')
        self.tensorboard_writer = TensorBoardWrapper(tensorboard_dir)
        assert opt.dec_temperature_changeval <= 0
        self.temperature_schedule = UniformValChangeScheduler(
            start_var=opt.dec_temperature_startval, min_val=1.0, max_val=None,
            interval_step=opt.dec_temperature_changeepoch, interval_gap=opt.dec_temperature_changeval, continuous_mode=True
        )
        self.d_g_use_temperature = opt.d_g_use_temperature
        if self.d_g_use_temperature:
            self.train_translator.dec_temperature = self.temperature_schedule.get_val()
        self.discr_sent_stop_aware = opt.discr_sent_stop_aware
        # [1, 1, 1]
        self.pad_tensor = torch.LongTensor([[[self.pad_idx]]]).to(device)

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.total_train_steps + 1)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.pad_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.total_train_steps)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _postprocess_discriminator_gain(self, logits, force_positive_gain_mode):
        if 'normal' == force_positive_gain_mode:
            return logits
        elif 'exp' == force_positive_gain_mode:
            return torch.exp(logits)
        elif 'sigmoid' == force_positive_gain_mode:
            return torch.sigmoid(logits) * self.force_positive_gain_ratio
        elif 'sigmoid0' == force_positive_gain_mode:
            return (torch.sigmoid(logits) - 0.0001) * self.force_positive_gain_ratio
        elif 'relu' == force_positive_gain_mode:
            return torch.relu(logits)
        elif 'relu0' == force_positive_gain_mode:
            return torch.relu(logits - 0.001)
        else:
            raise ValueError('no support force_positive_gain_mode: {}'.format(force_positive_gain_mode))

    def _call_text_discriminator(self, real_src, real_src_lengths):
        logits = self.text_discriminator(real_src, real_src_lengths)
        results = self._postprocess_discriminator_gain(logits, self.force_positive_gain_mode_t)
        if self.opt.fuse_protect:
            results = self.opt.fuse_text_min + F.relu(logits) - F.relu(logits - self.opt.fuse_text_max)
        return results

    def _call_style_discriminator(self, real_style_represents_t, real_src_lengths):
        logits = self.style_discriminator(real_style_represents_t, real_src_lengths)
        results = self._postprocess_discriminator_gain(logits, self.force_positive_gain_mode_s)
        if self.opt.fuse_protect:
            results = self.opt.fuse_style_min + F.relu(logits) - F.relu(logits - self.opt.fuse_style_max)
        return results

    def unpack_batch(self, batch):
        """
        :param batch:
        :returns:
            labels: LongTensor [N]
            labels_list: List
            real_src: LongTensor [max_len, N, 1]
            real_src_lengths: LongTensor [N]
            vocab_target: LongTensor [max_len, N, 1]
        """
        src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        real_src = src[1:]
        real_src_lengths = src_lengths - 1
        labels = src[0, :, 0]
        labels_list = labels.tolist()
        labels_list = [self.train_translator.convert_label_idx(lvidx) for lvidx in labels_list]
        labels = torch.LongTensor(labels_list).type_as(labels)
        # vocab_target_ori = batch.tgt[1:, :, 0]
        # # vocab_target = vocab_target_ori.contiguous().view(-1)
        # vocab_target = vocab_target_ori
        dec_out = torch.cat([batch.tgt[1:], self.pad_tensor.repeat(1, batch.tgt.size(1), 1)])
        vocab_target = dec_out
        return labels, labels_list, real_src, real_src_lengths, vocab_target

    def cal_style_gap(self, real_style_represents, gen_style_represents, real_src_lengths):
        style_gap = real_style_represents - gen_style_represents
        gap2 = style_gap * style_gap
        # [batch_size, seq_length]
        gap3 = torch.sum(gap2, 2).transpose(0, 1)
        mask = sequence_mask(real_src_lengths, max_len=real_style_represents.size(0))
        valid_gap = gap3[mask]
        return torch.mean(valid_gap)

    def init_launch(self, batch):
        self.style_generator.train()
        self.model.train()
        self.reconst_style_generator_optim.zero_grad()
        self.reconst_model_optim.zero_grad()

        labels, labels_list, real_src, real_src_lengths, vocab_target = self.unpack_batch(batch)
        vocab_target = vocab_target.contiguous().view(-1)
        batch_size = len(labels_list)

        real_style_represents = self._gen_real_style_represents(real_src, real_src_lengths)
        gen_style_represents = self._gen_style_represents(labels, real_src, real_src_lengths)
        # style_gap = real_style_represents - gen_style_represents
        # init_style_loss = torch.mean(style_gap * style_gap)
        init_style_loss = self.cal_style_gap(real_style_represents, gen_style_represents, real_src_lengths)
        init_style_loss_val = init_style_loss.item()
        init_style_loss.backward()

        # real_style_represents = self._gen_real_style_represents(real_src, real_src_lengths)
        noised_real_src = word_drop(real_src, real_src_lengths, self.sr_noise_prob, self.unk_idx)
        vocab_logits = self._self_reconstruct(noised_real_src, real_src_lengths, batch.tgt, real_style_represents)
        # vocab_loss = self.vocab_loss_fn(vocab_logits.view(-1, vocab_logits.size(-1)), vocab_target)
        vocab_loss = self.vocab_loss_fn.calculate_loss(vocab_logits.view(-1, vocab_logits.size(-1)), vocab_target, batch_size=batch_size)
        sr_real_vl_val = vocab_loss.item()
        vocab_loss.backward()

        if self.dynamic_lr:
            self.lr_adapter.apply_to_optimizer(self.reconst_style_generator_optim, self.lr)
            self.lr_adapter.apply_to_optimizer(self.reconst_model_optim, self.lr)
        if self.max_grad_norm > 0:
            clip_grad_norm_(self.style_generator.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.reconst_style_generator_optim.step()
        self.reconst_model_optim.step()

        self.init_logger.update(init_style_loss_val, sr_real_vl_val)

        if not self.without_disc_init:
            self.train_discriminator(batch)

    def train_discriminator(self, batch):
        # train style disc
        self.style_generator.eval()
        self.model.eval()
        self.style_discriminator.train()
        self.text_discriminator.train()
        self.style_discriminator_optim.zero_grad()
        self.text_discriminator_optim.zero_grad()

        labels, labels_list, real_src, real_src_lengths, vocab_target = self.unpack_batch(batch)
        log_vals = []
        real_text_lengths = (real_src_lengths + 1).detach()
        if self.discr_sent_stop_aware:
            real_text_lens_stop_aware = (torch.ones_like(real_src_lengths) * vocab_target.size(0)).detach()
            real_text_lengths = real_text_lens_stop_aware

        # ========================================================================
        # train text discriminator
        # ========================================================================
        # REAL
        real_logits = self._call_text_discriminator(vocab_target, real_text_lengths)
        gain_t_real = torch.mean(cal_disc_loss(real_logits, labels_list))
        gain_t_real_val = gain_t_real.item()
        gain_t_real.backward(self.mone_tensor)
        # FAKE
        with torch.no_grad():
            gen_s_style_represents = self._gen_style_represents(labels, real_src, real_src_lengths)
            gen_s_vocab_probs, gen_s_seq_lens = self._dynamic_decoding(
                real_src, real_src_lengths, gen_s_style_represents, max_dec_steps=vocab_target.size(0)
                , no_need_bp=True
            )
        fake_text_lengths = real_text_lengths
        if self.discr_sent_stop_aware:
            fake_text_lengths = real_text_lens_stop_aware
        fake_logits = self._call_text_discriminator(gen_s_vocab_probs, fake_text_lengths)
        gain_t_fake = torch.mean(cal_disc_loss(fake_logits, labels_list))
        gain_t_fake_val = gain_t_fake.item()
        gain_t_fake.backward(self.one_tensor)
        # batch first
        with torch.no_grad():
            gen_s_vocab_probs_t = gen_s_vocab_probs.transpose(0, 1)
            real_src_t_onehot = convert_sent_ids_to_onehot(vocab_target.transpose(0, 1).squeeze(2), gen_s_vocab_probs_t)
        t_gradient_penalty = penalize_gen_text_grad(
            self._call_text_discriminator, real_src_t_onehot, gen_s_vocab_probs_t,
            real_text_lengths, labels_list, gp_norm_seq=self.use_gp_norm_seq,
            k=self.opt.wgan_text_k, lamb=self.opt.wgan_text_lamb
        )
        t_gradient_penalty_val = t_gradient_penalty.item()
        t_gradient_penalty.backward()
        if not self.convert_gain2loss:
            D_cost_t = gain_t_fake_val - gain_t_real_val + t_gradient_penalty_val
            Wasserstein_D_t = gain_t_real_val - gain_t_fake_val
        else:
            D_cost_t = gain_t_real_val - gain_t_fake_val + t_gradient_penalty_val
            Wasserstein_D_t = gain_t_fake_val - gain_t_real_val
        log_vals.append(gain_t_real_val)
        log_vals.append(gain_t_fake_val)
        log_vals.append(t_gradient_penalty_val)
        log_vals.append(D_cost_t)
        log_vals.append(Wasserstein_D_t)
        self.tensorboard_writer.put_scalar('D/T/real_gain', gain_t_real_val, self.total_train_steps)
        self.tensorboard_writer.put_scalar('D/T/fake_gain', gain_t_fake_val, self.total_train_steps)
        self.tensorboard_writer.put_scalar('D/T/gp', t_gradient_penalty_val, self.total_train_steps)
        self.tensorboard_writer.put_scalar('D/T/cost', D_cost_t, self.total_train_steps)
        self.tensorboard_writer.put_scalar('D/T/w_distance', Wasserstein_D_t, self.total_train_steps)

        # ========================================================================
        # train style discriminator
        # ========================================================================
        if self.style_train_steps < 1 or self.total_train_steps < self.style_train_steps:
            # REAL
            with torch.no_grad():
                real_style_represents = self._gen_real_style_represents(real_src, real_src_lengths)
                real_style_represents_t = real_style_represents.transpose(0, 1)
            # [N, n_styles]
            real_logits = self._call_style_discriminator(real_style_represents_t, real_src_lengths)
            gain_s_real = torch.mean(cal_disc_loss(real_logits, labels_list))
            gain_s_real_val = gain_s_real.item()
            gain_s_real.backward(self.mone_tensor)
            # FAKE
            with torch.no_grad():
                gen_style_represents = self._gen_style_represents(labels, real_src, real_src_lengths)
                gen_style_represents_t = gen_style_represents.transpose(0, 1)
            fake_logits = self._call_style_discriminator(gen_style_represents_t, real_src_lengths)
            gain_s_fake = torch.mean(cal_disc_loss(fake_logits, labels_list))
            gain_s_fake_val = gain_s_fake.item()
            gain_s_fake.backward(self.one_tensor)

            # gradient penalty
            s_gradient_penalty = penalize_gen_style_grad(
                self._call_style_discriminator, real_style_represents_t, gen_style_represents_t,
                real_src_lengths, labels_list, gp_norm_seq=self.use_gp_norm_seq,
                k=self.opt.wgan_style_k, lamb=self.opt.wgan_style_lamb
            )
            s_gradient_penalty_val = s_gradient_penalty.item()
            s_gradient_penalty.backward()
            if not self.convert_gain2loss:
                D_cost_s = gain_s_fake_val - gain_s_real_val + s_gradient_penalty_val
                Wasserstein_D_s = gain_s_real_val - gain_s_fake_val
            else:
                D_cost_s = gain_s_real_val - gain_s_fake_val + s_gradient_penalty_val
                Wasserstein_D_s = gain_s_fake_val - gain_s_real_val
            log_vals.append(gain_s_real_val)
            log_vals.append(gain_s_fake_val)
            log_vals.append(s_gradient_penalty_val)
            log_vals.append(D_cost_s)
            log_vals.append(Wasserstein_D_s)
            self.tensorboard_writer.put_scalar('D/S/real_gain', gain_s_real_val, self.total_train_steps)
            self.tensorboard_writer.put_scalar('D/S/fake_gain', gain_s_fake_val, self.total_train_steps)
            self.tensorboard_writer.put_scalar('D/S/gp', s_gradient_penalty_val, self.total_train_steps)
            self.tensorboard_writer.put_scalar('D/S/cost', D_cost_s, self.total_train_steps)
            self.tensorboard_writer.put_scalar('D/S/w_distance', Wasserstein_D_s, self.total_train_steps)
        else:
            log_vals += [0, 0, 0, 0]

        if self.dynamic_lr:
            self.lr_adapter.apply_to_optimizer(self.style_discriminator_optim, self.lr)
            self.lr_adapter.apply_to_optimizer(self.text_discriminator_optim, self.lr)
        if self.max_grad_norm > 0:
            clip_grad_norm_(self.style_discriminator.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.text_discriminator.parameters(), self.max_grad_norm)
        if self.style_train_steps < 1 or self.total_train_steps < self.style_train_steps:
            self.style_discriminator_optim.step()
        self.text_discriminator_optim.step()
        self.disc_logger.update(*log_vals)

    def _gen_real_style_represents(self, real_src, real_src_lengths):
        """ notice: remove the first token (style token)"""
        classifier = self.cls_proxy.classifiers[0]

        src, src_lengths = real_src, real_src_lengths
        with torch.no_grad():
            # [seq_len, batch_size, emb_size]
            _, memory_bank, _ = classifier.encoder(src, src_lengths)
            out = memory_bank.detach()
        return out

    def _gen_style_represents(self, desired_style, real_src, real_src_lengths):
        """ notice: remove the first token (style token)"""
        style_represents = self.style_generator(desired_style, real_src, real_src_lengths)
        # [seq_len, N, dim]
        return style_represents

    def _self_reconstruct(self, real_src, real_src_lengths, tgt, style_represents):
        # dec_in = tgt[:-1]  # exclude last target from inputs
        dec_in = tgt
        dec_out = torch.cat([tgt[1:], self.pad_tensor.repeat(1, tgt.size(1), 1)])
        enc_state, memory_bank, lengths = self.model.encoder(real_src, real_src_lengths)
        new_memory_bank = self.model.bond_layer(memory_bank, style_represents)
        self.model.decoder.init_state(real_src, new_memory_bank, enc_state)
        dec_out, attns = self.model.decoder(
            dec_in, new_memory_bank, memory_lengths=lengths, with_align=self.with_align
        )
        # vocab_logits = self.model.generator(dec_out)
        n_g_layers = len(self.model.generator)
        # [1, N, vocab_size]
        probs = dec_out
        for i in range(n_g_layers - 1):
            probs = self.model.generator[i](probs)
        vocab_logits = self.model.generator[-1](probs / self.temperature_schedule.get_val())

        return vocab_logits

    def _dynamic_decoding(
            self, real_src, real_src_lengths, style_represents
            , max_dec_steps=None, return_ids_tensor=False
            , no_need_bp=False
    ):
        """
        :param real_src:
        :param real_src_lengths:
        :param style_represents:
        :param max_dec_steps:
        :param return_ids_tensor:
        :param no_need_bp:
        :return:
            vocab_probs: [max_len, N, vocab_size]
            pred_ids_tensor: [max_len, N]
        """
        enc_state, memory_bank, lengths = self.model.encoder(real_src, real_src_lengths)
        new_memory_bank = self.model.bond_layer(memory_bank, style_represents)
        # self.model.decoder.init_state(real_src, new_memory_bank, enc_state)
        # after softmax, size: [seq_len, N, vocab_size]
        vocab_probs, seq_lens_list, pred_ids_tensor, real_pred_ids_cpu = self.train_translator.decode_batch(
            src=real_src, enc_states=enc_state,
            memory_bank=new_memory_bank, src_lengths=real_src_lengths,
            max_dec_steps=max_dec_steps, no_need_bp=no_need_bp
        )
        seq_lens = torch.LongTensor(seq_lens_list).type_as(real_src_lengths)

        if not return_ids_tensor:
            return vocab_probs, seq_lens
        return vocab_probs, seq_lens, pred_ids_tensor, real_pred_ids_cpu

    def _cycle_bw_dynamic_decoding(
            self, real_src, real_src_lengths, style_represents
            , max_dec_steps=None, return_ids_tensor=False
            , no_need_bp=False
    ):
        enc_state, memory_bank, lengths = self.shadow_model.encoder(real_src, real_src_lengths)
        new_memory_bank = self.shadow_model.bond_layer(memory_bank, style_represents)
        # INFO temporarily swap
        self.train_translator.model = self.shadow_model
        # self.model.decoder.init_state(real_src, new_memory_bank, enc_state)
        # after softmax, size: [seq_len, N, vocab_size]
        vocab_probs, seq_lens_list, pred_ids_tensor, real_pred_ids_cpu = self.train_translator.decode_batch(
            src=real_src, enc_states=enc_state,
            memory_bank=new_memory_bank, src_lengths=real_src_lengths,
            max_dec_steps=max_dec_steps, no_need_bp=no_need_bp
        )
        seq_lens = torch.LongTensor(seq_lens_list).type_as(real_src_lengths)

        # INFO temporarily swap back
        self.train_translator.model = self.model

        if not return_ids_tensor:
            return vocab_probs, seq_lens
        return vocab_probs, seq_lens, pred_ids_tensor, real_pred_ids_cpu

    def _cycle_bw_gen_style_represents(self, desired_style, real_src, real_src_lengths):
        """ notice: remove the first token (style token)"""
        style_represents = self.shadow_style_generator(desired_style, real_src, real_src_lengths)
        # [seq_len, N, dim]
        return style_represents
    def train_reconstruct(self, batch):
        self.style_generator.train()
        self.model.train()
        self.reconst_style_generator_optim.zero_grad()
        self.reconst_model_optim.zero_grad()
        num_styles = self.num_styles
        classifier = self.cls_proxy.fetch_classifier()

        # src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        with torch.no_grad():
            labels, labels_list, real_src, real_src_lengths, vocab_target = self.unpack_batch(batch)
            tgt_len, batch_size = vocab_target.size(0), vocab_target.size(1)
            vocab_target = vocab_target.contiguous().view(-1)
            desired_labels, desired_labels_list = self.train_translator.gen_random_desired_styles(labels_list, num_styles, labels)
        max_dec_steps = tgt_len + self.max_extra_dec_steps

        #
        # INFO reconstruct decoding
        #
        with torch.no_grad():
            noised_real_src = word_drop(real_src, real_src_lengths, self.sr_noise_prob, self.unk_idx)
        # INFO self reconstruct: using style infos generated by classifier
        real_style_represents = self._gen_real_style_represents(real_src, real_src_lengths)
        noised_vocab_log_probs = self._self_reconstruct(noised_real_src, real_src_lengths, batch.tgt, real_style_represents)
        # if 1 != self.opt.alpha_reconst:
        #     noised_vocab_log_probs.register_hook(lambda grad_tensor: grad_tensor * self.opt.alpha_reconst)
        # noised_reconst_vocab_loss = self.vocab_loss_fn(noised_vocab_log_probs.view(-1, noised_vocab_log_probs.size(-1)), vocab_target)
        noised_reconst_vocab_loss = self.vocab_loss_fn.calculate_loss(noised_vocab_log_probs.view(-1, noised_vocab_log_probs.size(-1)), vocab_target, batch_size=batch_size)
        noised_reconst_vl_val = noised_reconst_vocab_loss.item()
        noised_reconst_vocab_loss = self.sr_loss_ratio * noised_reconst_vocab_loss
        total_bp_loss = noised_reconst_vocab_loss
        self.tensorboard_writer.put_scalar('reconst/ae', noised_reconst_vl_val, self.total_train_steps)

        # generate current styles
        gen_s_src_represents = self._gen_style_represents(labels, real_src, real_src_lengths)
        # generate desired styles
        gen_s_tgt_represents = self._gen_style_represents(desired_labels, real_src, real_src_lengths)
        gen_s_tgt_represents_t = gen_s_tgt_represents.transpose(0, 1)

        # INFO self reconstruct: using style infos generated by style_generator
        # vocab_log_probs = self._self_reconstruct(
        #     real_src, real_src_lengths, batch.tgt, gen_s_src_represents
        # )
        # if 1 != self.opt.alpha_reconst:
        #     vocab_log_probs.register_hook(lambda grad_tensor: grad_tensor * self.opt.alpha_reconst)
        reconst_vocab_probs, gen_d_seq_lens, gen_d_vocab_ids, _ = self._dynamic_decoding(
            real_src, real_src_lengths, gen_s_src_represents, max_dec_steps=tgt_len,
            return_ids_tensor=True
        )
        vocab_log_probs = reconst_vocab_probs.log()
        # reconst_vocab_loss = self.vocab_loss_fn(vocab_log_probs.view(-1, vocab_log_probs.size(-1)), vocab_target)
        reconst_vocab_loss = self.vocab_loss_fn.calculate_loss(vocab_log_probs.view(-1, vocab_log_probs.size(-1)), vocab_target, batch_size=batch_size)
        reconst_vl_val = reconst_vocab_loss.item()
        total_bp_loss = total_bp_loss + self.sr_loss_ratio * reconst_vocab_loss
        self.tensorboard_writer.put_scalar('reconst/srcStyle_ae', reconst_vl_val, self.total_train_steps)

        if self.use_cls_feedback:
            # current style cls
            gen_s_src_represents_t = gen_s_src_represents.transpose(0, 1)
            # if 1 != self.opt.alpha_style_cls:
            #     gen_s_src_represents_t.register_hook(lambda grad_tensor: grad_tensor * self.opt.alpha_style_cls)
            s_cls_src_logits = self.cls_proxy.classifiers[0].do_cls(gen_s_src_represents_t, real_src_lengths)
            s_cls_src_loss = self.label_loss_fn(F.log_softmax(s_cls_src_logits, -1), labels)
            s_cls_src_loss_val = s_cls_src_loss.item()
            s_cls_src_acc_val = torch.max(s_cls_src_logits, -1)[1].eq(labels).float().mean().item()
            total_bp_loss = total_bp_loss + self.cls_loss_ratio * s_cls_src_loss
            self.tensorboard_writer.put_scalar('reconst/cls_fb_S_src', s_cls_src_loss_val, self.total_train_steps)

            # target style cls
            s_cls_logits = self.cls_proxy.classifiers[0].do_cls(gen_s_tgt_represents_t, real_src_lengths)
            s_cls_tgt_loss = self.label_loss_fn(F.log_softmax(s_cls_logits, -1), desired_labels)
            s_cls_tgt_loss_val = s_cls_tgt_loss.item()
            s_cls_tgt_acc_val = torch.max(s_cls_logits, -1)[1].eq(desired_labels).float().mean().item()
            total_bp_loss = total_bp_loss + self.cls_loss_ratio * s_cls_tgt_loss
            self.tensorboard_writer.put_scalar('reconst/cls_fb_S_tgt', s_cls_tgt_loss_val, self.total_train_steps)

        # INFO control style gap
        if not self.without_stl_penalty:
            s_gap_src_loss = self.cal_style_gap(real_style_represents, gen_s_src_represents, real_src_lengths)
            s_gap_src_loss_val = s_gap_src_loss.item()
            total_bp_loss = total_bp_loss + self.opt.alpha_style_gap * s_gap_src_loss
            self.tensorboard_writer.put_scalar('reconst/style_gap', s_gap_src_loss_val, self.total_train_steps)

        if self.use_cls_feedback or self.use_recycle:
            cycle_gen_tgt_vocab_probs, cycle_gen_seq_lens, _, _ = self._dynamic_decoding(
                real_src, real_src_lengths, gen_s_tgt_represents,
                max_dec_steps=max_dec_steps,
                return_ids_tensor=not self.no_recycle_to_fw
            )
            style1_to_style2_lens = cycle_gen_seq_lens
        if self.use_cls_feedback:
            # if 1 != self.opt.alpha_text_cls:
            #     gen_d_vocab_probs.register_hook(lambda grad_tensor: grad_tensor * self.opt.alpha_text_cls)
            t_cls_logits = classifier(cycle_gen_tgt_vocab_probs, style1_to_style2_lens)
            t_cls_src_loss = self.label_loss_fn(F.log_softmax(t_cls_logits, -1), desired_labels)
            t_cls_src_loss_val = t_cls_src_loss.item()
            t_cls_src_acc_val = torch.max(t_cls_logits, -1)[1].eq(desired_labels).float().mean().item()
            total_bp_loss = total_bp_loss + self.cls_loss_ratio * t_cls_src_loss
            self.tensorboard_writer.put_scalar('reconst/cls_fb_T_tgt', t_cls_src_loss_val, self.total_train_steps)

        # INFO recycle here
        if self.use_recycle:
            gen_src = cycle_gen_tgt_vocab_probs
            if not self.no_recycle_to_fw and 1 != self.opt.alpha_recycle_fw:
                gen_src.register_hook(lambda grad_tensor: grad_tensor * self.opt.alpha_recycle_fw)
            cycle_gen_s_represents = self._cycle_bw_gen_style_represents(labels, gen_src, style1_to_style2_lens)
            cycle_gen_src_vocab_probs, gen_s_seq_lens = self._cycle_bw_dynamic_decoding(
                cycle_gen_tgt_vocab_probs, style1_to_style2_lens, cycle_gen_s_represents, max_dec_steps=tgt_len
            )
            cycle_gen_src_vocab_log_probs = torch.log(cycle_gen_src_vocab_probs)
            # recycle_loss = self.vocab_loss_fn(cycle_gen_src_vocab_log_probs.view(-1, cycle_gen_src_vocab_log_probs.size(-1)), vocab_target)
            recycle_loss = self.vocab_loss_fn.calculate_loss(cycle_gen_src_vocab_log_probs.view(-1, cycle_gen_src_vocab_log_probs.size(-1)), vocab_target, batch_size=batch_size)
            recycle_loss_val = recycle_loss.item()
            total_bp_loss = total_bp_loss + self.cycle_loss_ratio * recycle_loss
            self.recycle_logger.update(recycle_loss_val)
            self.tensorboard_writer.put_scalar('reconst/recycle', recycle_loss_val, self.total_train_steps)

        log_vals = [
            noised_reconst_vl_val, reconst_vl_val
        ]
        if self.use_cls_feedback:
            log_vals.append(t_cls_src_loss_val)
            log_vals.append(t_cls_src_acc_val)
            log_vals.append(s_cls_tgt_loss_val)
            log_vals.append(s_cls_tgt_acc_val)
            log_vals.append(s_cls_src_loss_val)
            log_vals.append(s_cls_src_acc_val)
        if not self.without_stl_penalty:
            log_vals.append(s_gap_src_loss_val)

        total_bp_loss.backward()
        self.reconstruct_logger.update(*log_vals)
        if self.use_cls_feedback or not self.without_stl_penalty:
            self.reconst_style_generator_optim.step()
        self.reconst_model_optim.step()

    def train_generator(self, batch):
        self.style_generator.train()
        self.model.train()
        self.style_discriminator.eval()
        self.text_discriminator.eval()
        self.style_generator_optim.zero_grad()
        self.model_optim.zero_grad()
        self.total_train_reconstruct_steps += 1
        self.total_train_recycle_steps += 1
        num_styles = self.num_styles
        classifier = self.cls_proxy.fetch_classifier()

        # src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        with torch.no_grad():
            labels, labels_list, real_src, real_src_lengths, vocab_target = self.unpack_batch(batch)
            tgt_len, batch_size = vocab_target.size(0), vocab_target.size(1)
            desired_labels, desired_labels_list = self.train_translator.gen_random_desired_styles(labels_list, num_styles, labels)
        max_dec_steps = tgt_len + self.max_extra_dec_steps
        real_text_lengths = (real_src_lengths + 1).detach()
        if self.discr_sent_stop_aware:
            real_text_lens_stop_aware = (torch.ones_like(real_src_lengths) * max_dec_steps).detach()
            real_text_lengths = real_text_lens_stop_aware
        log_vals = []
        if self.force_ap_text_stop:
            if self.without_bp2stlgen:
                with torch.no_grad():
                    gen_d_style_represents = self._gen_style_represents(desired_labels, real_src, real_src_lengths)
                    gen_d_style_represents = gen_d_style_represents.detach()
            else:
                gen_d_style_represents = self._gen_style_represents(desired_labels, real_src, real_src_lengths)

            gen_d_vocab_probs, gen_d_seq_lens, gen_d_vocab_ids, _ = self._dynamic_decoding(
                real_src, real_src_lengths, gen_d_style_represents, max_dec_steps=tgt_len,
                return_ids_tensor=True
            )
            penalty_mask = torch.eq(vocab_target, self.eos_idx) | torch.eq(vocab_target, self.pad_idx)
            penalty_mask = penalty_mask.contiguous().view(-1)
            penalty_log_probs = torch.log(gen_d_vocab_probs.contiguous().view(-1, gen_d_vocab_probs.size(-1)))[penalty_mask]
            penalty_gold = vocab_target.view(-1)[penalty_mask]
            force_apts_loss = self.force_apts_loss_fn.calculate_loss(penalty_log_probs, penalty_gold, batch_size=None)
            force_apts_loss_val = force_apts_loss.item()
            force_apts_loss.backward()
            self.tensorboard_writer.put_scalar('G/ap_t_stop', force_apts_loss_val, self.total_train_steps)
        #
        # apply text discriminator
        #
        if self.without_bp2stlgen:
            with torch.no_grad():
                gen_d_style_represents = self._gen_style_represents(desired_labels, real_src, real_src_lengths)
                gen_d_style_represents = gen_d_style_represents.detach()
        else:
            gen_d_style_represents = self._gen_style_represents(desired_labels, real_src, real_src_lengths)
        gen_d_vocab_probs, gen_d_seq_lens, gen_d_vocab_ids, _ = self._dynamic_decoding(
            real_src, real_src_lengths, gen_d_style_represents, max_dec_steps=max_dec_steps,
            return_ids_tensor=True
        )
        if 1 != self.opt.alpha_text_ap:
            gen_d_vocab_probs.register_hook(lambda grad_tensor: grad_tensor * self.opt.alpha_text_ap)
        style1_to_style2_lens = gen_d_seq_lens if not self.discr_sent_stop_aware else real_text_lengths
        gen_d_text_logits = self._call_text_discriminator(gen_d_vocab_probs, style1_to_style2_lens)
        gain_t_fake = torch.mean(cal_disc_loss(gen_d_text_logits, desired_labels_list))
        gain_t_fake_val = gain_t_fake.item()
        t_cost = -gain_t_fake_val
        log_vals.append(gain_t_fake_val)
        gain_t_fake.backward(self.text_ap_mone_tensor)
        with torch.no_grad():
            t_cls_logits = classifier(gen_d_vocab_probs, style1_to_style2_lens)
            t_cls_acc_val = torch.max(t_cls_logits, -1)[1].eq(desired_labels).float().mean().item()
        log_vals.append(t_cls_acc_val)
        self.tensorboard_writer.put_scalar('G/T_gain', gain_t_fake_val, self.total_train_steps)
        self.tensorboard_writer.put_scalar('G/T_acc', t_cls_acc_val, self.total_train_steps)
        #
        # train style generator
        #
        # generate desired styles
        if self.style_train_steps < 1 or self.total_train_steps < self.style_train_steps:
            gen_d_style_represents = self._gen_style_represents(desired_labels, real_src, real_src_lengths)
            gen_d_style_represents_t = gen_d_style_represents.transpose(0, 1)
            if 1 != self.opt.alpha_style_ap:
                gen_d_style_represents_t.register_hook(lambda grad_tensor: grad_tensor * self.opt.alpha_style_ap)
            fake_logits = self._call_style_discriminator(gen_d_style_represents_t, real_src_lengths)
            gain_s_fake = torch.mean(cal_disc_loss(fake_logits, desired_labels_list))
            gain_s_fake_val = gain_s_fake.item()
            gain_s_fake.backward(self.style_ap_mone_tensor)
            s_cost = -gain_s_fake_val
            log_vals.append(gain_s_fake_val)
            self.tensorboard_writer.put_scalar('G/S_gain', gain_s_fake_val, self.total_train_steps)
        else:
            log_vals.append(0.0)

        if self.ap_gen_self_style_text:
            gen_s_style_represents = self._gen_style_represents(labels, real_src, real_src_lengths)
            gen_s_vocab_probs, gen_s_seq_lens, _, _ = self._dynamic_decoding(
                real_src, real_src_lengths, gen_s_style_represents, max_dec_steps=tgt_len,
                return_ids_tensor=True
            )
            vocab_log_probs = gen_s_vocab_probs.log()
            gen_self_vocab_loss = self.vocab_loss_fn.calculate_loss(
                vocab_log_probs.view(-1, vocab_log_probs.size(-1)),
                vocab_target.contiguous().view(-1), batch_size=batch_size
            )
            gen_s_vl_val = gen_self_vocab_loss.item()
            gen_self_vocab_loss.backward()
            self.tensorboard_writer.put_scalar('G/T_selfstyle_vloss', gen_s_vl_val, self.total_train_steps)

        if self.dynamic_lr:
            self.lr_adapter.apply_to_optimizer(self.style_generator_optim, self.lr)
            self.lr_adapter.apply_to_optimizer(self.model_optim, self.lr)
        if self.max_grad_norm > 0:
            clip_grad_norm_(self.style_generator.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.style_generator_optim.step()
        self.model_optim.step()
        if self.force_ap_text_stop:
            log_vals.append(force_apts_loss_val)
        if self.ap_gen_self_style_text:
            log_vals.append(gen_s_vl_val)
        self.main_logger.update(*log_vals)

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...', valid_steps)

        init_steps = self.init_steps
        learn_disc_steps = self.learn_disc_steps
        main_steps = self.main_steps
        learn_reconst_steps = self.opt.reconstruct_steps
        reconstruct_every_step = self.opt.reconstruct_every_step
        if reconstruct_every_step:
            learn_reconst_steps = 0
        reconst_accu_steps = 0
        disc_accu_steps = 0
        main_accu_steps = 0
        best_acc = 0.0
        for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):
            # UPDATE DROPOUT
            self._maybe_update_dropout(self.total_train_steps)
            for k, batch in enumerate(batches):
                self.total_train_steps += 1
                if self.dynamic_lr:
                    self.lr = self.lr_adapter.step_and_get_lr()
                try:
                    if init_steps >= self.total_train_steps:
                        self.init_launch(batch)
                    else:
                        if self.start_peak_step > 0 and self.start_peak_step > self.total_train_steps:
                            main_steps = self.peak_main_steps
                        if reconstruct_every_step:
                            self.train_reconstruct(batch)
                        if reconst_accu_steps < learn_reconst_steps:
                            reconst_accu_steps += 1
                            self.train_reconstruct(batch)
                        elif disc_accu_steps < learn_disc_steps:
                            disc_accu_steps += 1
                            self.train_discriminator(batch)
                        elif main_accu_steps < main_steps:
                            main_accu_steps += 1
                            self.train_generator(batch)
                        if main_accu_steps >= main_steps:
                            reconst_accu_steps = 0
                            disc_accu_steps = 0
                            main_accu_steps = 0
                except Exception as e:
                    if 'CUDA out of memory' in str(e):
                        logger.info(e)
                        logger.info('GPU OOM: SKIP THIS BATCH')
                    else:
                        raise e
                if 0 == self.total_train_steps % valid_steps:
                    learn_disc_steps += self.opt.learn_disc_increase
                    if learn_disc_steps < 1:
                        learn_disc_steps = 1
                    main_steps += self.opt.apply_disc_increase
                    if main_steps < 1:
                        main_steps = 1
                    learn_reconst_steps += self.opt.learn_reconst_increase
                    if learn_reconst_steps < 1:
                        learn_reconst_steps = 1
                if 0 == self.total_train_steps % self.report_every:
                    logger.info('-' * 120)
                    logger.info('TRAIN_STEP: {}'.format(self.total_train_steps))
                    logger.info('LR: {}'.format(self.lr))
                    if self.total_train_steps <= init_steps:
                        logger.info('#### INIT TRAINING ####')
                        logger.info('{}'.format(self.init_logger.get_metric()))
                        self.init_logger.clear()

                        if not self.without_disc_init:
                            logger.info('#### DISCRIMINATOR ####')
                            logger.info('{}'.format(self.disc_logger.get_metric()))
                            self.disc_logger.clear()
                    else:
                        logger.info('#### DISCRIMINATOR ####')
                        logger.info('{}'.format(self.disc_logger.get_metric()))
                        self.disc_logger.clear()
                        logger.info('#### MAIN ####')
                        logger.info('{}'.format(self.main_logger.get_metric()))
                        logger.info('{}'.format(self.reconstruct_logger.get_metric()))
                        self.main_logger.clear()
                        self.reconstruct_logger.clear()
                        # if self.use_recycle and 0 == self.total_train_recycle_steps % self.recycle_steps:
                        if self.use_recycle:
                            logger.info('{}'.format(self.recycle_logger.get_metric()))
                            self.recycle_logger.clear()
                    pass
                    self.shadow_model.load_state_dict(self.model.state_dict())
                    self.shadow_style_generator.load_state_dict(self.style_generator.state_dict())
                    self.shadow_model.eval()
                    self.shadow_style_generator.eval()
            if valid_iter is not None and self.total_train_steps % valid_steps == 0:
                logger.info('## =======================')
                logger.info('Validating ...')
                valid_acc = self.validate(valid_iter)
                logger.info('Valid acc: {}'.format(valid_acc))
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    logger.info('Finding new best')
                logger.info('## =======================')

            if (self.save_path is not None
                    and (save_checkpoint_steps != 0 and self.total_train_steps % save_checkpoint_steps == 0)):
                ckpt = {
                    'main_model': self.model,
                    'style_generator': self.style_generator,
                    'style_discriminator': self.style_discriminator,
                    'text_discriminator': self.text_discriminator,
                    'step': self.total_train_steps
                }
                torch.save(ckpt, self.save_path + '_{}.pt'.format(self.total_train_steps))
            if train_steps > 0 and self.total_train_steps >= train_steps:
                break
            if self.total_train_steps % valid_steps == 0:
                self.temperature_schedule.update()
                if self.d_g_use_temperature:
                    self.train_translator.dec_temperature = self.temperature_schedule.get_val()

        ckpt = {
            'main_model': self.model,
            'style_generator': self.style_generator,
            'style_discriminator': self.style_discriminator,
            'text_discriminator': self.text_discriminator,
            'step': self.total_train_steps
        }
        torch.save(ckpt, self.save_path + '_{}.pt'.format(self.total_train_steps))
        return None

    def _show_gen_results(self, src_vocab, pred_vocab, gen_d_seq_lens, labels_list, desired_labels_list):
        src_vocab = src_vocab[:self.n_show_gen_texts].tolist()
        pred_vocab = pred_vocab[:self.n_show_gen_texts].tolist()
        gen_d_seq_lens = gen_d_seq_lens[:self.n_show_gen_texts].tolist()
        _pred_vocab = []
        for sent_toks, sent_len in zip(pred_vocab, gen_d_seq_lens):
            _pred_vocab.append(sent_toks[:sent_len])
        pred_vocab = _pred_vocab
        src_toks = [[self.train_translator._tgt_vocab.itos[widx] for widx in sent_toks] for sent_toks in src_vocab]
        pred_toks = [[self.train_translator._tgt_vocab.itos[widx] for widx in sent_toks] for sent_toks in pred_vocab]
        src_sents = [' '.join(sent_toks).replace(' <blank>', '') for sent_toks in src_toks]
        pred_sents = [' '.join(sent_toks) for sent_toks in pred_toks]

        labels = labels_list[:self.n_show_gen_texts]
        desired_labels = desired_labels_list[:self.n_show_gen_texts]
        for idx in range(len(labels)):
            logger.info('SOURCE STYLE::: {} == {}'.format(labels[idx], src_sents[idx]))
            logger.info('TARGET STYLE::: {} == {}'.format(desired_labels[idx], pred_sents[idx]))

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        print('=' * 80)
        print('Validating ...')

        num_styles = self.num_styles
        self.style_generator.eval()
        self.model.eval()
        classifier = self.cls_proxy.fetch_classifier()
        # reset temperature temporarily
        old_dec_temperature = self.train_translator.dec_temperature
        self.train_translator.dec_temperature = 1.0

        total_acc = 0.0
        total_step = 0.0
        with torch.no_grad():
            for batch in valid_iter:
                total_step += 1
                labels, labels_list, real_src, real_src_lengths, vocab_target = self.unpack_batch(batch)
                real_text_lengths = (real_src_lengths + 1).detach()
                if self.discr_sent_stop_aware:
                    real_text_lens_stop_aware = (torch.ones_like(real_src_lengths) * vocab_target.size(0)).detach()
                    real_text_lengths = real_text_lens_stop_aware
                desired_labels, desired_labels_list = self.train_translator.gen_random_desired_styles(labels_list, num_styles, labels)

                gen_d_style_represents = self._gen_style_represents(desired_labels, real_src, real_src_lengths)
                gen_d_vocab_probs, gen_d_seq_lens, gen_d_vocab_ids, _ = self._dynamic_decoding(
                    real_src, real_src_lengths, gen_d_style_represents, max_dec_steps=None,
                    return_ids_tensor=True, no_need_bp=True
                )
                style1_to_style2_lens = gen_d_seq_lens if not self.discr_sent_stop_aware else real_text_lengths
                t_cls_logits = classifier(gen_d_vocab_ids.unsqueeze(2), style1_to_style2_lens)
                t_cls_ap_acc_val = torch.max(t_cls_logits, -1)[1].eq(desired_labels).float().mean().item()
                total_acc += t_cls_ap_acc_val

                if 1 == total_step:
                    cur_pred_vocab = gen_d_vocab_ids.transpose(0, 1)
                    self._show_gen_results(
                        real_src.transpose(0, 1).squeeze(2), cur_pred_vocab, style1_to_style2_lens,
                        labels_list, desired_labels_list
                    )

        # Set model back to training mode.
        self.style_generator.train()
        self.model.train()
        print('=' * 80)
        # restore old temperature
        self.train_translator.dec_temperature = old_dec_temperature
        return total_acc/total_step


def cal_disc_loss(probs, labels):
    """
    probs: [N, n_styles]
    """
    bz = probs.size(0)
    gains = []
    for idx in range(bz):
        gains.append(probs[idx][labels[idx]].unsqueeze(-1))
    return torch.cat(gains, 0)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.

    return: [N, x_len]
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def penalize_gen_style_grad(d_model, real, fake, real_src_lengths, labels,
                            k=1.0, lamb=10, gp_norm_seq=True):
    """
    :param d_model: style_D
    :param real: [N, x_len, dim]
    :param fake: [N, x_len, dim]
    :param real_src_lengths: [N]
    :param labels: [N]
    :param k:
    :param lamb:
    :param gp_norm_seq:
        the default implementation of wgan-gp by caogang version norms across the dim=1,
        which means norm on x_len axis. see: https://github.com/caogang/wgan-gp/issues/25 , for details.

        However, in the official implementation of the original paper "Improved Training of Wasserstein GANs",
        they pat the input of discriminator to shape [-1, dim].

        In our implementation, we simply reshape the gradients to [-1, dim], and norm across axis 1 .
    :return:
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1).type_as(real).expand_as(real)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates = interpolates.detach()
    interpolates.requires_grad = True

    d_interpolates = d_model(interpolates, real_src_lengths)
    error_middle = cal_disc_loss(d_interpolates, labels)

    ones = torch.ones_like(error_middle)
    gradients = grad_fn(
        outputs=error_middle, inputs=interpolates,
        grad_outputs=ones, create_graph=True,
        retain_graph=True, only_inputs=True)[0]
    # grad_penalty = ((gradients.norm(2, dim=1) - k) ** 2).mean() * lamb
    if gp_norm_seq:
        last_dim = gradients.size(-1)
        gradients = gradients.contiguous().view(-1, last_dim)
        useful_mask = sequence_mask(real_src_lengths)
        useful_mask = useful_mask.contiguous().view(-1, 1).expand_as(gradients)
        gradients = gradients[useful_mask].view(-1, last_dim)

    # [N, dim]
    grad_norm = gradients.norm(2, dim=1)
    grad_penalty = ((grad_norm - k) ** 2).mean() * lamb

    return grad_penalty


def convert_sent_ids_to_onehot(sent_ids, refer_logits):
    """
    :param sent_ids: [N, max_x_len]
    :param refer_logits: [N, max_x_len, vocab_size]
    :return:
    """
    vocab_size = refer_logits.size(2)
    batch_size, max_x_len = sent_ids.size()
    sent_ids = sent_ids.contiguous().view(-1, 1)
    # one_hots = torch.zeros(batch_size, 1).type_as(refer_logits).expand(batch_size * max_x_len, vocab_size)  # bug here
    # one_hots = torch.zeros_like(refer_logits).contiguous().view(batch_size * max_x_len, vocab_size)
    one_hots = torch.zeros(batch_size * max_x_len, vocab_size).to(refer_logits.device)
    one_hots = one_hots.scatter_(1, sent_ids, 1)
    one_hots = one_hots.contiguous().view(batch_size, max_x_len, vocab_size)
    return one_hots


def penalize_gen_text_grad(d_model, real, fake, real_src_lengths, labels,
                            k=1.0, lamb=10, gp_norm_seq=True):
    """
    :param d_model: style_D
    :param real: [N, x_len, dim]
    :param fake: [N, x_len, dim]
    :param real_src_lengths: [N]
    :param labels: [N]
    :param k:
    :param lamb:
    :param gp_norm_seq:
        the default implementation of wgan-gp by caogang version norms across the dim=1,
        which means norm on x_len axis. see: https://github.com/caogang/wgan-gp/issues/25 , for details.

        However, in the official implementation of the original paper "Improved Training of Wasserstein GANs",
        they pat the input of discriminator to shape [-1, dim].

        In our implementation, we simply reshape the gradients to [-1, dim], and norm across axis 1 .
    :return:
    """
    batch_size = real.size(0)
    x_len = real.size(1)
    alpha = torch.rand(batch_size, 1, 1).type_as(real).expand_as(real)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates = interpolates.transpose(0, 1)
    interpolates = interpolates.detach()
    interpolates.requires_grad = True

    d_interpolates = d_model(interpolates, real_src_lengths)
    error_middle = cal_disc_loss(d_interpolates, labels)

    ones = torch.ones_like(error_middle)
    gradients = grad_fn(
        outputs=error_middle, inputs=interpolates,
        grad_outputs=ones, create_graph=True,
        retain_graph=True, only_inputs=True)[0]
    gradients = gradients.transpose(0, 1)
    # grad_penalty = ((gradients.norm(2, dim=1) - k) ** 2).mean() * lamb
    if gp_norm_seq:
        last_dim = gradients.size(-1)
        gradients = gradients.contiguous().view(-1, last_dim)
        useful_mask = sequence_mask(real_src_lengths, max_len=x_len)
        useful_mask = useful_mask.contiguous().view(-1)
        gradients = gradients[useful_mask]
    # [N, dim]
    grad_norm = gradients.norm(2, dim=1)
    grad_penalty = ((grad_norm - k) ** 2).mean() * lamb

    return grad_penalty





