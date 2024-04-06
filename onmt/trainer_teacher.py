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
import traceback

import onmt.utils
from onmt.utils.logging import logger
import os
from torch.nn import functional as F
from torch.nn import NLLLoss
from onmt.train_and_translate import TrainTranslator


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    tgt_field = dict(fields)["tgt"].base_field
    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level
    save_model_path = os.path.abspath(opt.save_model)

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    source_noise = None
    if len(opt.src_noise) > 0:
        src_field = dict(fields)["src"].base_field
        corpus_id_field = dict(fields).get("corpus_id", None)
        if corpus_id_field is not None:
            ids_to_noise = corpus_id_field.numericalize(opt.data_to_noise)
        else:
            ids_to_noise = None
        source_noise = onmt.modules.source_noise.MultiNoise(
            opt.src_noise,
            opt.src_noise_prob,
            ids_to_noise=ids_to_noise,
            pad_idx=src_field.pad_token,
            end_of_sentence_mask=src_field.end_of_sentence_mask,
            word_start_mask=src_field.word_start_mask,
            device_id=device_id
        )

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = Trainer(model, optim,
                      fields,
                      trunc_size,
                      shard_size, norm_method,
                      accum_count, accum_steps,
                      n_gpu, gpu_rank,
                      gpu_verbose_level, report_manager,
                      with_align=True if opt.lambda_align > 0 else False,
                      model_saver=model_saver if gpu_rank == 0 else None,
                      average_decay=average_decay,
                      average_every=average_every,
                      model_dtype=opt.model_dtype,
                      earlystopper=earlystopper,
                      dropout=dropout,
                      dropout_steps=dropout_steps,
                      source_noise=source_noise,
                      save_path=save_model_path, opt=opt, padding_idx=padding_idx
    )
    return trainer


class Trainer(object):
    def __init__(self, model, optim,
                 fields,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0],
                 source_noise=None, save_path=None, opt=None, padding_idx=4):
        # Basic attributes.
        self.model = model
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.source_noise = source_noise
        self.save_path = save_path
        self.opt = opt
        self.padding_idx = padding_idx
        self.loss_fn = NLLLoss()

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""
        self.opt = opt
        self.train_translator = TrainTranslator(
            model=model, style_generator=None,
            fields=fields,
            opt=opt,
            build_translator=False
        )
        # Set model in training mode.
        self.model.train()
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
        # [1, 1, 1]
        self.pad_tensor = torch.LongTensor([[[self.train_translator._tgt_pad_idx]]]).to(device)

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
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_train_loss = 0.0
        total_step = 0
        total_acc = 0.0
        report_every = self.opt.report_every
        best_acc = 0.0
        optimizer = torch.optim.Adam(params=self.model.parameters())

        for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):
            # UPDATE DROPOUT
            self._maybe_update_dropout(total_step)

            for k, batch in enumerate(batches):
                batch = self.maybe_noise_source(batch)

                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)

                real_src = src[1:]
                real_rec_lengths = src_lengths - 1
                labels = src[0, :, 0]
                labels_list = labels.tolist()
                labels_list = [self.train_translator.convert_label_idx(lvidx) for lvidx in labels_list]
                labels = torch.LongTensor(labels_list).type_as(labels)

                logits = self.model(real_src, real_rec_lengths)
                loss = self.loss_fn(F.log_softmax(logits, dim=-1), labels)
                loss_val = loss.item()

                # Compatible with other situations
                ## end of text has EOS
                new_x = torch.cat([batch.tgt[1:], self.pad_tensor.repeat(1, batch.tgt.size(1), 1)])
                new_x_lens = real_rec_lengths + 1
                new_logits = self.model(new_x, new_x_lens)
                new_loss = self.loss_fn(F.log_softmax(new_logits, dim=-1), labels)
                loss = loss + new_loss
                ## considering: eos pad
                new_logits = self.model(new_x, torch.ones_like(src_lengths) * new_x.size(0))
                new_loss = self.loss_fn(F.log_softmax(new_logits, dim=-1), labels)
                loss = loss + new_loss

                total_train_loss += loss_val
                total_step += 1
                pred_label = torch.max(logits, 1)[1]
                acc = labels.eq(pred_label).float().mean().item()
                total_acc += acc
                if 0 == total_step % report_every:
                    logger.info('step: {}, loss: {}, acc: {}'.format(
                        total_step, total_train_loss/report_every, total_acc/report_every))
                    total_train_loss = 0.0
                    total_acc = 0.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if valid_iter is not None and total_step % valid_steps == 0:
                logger.info('## =======================')
                logger.info('Validating ...')
                valid_acc = self.validate(valid_iter, moving_average=self.moving_average)
                logger.info('Valid acc: {}'.format(valid_acc))
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    logger.info('Finding new best')
                logger.info('## =======================')

            if (self.save_path is not None
                and (save_checkpoint_steps != 0
                     and total_step % save_checkpoint_steps == 0)):
                torch.save(self.model, self.save_path + '_{}.pt'.format(total_step))
            if train_steps > 0 and total_step >= train_steps:
                break

        torch.save(self.model, self.save_path + '_{}.pt'.format(total_step))
        return None

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.
        valid_model.eval()

        total_acc = 0.0
        total_step = 0.0
        with torch.no_grad():
            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
                real_src = src[1:]
                real_rec_lengths = src_lengths - 1
                labels = src[0, :, 0]
                labels_list = labels.tolist()
                labels_list = [self.train_translator.convert_label_idx(lvidx) for lvidx in labels_list]
                labels = torch.LongTensor(labels_list).type_as(labels)
                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    # F-prop through the model.
                    logits = valid_model(real_src, real_rec_lengths)

                    pred_label = torch.max(logits, 1)[1]
                    acc = labels.eq(pred_label).float().mean().item()
                    total_acc += acc
                    total_step += 1

        if moving_average:
            for param_data, param in zip(model_params_data,
                                         self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()
        return total_acc/total_step

    def maybe_noise_source(self, batch):
        if self.source_noise is not None:
            return self.source_noise(batch)
        return batch
