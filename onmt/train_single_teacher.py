#!/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import build_dataset_iter, patch_fields, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple, IterOnDevice
from onmt.style_generator_builder import build_classifier_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer_teacher import build_trainer
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    other = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            other += param.nelement()
    return enc + other, enc, other


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    print('#' * 40)
    print('Setting Random Seed ...')
    print('#' * 40)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    checkpoint = None
    model_opt = opt
    vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # patch for fields that may be missing in old data/model
    patch_fields(opt, fields)

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_classifier_model(model_opt, opt, fields, checkpoint)
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        ckpt = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
        ckpt.load_state_dict(ckpt.state_dict())
    n_params, n_enc_params, n_other_params = _tally_parameters(model)
    logger.info('encoder params: %d' % n_enc_params)
    logger.info('other_params: %d' % n_other_params)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # saving vocab and model_opt
    logger.info('+' * 80)
    logger.info('Saving vocab and model_opt ...')
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    torch.save(
        {
            'model_opt': model_opt,
            'fields': fields
        }, os.path.join(model_dirname, 'vocab_model_opt.pt')
    )
    logger.info('+' * 80)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=None)

    if batch_queue is None:
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            train_iter = build_dataset_iter(shard_base, fields, opt)
        train_iter = IterOnDevice(train_iter, device_id)
    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                # Move batch to specified device
                IterOnDevice.batch_to_device(batch, device_id)
                yield batch

        train_iter = _train_iter()

    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, device_id)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
