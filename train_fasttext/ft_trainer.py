# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from train_fasttext.ft_dataset import FtClsDataset
from onmt.utils.logging import logger
from torch.nn.functional import kl_div
from torch.nn import functional as F
from torch.nn import KLDivLoss


def main(config):
    # loading model
    restore_from = config.restore_from
    use_gpu = config.use_gpu
    logger.info('+' * 80)
    logger.info('Restoring from: {}'.format(restore_from))
    logger.info('+' * 80)
    model = torch.load(restore_from, map_location=lambda storage, loc: storage)
    if use_gpu:
        model = model.cuda()

    # loading vocab
    vocab_opt_path = config.vocab_opt_path
    vocab_opt = torch.load(vocab_opt_path)
    fields = vocab_opt['fields']
    tgt_field = dict(fields)["tgt"].base_field

    # building dataset
    batch_size = config.batch_size
    train_data_path = config.train_data_path
    valid_data_path = config.valid_data_path
    src_seq_length = config.src_seq_length
    train_loader = DataLoader(
        FtClsDataset(tgt_field, train_data_path, src_seq_length),
        batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        FtClsDataset(tgt_field, valid_data_path, src_seq_length),
        batch_size=batch_size, shuffle=False
    )
    # training loops
    n_epoch = config.n_epoch
    report_every = config.report_every
    save_path = config.save_path
    learning_rate = config.learning_rate
    loss_fn = KLDivLoss()
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_loss = 1000000000
    for epid in range(n_epoch):
        logger.info('=' * 80)
        logger.info('## Epoch: {}'.format(epid))
        step = 0
        total_loss = 0.0
        total_loss2 = 0.0
        model.train()
        for batch in train_loader:
            step += 1
            x_ids = batch['x_ids']
            x_len = batch['x_len']
            probs = batch['probs']
            x_ids_with_eos = batch['x_ids_with_eos']
            x_len = x_len.squeeze(1)
            real_src = x_ids.transpose(0, 1).unsqueeze(2)
            real_src_with_eos = x_ids_with_eos.transpose(0, 1).unsqueeze(2)
            real_rec_lengths = x_len
            if use_gpu:
                real_src = real_src.cuda()
                real_rec_lengths = real_rec_lengths.cuda()
                probs = probs.cuda()
                real_src_with_eos = real_src_with_eos.cuda()
            logits = model(real_src, real_rec_lengths)
            loss = loss_fn(F.log_softmax(logits, -1), probs)
            loss_val = loss.item()

            # 兼容其他情况
            ## 末尾有 eos
            new_logits = model(real_src_with_eos, real_rec_lengths + 1)
            new_loss = loss_fn(F.log_softmax(new_logits, -1), probs)
            loss = loss + new_loss
            total_loss2 += new_loss.item()
            ## 将 eos pad情况都考虑上
            new_logits = model(real_src_with_eos, torch.ones_like(real_rec_lengths) * real_src_with_eos.size(0))
            new_loss = loss_fn(F.log_softmax(new_logits, -1), probs)
            loss = loss + new_loss
            total_loss2 += new_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss_val
            if 0 == step % report_every:
                logger.info('Step: {}, Loss: {}, Loss2: {}'.format(
                    step, total_loss / report_every,
                    total_loss2 / (2 * report_every)
                ))
                total_loss = 0.0
                total_loss2 = 0.0
        torch.save(model, save_path + '_{}.pt'.format(epid))
        logger.info('-' * 40)
        logger.info('VALID ...')
        valid_step = 0
        total_valid_loss = 0.0
        model.eval()
        for batch in valid_loader:
            valid_step += 1
            x_ids = batch['x_ids']
            x_len = batch['x_len']
            probs = batch['probs']
            x_len = x_len.squeeze(1)
            real_src = x_ids.transpose(0, 1).unsqueeze(2)
            real_rec_lengths = x_len
            if use_gpu:
                real_src = real_src.cuda()
                real_rec_lengths = real_rec_lengths.cuda()
                probs = probs.cuda()
            logits = model(real_src, real_rec_lengths)
            loss = loss_fn(F.log_softmax(logits, -1), probs)
            loss_val = loss.item()
            total_valid_loss += loss_val
        valid_loss = total_valid_loss / valid_step
        logger.info('valid loss: {}'.format(valid_loss))
        if best_loss > valid_loss:
            best_loss = valid_loss
            logger.info('FOUND NEW BEST')
    pass




