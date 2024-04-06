#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import torch
from torch.nn import functional as F
from onmt.tensor_utils import convert_to_onehot
from onmt.utils.logging import logger
import random


class TrainTranslator(object):
    def __init__(
            self,
            model,
            style_generator,
            fields,
            opt,
            build_translator=True
    ):
        print("Trying to build TrainTranslator :: inner")
        assert opt is not None
        self.model = model
        self.style_generator = style_generator
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self.sent_stop_aware = opt.sent_stop_aware
        self.dot_idx = self._tgt_vocab.stoi['.'] if '.' in self._tgt_vocab.stoi else self._tgt_eos_idx
        self.excalmatory_idx = self._tgt_vocab.stoi['!'] if '!' in self._tgt_vocab.stoi else self._tgt_eos_idx
        self.interrogation_idx = self._tgt_vocab.stoi['?'] if '?' in self._tgt_vocab.stoi else self._tgt_eos_idx
        logger.info("EOS: {}, PAD: {}, BOS: {}, UNK: {}, DOT: {}, EXCAL: {}, INTERR: {}".format(
            self._tgt_eos_idx, self._tgt_pad_idx, self._tgt_bos_idx, self._tgt_unk_idx, self.dot_idx
            , self.excalmatory_idx, self.interrogation_idx
        ))

        self.fields = fields
        src_field = dict(self.fields)["src"].base_field
        tgt_field = dict(self.fields)["tgt"].base_field
        self._src_vocab = src_field.vocab
        self._tgt_vocab = tgt_field.vocab
        label_tokens = opt.label_tokens
        label_toks = label_tokens.split(",")
        self.label_map = {self._tgt_vocab.stoi[tok]: lidx for lidx, tok in enumerate(label_toks)}
        logger.info("-" * 80)
        logger.info("label_map: {}".format(self.label_map))

        self._tgt_vocab_len = len(self._tgt_vocab)
        self.max_dec_steps = opt.max_dec_steps
        self.dynamic_dec_type = opt.dynamic_dec_type
        if build_translator:
            from onmt.translate.during_training_translator import build_translator_for_train
            self.official_translator = build_translator_for_train(opt, opt, model, style_generator, fields)

        self.dec_temperature = 1.0

    def convert_label_idx(self, vocab_idx):
        return self.label_map[vocab_idx]

    def gen_random_desired_styles(self, labels_list, num_styles, refer_type_tensor):
        cur_labels = labels_list
        if num_styles > 2:
            new_labels = []
            for i in cur_labels:
                idx = random.randint(0, num_styles-1)
                if i == idx:
                    idx = (i + 1) % num_styles
                new_labels.append(idx)
        else:
            new_labels = [1 - x for x in cur_labels]
        new_labels_tensor = torch.LongTensor(new_labels).type_as(refer_type_tensor)
        return new_labels_tensor, new_labels

    # def translate_batch(self, batch):
    #     """Translate a batch of sentences."""
    #     return self._translate_batch_with_strategy(batch)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)

        enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            memory_lengths,
            step=None):
        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )
        # log_probs = self.model.generator(dec_out.squeeze(0))
        # return log_probs
        n_g_layers = len(self.model.generator)
        # [1, N, vocab_size]
        probs = dec_out
        for i in range(n_g_layers - 1):
            probs = self.model.generator[i](probs)
        probs = F.softmax(probs / self.dec_temperature, -1)
        return probs

    def decode_batch(self, src, enc_states, memory_bank, src_lengths, max_dec_steps=None, no_need_bp=False):
        if 'use_ori_train_dec' == self.dynamic_dec_type:
            return self.decode_batch_ori(src, enc_states, memory_bank, src_lengths, max_dec_steps, no_need_bp=no_need_bp)
        if 'use_discrete_accum_dec' == self.dynamic_dec_type:
            return self.decode_batch_accum_discrete(src, enc_states, memory_bank, src_lengths, max_dec_steps, no_need_bp=no_need_bp)
        if 'use_official_dec' == self.dynamic_dec_type:
            return self.decode_batch_official(src, enc_states, memory_bank, src_lengths, max_dec_steps, no_need_bp=no_need_bp)
        if 'use_accum_dec' == self.dynamic_dec_type:
            return self.decode_batch_accum(src, enc_states, memory_bank, src_lengths, max_dec_steps, no_need_bp=no_need_bp)
        raise ValueError('illegal dynamic decoding strategy')

    def decode_batch_accum(self, src, enc_states, memory_bank, src_lengths, max_dec_steps=None, no_need_bp=False):
        if max_dec_steps is None:
            max_dec_steps = self.max_dec_steps

        # (0) Prep the components of the search.
        batch_size = src_lengths.size(0)
        # (1) Run the encoder on the src.
        # self.model.decoder.init_state(src, memory_bank, enc_states)
        # (3) Begin decoding step by step:
        # [1, N, 1]
        # decoder_input = torch.LongTensor([[self._tgt_bos_idx] * batch_size]).type_as(src_lengths).unsqueeze(2)
        decoder_input = torch.LongTensor([self._tgt_bos_idx] * batch_size).type_as(src_lengths)
        decoder_input = convert_to_onehot(decoder_input, self._tgt_vocab_len, src.device).unsqueeze(0)
        with torch.no_grad():
            for step in range(max_dec_steps - 1):
                self.model.decoder.init_state(src, memory_bank, enc_states)
                # decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
                # probs: [1, N, vocab_size]
                probs = self._decode_and_generate(
                    decoder_in=decoder_input,
                    memory_bank=memory_bank,
                    memory_lengths=src_lengths,
                    step=None)
                decoder_input = torch.cat((decoder_input, probs[-1:]), 0)
        decoder_input = decoder_input.detach()
        self.model.decoder.init_state(src, memory_bank, enc_states)
        probs = self._decode_and_generate(
            decoder_in=decoder_input,
            memory_bank=memory_bank,
            memory_lengths=src_lengths,
            step=None)

        # [seq_len, batch_size, vocab_size]
        vocab_probs = probs
        seq_lens_list, pred_ids_tensor, real_pred_ids_cpu = self.fetch_dec_output_length(vocab_probs)
        return vocab_probs, seq_lens_list, pred_ids_tensor, real_pred_ids_cpu

    def decode_batch_accum_discrete(self, src, enc_states, memory_bank, src_lengths, max_dec_steps=None, no_need_bp=False):
        if max_dec_steps is None:
            max_dec_steps = self.max_dec_steps

        # (0) Prep the components of the search.
        batch_size = src_lengths.size(0)
        # (1) Run the encoder on the src.
        # self.model.decoder.init_state(src, memory_bank, enc_states)
        # (3) Begin decoding step by step:
        # [1, N, 1]
        decoder_input = torch.LongTensor([self._tgt_bos_idx] * batch_size).type_as(src_lengths).unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            for step in range(max_dec_steps - 1):
                self.model.decoder.init_state(src, memory_bank, enc_states)
                # decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
                # probs: [1, N, vocab_size]
                probs = self._decode_and_generate(
                    decoder_in=decoder_input,
                    memory_bank=memory_bank,
                    memory_lengths=src_lengths,
                    step=None)
                pred_id = torch.argmax(probs[-1:], 2).unsqueeze(2)
                decoder_input = torch.cat((decoder_input, pred_id), 0)
        decoder_input = decoder_input.detach()
        self.model.decoder.init_state(src, memory_bank, enc_states)
        probs = self._decode_and_generate(
            decoder_in=decoder_input,
            memory_bank=memory_bank,
            memory_lengths=src_lengths,
            step=None)

        # [seq_len, batch_size, vocab_size]
        vocab_probs = probs
        seq_lens_list, pred_ids_tensor, real_pred_ids_cpu = self.fetch_dec_output_length(vocab_probs)
        return vocab_probs, seq_lens_list, pred_ids_tensor, real_pred_ids_cpu

    def decode_batch_ori(self, src, enc_states, memory_bank, src_lengths, max_dec_steps=None, no_need_bp=False):
        if max_dec_steps is None:
            max_dec_steps = self.max_dec_steps

        # (0) Prep the components of the search.
        batch_size = src_lengths.size(0)
        # (1) Run the encoder on the src.
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # (3) Begin decoding step by step:
        # [1, N, 1]
        # decoder_input = torch.LongTensor([[self._tgt_bos_idx] * batch_size]).type_as(src_lengths).unsqueeze(2)
        decoder_input = torch.LongTensor([self._tgt_bos_idx] * batch_size).type_as(src_lengths)
        decoder_input = convert_to_onehot(decoder_input, self._tgt_vocab_len, src.device).unsqueeze(0)
        dec_outs = []
        for step in range(max_dec_steps):
            # decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            # probs: [1, N, vocab_size]
            probs = self._decode_and_generate(
                decoder_in=decoder_input,
                memory_bank=memory_bank,
                memory_lengths=src_lengths,
                step=step)
            dec_outs.append(probs[-1:])
            decoder_input = dec_outs[-1]
        results = torch.cat(dec_outs, 0)

        # [seq_len, batch_size, vocab_size]
        vocab_probs = results
        seq_lens_list, pred_ids_tensor, real_pred_ids_cpu = self.fetch_dec_output_length(vocab_probs)
        return vocab_probs, seq_lens_list, pred_ids_tensor, real_pred_ids_cpu

    def decode_batch_official(self, src, enc_states, memory_bank, src_lengths, max_dec_steps=None, no_need_bp=False):
        if max_dec_steps is None:
            max_dec_steps = self.max_dec_steps
        with torch.no_grad():
            ng_enc_states = enc_states.detach()
            ng_memory_bank = memory_bank.detach()
            raw_results = self.official_translator.translate_batch_for_train(
                src, ng_enc_states, ng_memory_bank, src_lengths
                , max_dec_steps=max_dec_steps
            )
            predictions = raw_results['predictions']
            output_states = raw_results['output_states']

            batch_size = src_lengths.size(0)
            pred_tgt_lens = []
            cur_device = src_lengths.get_device() if src_lengths.get_device() > -1 else src_lengths.device
            pred_tgt_ids = torch.full(
                [max_dec_steps, batch_size], self._tgt_bos_idx,
                dtype=torch.long, device=cur_device)
            pred_tgt_states = torch.full(
                [max_dec_steps, batch_size, memory_bank.size(-1)], 0.0,
                dtype=torch.float, device=cur_device)
            for bidx in range(batch_size):
                cur_len = min(predictions[bidx][0].size(0), output_states[bidx][0].size(0), max_dec_steps)
                # delete EOS at the end of text
                cur_len = cur_len - 1 if cur_len > 1 and predictions[bidx][0][cur_len-1] == self._tgt_eos_idx else 1
                pred_tgt_ids[:cur_len, bidx] = predictions[bidx][0][:cur_len]
                pred_tgt_lens.append(cur_len)
                pred_tgt_states[:cur_len, bidx, :] = output_states[bidx][0][:cur_len]
            n_g_layers = len(self.model.generator)
            probs = pred_tgt_states
            for i in range(n_g_layers - 1):
                probs = self.model.generator[i](probs)
            vocab_probs = F.softmax(probs / self.dec_temperature, -1)
            pred_tgt_ids = pred_tgt_ids.detach_()
            vocab_probs = vocab_probs.detach_()
        if no_need_bp:
            return vocab_probs, pred_tgt_lens, pred_tgt_ids, None

        # (0) Prep the components of the search.
        # (1) Run the encoder on the src.
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # (3) Begin decoding step by step:
        with torch.no_grad():
            decoder_input = torch.LongTensor([self._tgt_bos_idx] * batch_size).type_as(src_lengths)
            decoder_input = convert_to_onehot(decoder_input, self._tgt_vocab_len, src.device).unsqueeze(0)
            decoder_input = torch.cat((decoder_input, vocab_probs), 0).detach_()

        vocab_probs = self._decode_and_generate(
            decoder_in=decoder_input,
            memory_bank=memory_bank,
            memory_lengths=src_lengths,
            step=None)
        vocab_probs = vocab_probs[1:]
        return vocab_probs, pred_tgt_lens, pred_tgt_ids, None

    def fetch_dec_output_length(self, vocab_probs):
        seq_len = vocab_probs.size(0)
        batch_size = vocab_probs.size(1)
        seq_lens = [seq_len] * batch_size
        pred_ids_tensor = torch.argmax(vocab_probs, 2)
        pred_ids = pred_ids_tensor.transpose(0, 1).tolist()
        max_len_idx = 0
        max_len = 0
        real_pred_ids_cpu = []

        for i in range(batch_size):
            cur_sent = pred_ids[i]
            cur_eos_idx = cur_sent.index(self._tgt_eos_idx) + 1 if self._tgt_eos_idx in cur_sent else seq_len
            if self.sent_stop_aware:
                cur_dot_idx = cur_sent.index(self.dot_idx) + 1 if self.dot_idx in cur_sent else seq_len
                cur_excalmatory_idx = cur_sent.index(
                    self.excalmatory_idx) + 1 if self.excalmatory_idx in cur_sent else seq_len
                cur_interrogation_idx = cur_sent.index(
                    self.interrogation_idx) + 1 if self.interrogation_idx in cur_sent else seq_len
                cur_eos_idx = int(min(cur_eos_idx, cur_dot_idx, cur_excalmatory_idx, cur_interrogation_idx))
            if cur_eos_idx <= 1:
                cur_eos_idx = seq_len
            if cur_eos_idx > seq_len:
                cur_eos_idx = seq_len
            cur_sent_len = cur_eos_idx
            seq_lens[i] = cur_sent_len
            real_pred_ids_cpu.append(cur_sent[:cur_sent_len])
            if seq_lens[i] > max_len:
                max_len_idx = i
                max_len = seq_lens[i]
        # make sure the max_len equals seq_len, so that mask length equals memory_bank
        seq_lens[max_len_idx] = seq_len
        seq_lens_list = seq_lens

        # seq_lens_list: list, [batch_size]
        # pred_ids_tensor: [seq_len, batch_size]
        # real_pred_ids_cpu: [batch_size, seq_len]
        return seq_lens_list, pred_ids_tensor, real_pred_ids_cpu




