"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from onmt.encoders import str2enc
from onmt.modules import Embeddings, VecEmbedding
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
import copy
from onmt.utils.misc import sequence_mask
from onmt.encoders.transformer import TransformerEncoderLayer
from onmt.style_generator import StyleGenerator2


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    if opt.model_type == "vec" and for_encoder:
        return VecEmbedding(
            opt.feat_vec_size,
            emb_dim,
            position_encoding=opt.position_encoding,
            dropout=(opt.dropout[0] if type(opt.dropout) is list
                     else opt.dropout),
        )

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" \
        or opt.model_type == "vec" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


class SentMeanPoolOutLayer(nn.Module):
    def __init__(self, h_size, num_styles):
        super(SentMeanPoolOutLayer, self).__init__()
        # self.w_out = nn.Sequential(
        #     nn.Linear(h_size, h_size),
        #     nn.ReLU(),
        #     nn.Linear(h_size, num_styles)
        # )
        self.w_out = nn.Linear(h_size, num_styles)

    def forward(self, x_states, x_lengths):
        """
        x_states: batch first
        """
        x_valid_mask = sequence_mask(x_lengths).unsqueeze(2).float()
        xl_float = x_lengths.float().contiguous().unsqueeze(1)
        new_states = x_states * x_valid_mask
        sent_vec = torch.sum(new_states, 1) / xl_float

        logits = self.w_out(sent_vec)
        return logits

class SentMaxPoolOutLayer(nn.Module):
    def __init__(self, h_size, num_styles):
        super(SentMaxPoolOutLayer, self).__init__()
        # self.w_out = nn.Sequential(
        #     nn.Linear(h_size, h_size),
        #     nn.ReLU(),
        #     nn.Linear(h_size, num_styles)
        # )
        self.w_out = nn.Linear(h_size, num_styles)

    def forward(self, x_states, x_lengths):
        """
        x_states: batch first
        """
        x_valid_mask = sequence_mask(x_lengths).unsqueeze(2)
        new_states = x_states.masked_fill(x_valid_mask, float('-inf'))
        sent_vec = torch.max(new_states, 1)[0]

        logits = self.w_out(sent_vec)
        return logits

class SentMixPoolOutLayer(nn.Module):
    def __init__(self, h_size, num_styles):
        super(SentMixPoolOutLayer, self).__init__()
        # self.w_out = nn.Sequential(
        #     nn.Linear(2 * h_size, h_size),
        #     nn.ReLU(),
        #     nn.Linear(h_size, num_styles)
        # )
        self.w_out = nn.Linear(2 * h_size, num_styles)

    def forward(self, x_states, x_lengths):
        """
        x_states: batch first
        """
        x_valid_mask_ori = sequence_mask(x_lengths).unsqueeze(2)
        x_valid_mask = x_valid_mask_ori.float()
        xl_float = x_lengths.float().contiguous().unsqueeze(1)
        new_states = x_states * x_valid_mask
        sent_vec1 = torch.sum(new_states, 1) / xl_float

        new_states = x_states.masked_fill(x_valid_mask_ori, float('-inf'))
        sent_vec2 = torch.max(new_states, 1)[0]

        sent_vec = torch.cat((sent_vec1, sent_vec2), 1)

        logits = self.w_out(sent_vec)
        return logits


class StyleClassifier(nn.Module):
    """
    X --> label

    the ori encoder:
    ------------------------------------------------------
    (encoder): TransformerEncoder(
        (embeddings): Embeddings(
          (make_embedding): Sequential(
            (emb_luts): Elementwise(
              (0): Embedding(9606, 512, padding_idx=3)
            )
            (pe): PositionalEncoding(
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (transformer): ModuleList(
    ------------------------------------------------------
    """
    def __init__(self, encoder, num_styles, opt):
        super(StyleClassifier, self).__init__()
        self.encoder = encoder
        # model_dim = encoder.layer_norm.weight.data.size()[0]
        model_dim = opt.enc_rnn_size
        self.sent_fn = ContextReps2(model_dim)
        self.sent_out_layer = nn.Linear(model_dim, num_styles)

    def do_cls(self, style_representation, src_lengths):
        """
        src: [batch_size, seq_len, model_dim]
        src_lengths: [batch_size]
        """
        sent_vec = self.sent_fn(style_representation, src_lengths)
        logits = self.sent_out_layer(sent_vec)
        return logits

    def forward(self, src, src_lengths):
        """
        src: [seq_len, batch_size, 1]
        src_lengths: [batch_size]
        """
        enc_state, memory_bank, lengths = self.encoder(src, src_lengths)
        # batch first
        style_representation = memory_bank.transpose(0, 1)
        return self.do_cls(style_representation, src_lengths)


class ContextReps(nn.Module):
    def __init__(self, opt):
        super(ContextReps, self).__init__()
        model_dim = opt.enc_rnn_size
        self.cls_vec = nn.Parameter(torch.Tensor(1, 1, model_dim))
        self.sent_fn = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    model_dim,
                    opt.heads,
                    opt.transformer_ff,
                    opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
                    opt.attention_dropout[0] if type(opt.attention_dropout) is list else opt.attention_dropout,
                    opt.max_relative_positions
                )
            ]
        )

    def forward(self, x_states, src_lengths):
        """
        src: [batch_size, seq_len, model_dim]
        src_lengths: [batch_size]

        :return [batch_size, model_dim]
        """
        # INFO need +1 for cls token
        mask = ~sequence_mask(src_lengths + 1).unsqueeze(1)
        out = torch.cat((self.cls_vec.repeat(x_states.size(0), 1, 1), x_states), 1)
        for layer in self.sent_fn:
            out = layer(out, mask)
        return out[:, 0, :]


class ContextReps2(nn.Module):
    def __init__(self, in_size, hidden_size=None):
        super(ContextReps2, self).__init__()
        # model_dim = opt.enc_rnn_size
        # model_dim = opt.n_heads_sgenerator * opt.dim_each_head_sgenerator
        model_dim = in_size
        if hidden_size is None:
            hidden_size = int(model_dim / 2) + 1
        self.w_fn = nn.Sequential(
            nn.Linear(model_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x_states, src_lengths):
        """
        src: [batch_size, seq_len, model_dim]
        src_lengths: [batch_size]

        :return [batch_size, model_dim]
        """
        # [N, x_len, 1]
        w = self.w_fn(x_states)
        mask = ~sequence_mask(src_lengths, max_len=w.size(1)).unsqueeze(2)
        w.masked_fill_(mask, float('-inf'))
        ww = torch.softmax(w, 1)
        vec = torch.sum(ww * x_states, 1)
        return vec


class StyleDiscriminator(nn.Module):
    def __init__(self, num_styles, opt):
        super(StyleDiscriminator, self).__init__()
        style_dim = opt.n_heads_sgenerator * opt.dim_each_head_sgenerator
        hidden_size = opt.n_heads_sdiscr * opt.dim_each_head_sdiscr
        self.sent_fn = ContextReps2(style_dim, hidden_size=hidden_size)
        assert opt.n_layers_sdiscr >= 1
        middle_layers = []
        for i in range(opt.n_layers_sdiscr):
            middle_layers.append(nn.Linear(hidden_size if i != 0 else style_dim, hidden_size))
            middle_layers.append(nn.ReLU())
        self.sent_middle_fn = nn.Sequential(
            *middle_layers
        )
        self.sent_out_layer = nn.Linear(hidden_size, num_styles)

    def forward(self, x_states, src_lengths):
        """
        src: [batch_size, seq_len, model_dim]
        src_lengths: [batch_size]

        :return [batch_size, num_styles]
        """
        style_representation = x_states
        sent_vec = self.sent_fn(style_representation, src_lengths)
        sent_vec = self.sent_middle_fn(sent_vec)
        logits = self.sent_out_layer(sent_vec)
        return logits


def build_classifier_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None, force_num_style=None):
    old_enc_layers = model_opt.enc_layers
    old_enc_rnn_size = model_opt.enc_rnn_size
    old_heads = model_opt.heads
    old_transformer_ff = model_opt.transformer_ff
    old_src_word_vec_size = model_opt.src_word_vec_size
    old_tgt_word_vec_size = model_opt.tgt_word_vec_size
    # # params about encoder:
    # opt.enc_layers,
    # opt.enc_rnn_size,
    # opt.heads,
    # opt.transformer_ff,
    n_layers_cls = model_opt.n_layers_cls
    n_heads_cls = model_opt.n_heads_cls
    dim_each_head_cls = model_opt.dim_each_head_cls
    emb_dim_cls = n_heads_cls * dim_each_head_cls
    h_size = n_heads_cls * dim_each_head_cls
    # resetting
    model_opt.enc_layers = n_layers_cls
    model_opt.enc_rnn_size = h_size
    model_opt.heads = n_heads_cls
    model_opt.transformer_ff = h_size * 4
    model_opt.src_word_vec_size = emb_dim_cls
    model_opt.tgt_word_vec_size = emb_dim_cls

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build embeddings.
    if model_opt.model_type == "text" or model_opt.model_type == "vec":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)

    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")

    num_styles = model_opt.num_styles if force_num_style is None else force_num_style
    model = StyleClassifier(encoder, num_styles, model_opt)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)

    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()

    model_opt.enc_layers = old_enc_layers
    model_opt.enc_rnn_size = old_enc_rnn_size
    model_opt.heads = old_heads
    model_opt.transformer_ff = old_transformer_ff
    model_opt.src_word_vec_size = old_src_word_vec_size
    model_opt.tgt_word_vec_size = old_tgt_word_vec_size
    return model


def build_text_discriminator_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None, force_num_style=None):
    old_enc_layers = model_opt.enc_layers
    old_enc_rnn_size = model_opt.enc_rnn_size
    old_heads = model_opt.heads
    old_transformer_ff = model_opt.transformer_ff
    old_src_word_vec_size = model_opt.src_word_vec_size
    old_tgt_word_vec_size = model_opt.tgt_word_vec_size

    n_layers_cls = model_opt.n_layers_tdiscr
    n_heads_cls = model_opt.n_heads_tdiscr
    dim_each_head_cls = model_opt.dim_each_head_tdiscr
    emb_dim_cls = n_heads_cls * dim_each_head_cls
    h_size = n_heads_cls * dim_each_head_cls
    # resetting
    model_opt.enc_layers = n_layers_cls
    model_opt.enc_rnn_size = h_size
    model_opt.heads = n_heads_cls
    model_opt.transformer_ff = h_size * 4
    model_opt.src_word_vec_size = emb_dim_cls
    model_opt.tgt_word_vec_size = emb_dim_cls

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build embeddings.
    if model_opt.model_type == "text" or model_opt.model_type == "vec":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)

    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")

    num_styles = model_opt.num_styles if force_num_style is None else force_num_style
    model = StyleClassifier(encoder, num_styles, model_opt)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)

    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()

    model_opt.enc_layers = old_enc_layers
    model_opt.enc_rnn_size = old_enc_rnn_size
    model_opt.heads = old_heads
    model_opt.transformer_ff = old_transformer_ff
    model_opt.src_word_vec_size = old_src_word_vec_size
    model_opt.tgt_word_vec_size = old_tgt_word_vec_size
    return model


def build_classifier_model(model_opt, opt, fields, checkpoint):
    logger.info('Building classifier model...')
    model = build_classifier_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    if opt.show_model_struct:
        logger.info(model)
    return model


def build_style_discriminator_model(model_opt, opt, fields, checkpoint):
    model_opt, opt = copy.deepcopy(model_opt), copy.deepcopy(opt)

    n_layers_cls = model_opt.n_layers_sdiscr
    n_heads_cls = model_opt.n_heads_sdiscr
    dim_each_head_cls = model_opt.dim_each_head_sdiscr
    emb_dim_cls = n_heads_cls * dim_each_head_cls
    h_size = n_heads_cls * dim_each_head_cls
    # resetting
    model_opt.enc_layers = n_layers_cls
    model_opt.enc_rnn_size = h_size
    model_opt.heads = n_heads_cls
    model_opt.transformer_ff = h_size * 4
    model_opt.src_word_vec_size = emb_dim_cls
    model_opt.tgt_word_vec_size = emb_dim_cls

    if opt.faketruthstyle_mode:
        num_styles = model_opt.num_styles * 2
    else:
        num_styles = model_opt.num_styles

    model = StyleDiscriminator(num_styles, model_opt)

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
    model = model.to(device)
    return model


def build_text_discriminator_model(model_opt, opt, fields, checkpoint):
    model_opt, opt = copy.deepcopy(model_opt), copy.deepcopy(opt)
    logger.info('Building classifier model...')
    if opt.faketruthstyle_mode:
        num_styles = model_opt.num_styles * 2
    else:
        num_styles = model_opt.num_styles
    model = build_text_discriminator_base_model(model_opt, fields, use_gpu(opt), checkpoint, force_num_style=num_styles)
    if opt.show_model_struct:
        logger.info(model)
    return model


class BondLayer(nn.Module):
    def __init__(self, h_size, model_dim):
        super(BondLayer, self).__init__()
        self.w = nn.Linear(h_size, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, memory_bank, style_represents):
        sf = self.w(style_represents)
        sf = self.layer_norm(sf)
        return memory_bank + sf


def build_style_generator_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    old_enc_layers = model_opt.enc_layers
    old_enc_rnn_size = model_opt.enc_rnn_size
    old_heads = model_opt.heads
    old_transformer_ff = model_opt.transformer_ff
    old_src_word_vec_size = model_opt.src_word_vec_size
    old_tgt_word_vec_size = model_opt.tgt_word_vec_size
    # # params about encoder:
    # opt.enc_layers,
    # opt.enc_rnn_size,
    # opt.heads,
    # opt.transformer_ff,
    n_layers_cls = model_opt.n_layers_sgenerator
    n_heads_cls = model_opt.n_heads_sgenerator
    dim_each_head_cls = model_opt.dim_each_head_sgenerator
    emb_dim_cls = n_heads_cls * dim_each_head_cls
    h_size = n_heads_cls * dim_each_head_cls
    # resetting
    model_opt.enc_layers = n_layers_cls
    model_opt.enc_rnn_size = h_size
    model_opt.heads = n_heads_cls
    model_opt.transformer_ff = h_size * 4
    model_opt.src_word_vec_size = emb_dim_cls
    model_opt.tgt_word_vec_size = emb_dim_cls

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build embeddings.
    if model_opt.model_type == "text" or model_opt.model_type == "vec":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)
    num_styles = model_opt.num_styles
    model = StyleGenerator2(encoder, h_size, num_styles)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)

    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()

    model_opt.enc_layers = old_enc_layers
    model_opt.enc_rnn_size = old_enc_rnn_size
    model_opt.heads = old_heads
    model_opt.transformer_ff = old_transformer_ff
    model_opt.src_word_vec_size = old_src_word_vec_size
    model_opt.tgt_word_vec_size = old_tgt_word_vec_size
    return model


def build_style_generator_model(model_opt, opt, fields, checkpoint):
    logger.info('Building style generator model...')
    model = build_style_generator_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    if opt.show_model_struct:
        logger.info(model)
    return model











