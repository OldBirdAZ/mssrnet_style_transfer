# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function
from onmt.encoders.transformer import TransformerEncoder
from torch import nn
from onmt.utils.misc import sequence_mask
import torch


class StyleGenerator2(nn.Module):
    """
    StyleGenerator:
        desired_style + X --> [N, x_len, dim]
    """
    def __init__(self, encoder, h_size, num_styles):
        super(StyleGenerator2, self).__init__()
        assert isinstance(encoder, TransformerEncoder)
        self.style_cls = nn.Embedding(num_styles, h_size)
        self.encoder = encoder

    def forward(self, desired_style, src, src_lengths):
        """
        desired_style: [batch_size]
        src: [seq_len, batch_size, 1]
        src_lengths: [batch_size], value: actual_seq_len

        return: [seq_len, batch_size, emb_size]
        """
        lengths = src_lengths
        self.encoder._check_args(src, lengths)
        emb = self.encoder.embeddings(src)
        cls_emb = self.style_cls(desired_style)
        emb = torch.cat((cls_emb.unsqueeze(0), emb), 0)
        lengths = lengths + 1
        # batch first here
        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths, max_len=src.size(0)+1).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.encoder.transformer:
            out = layer(out, mask)
        out = self.encoder.layer_norm(out)
        out = out.transpose(0, 1).contiguous()
        # out: [seq_len, batch_size, emb_size]
        return out[1:]

