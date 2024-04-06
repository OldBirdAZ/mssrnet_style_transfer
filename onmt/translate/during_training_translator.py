""" Translator builder used during training """
from __future__ import print_function

from onmt.translate.translator_wh import Translator
from onmt.translate import GNMTGlobalScorer


def build_translator_for_train(opt, model_opt, model, style_generator, fields):

    trans_alpha = opt.trans_alpha
    trans_beta = opt.trans_beta
    trans_length_penalty = opt.trans_length_penalty
    trans_coverage_penalty = opt.trans_coverage_penalty
    trans_n_best = opt.trans_n_best
    trans_min_length = opt.trans_min_length
    trans_max_length = opt.trans_max_length
    trans_ratio = opt.trans_ratio
    trans_beam_size = opt.trans_beam_size
    trans_random_sampling_topk = opt.trans_random_sampling_topk
    trans_random_sampling_temp = opt.trans_random_sampling_temp
    trans_stepwise_penalty = opt.trans_stepwise_penalty
    trans_block_ngram_repeat = opt.trans_block_ngram_repeat
    trans_ignore_when_blocking = opt.trans_ignore_when_blocking
    trans_replace_unk = opt.trans_replace_unk
    trans_phrase_table = opt.trans_phrase_table
    trans_tgt_prefix = False
    trans_data_type = "text"
    trans_report_time = False
    # scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
    scorer = GNMTGlobalScorer(trans_alpha, trans_beta, trans_length_penalty, trans_coverage_penalty)
    desired_style = None
    src_reader = None
    tgt_reader = None

    new_translator = Translator(
        model,
        style_generator,
        desired_style,
        fields,
        src_reader,
        tgt_reader,
        # gpu=opt.gpu,
        gpu=opt.gpu_ranks[0] if len(opt.gpu_ranks) > 0 else -1,
        n_best=trans_n_best,
        min_length=trans_min_length,
        max_length=trans_max_length,
        ratio=trans_ratio,
        beam_size=trans_beam_size,
        random_sampling_topk=trans_random_sampling_topk,
        random_sampling_temp=trans_random_sampling_temp,
        stepwise_penalty=trans_stepwise_penalty,
        dump_beam=False,
        block_ngram_repeat=trans_block_ngram_repeat,
        ignore_when_blocking=set(trans_ignore_when_blocking),
        replace_unk=trans_replace_unk,
        tgt_prefix=trans_tgt_prefix,
        phrase_table=trans_phrase_table,
        data_type=trans_data_type,
        verbose=False,
        report_time=trans_report_time,
        copy_attn=model_opt.copy_attn,
        global_scorer=scorer,
        out_file=None,
        report_align=False,
        report_score=False,
        logger=None,
        seed=opt.seed
    )
    return new_translator






