# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.decoder_load_balancing_loss_coeff = args.decoder_load_balancing_loss_coeff
        self.encoder_load_balancing_loss_coeff = args.encoder_load_balancing_loss_coeff
        self.thor_consistency_alpha = args.thor_consistency_alpha

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--encoder-load-balancing-loss-coeff', type=float, default=0.0)
        parser.add_argument('--decoder-load-balancing-loss-coeff', type=float, default=0.0)
        parser.add_argument('--thor-consistency-alpha', type=float, default=0.0)
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        if self.thor_consistency_alpha > 0:
            loss, nll_loss, lprobs, target = self.compute_loss(model, net_output, sample, reduce=reduce)
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        # check to add switch load balancing loss
        if self.encoder_load_balancing_loss_coeff > 0 or self.decoder_load_balancing_loss_coeff > 0:
            switch_lb_loss = 0
            if "encoder_switch_loss_balancing_params" in net_output[1]:
                loss += self.encoder_load_balancing_loss_coeff  * net_output[1]["encoder_switch_loss_balancing_params"]
                switch_lb_loss += net_output[1]["encoder_switch_loss_balancing_params"]
            if "decoder_switch_loss_balancing_params" in net_output[1]:
                loss += self.decoder_load_balancing_loss_coeff  * net_output[1]["decoder_switch_loss_balancing_params"]
                switch_lb_loss += net_output[1]["decoder_switch_loss_balancing_params"]
            logging_output['switch_lb_loss'] = switch_lb_loss.item()

        # check to add thor consistency loss
        if self.thor_consistency_alpha > 0:
            net_output_seed2 = model(**sample['net_input'])
            loss_seed2, nll_loss_seed2, lprobs_seed2, target_seed2 = self.compute_loss(model, net_output_seed2, sample, reduce=reduce)
            # update logging output
            logging_output['loss'] =  0.5 * (logging_output['loss'] + utils.item(loss_seed2.data) if reduce else loss_seed2.data)
            logging_output['nll_loss'] =  0.5 * (logging_output['nll_loss'] + utils.item(nll_loss_seed2.data) if reduce else nll_loss_seed2.data)
            # calculate consistency loss
            consistency_loss = self.symmetric_KL_loss(lprobs, lprobs_seed2, target_seed2, self.padding_idx)
            logging_output['consistency_loss'] = consistency_loss.item()
            loss = (0.5*(loss + loss_seed2)) + (self.thor_consistency_alpha * consistency_loss)

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        if self.thor_consistency_alpha > 0:
            return loss, nll_loss, model.get_normalized_probs(net_output, log_probs=False), target
        return loss, nll_loss

    def symmetric_KL_loss(self, p, q, target, ignore_index):
        """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
        non_pad_mask = target.ne(ignore_index)
        p = p.view(-1, p.size(-1))[non_pad_mask.squeeze()]
        q = q.view(-1, q.size(-1))[non_pad_mask.squeeze()]
        loss = (p - q) * (torch.log(p) - torch.log(q))
        return 0.5 * loss.sum()

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        aggregate_outputs = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        # check to add switch load balancing loss aggregate
        if len(logging_outputs) > 0 and 'switch_lb_loss' in logging_outputs[0]:
            aggregate_outputs['switch_lb_loss'] = sum(log.get('switch_lb_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.
        # check to add consistency loss
        if len(logging_outputs) > 0 and 'consistency_loss' in logging_outputs[0]:
            aggregate_outputs['consistency_loss'] = sum(log.get('consistency_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.


        return aggregate_outputs







