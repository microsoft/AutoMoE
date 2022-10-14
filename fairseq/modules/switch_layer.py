

"""
---
title: Switch Transformer
summary: >
  This is an annotated implementation/tutorial a miniature version of Switch Transformer in PyTorch.
---

# Switch Transformer

This is a miniature [PyTorch](https://pytorch.org) implementation of the paper
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://papers.labml.ai/paper/2101.03961).
Our implementation only has a few million parameters and doesn't do model parallel distributed training.
It does single GPU training, but we implement the concept of switching as described in the paper.

The Switch Transformer uses different parameters for each token by switching among parameters
based on the token.
Therefore, only a fraction of parameters are chosen for each token.
So you can have more parameters but less computational cost.

The switching happens at the Position-wise Feedforward network (FFN) of each transformer block.
Position-wise feedforward network consists of two sequentially fully connected layers.
In switch transformer we have multiple FFNs (multiple experts),
and we chose which one to use based on a router.
The output is a set of probabilities for picking a FFN,
and we pick the one with the highest probability and only evaluate that.
So essentially the computational cost is the same as having a single FFN.
In our implementation this doesn't parallelize well when you have many or large FFNs since it's all
happening on a single GPU.
In a distributed setup you would have each FFN (each very large) on a different device.

The paper introduces another loss term to balance load among the experts (FFNs) and
discusses dropping tokens when routing is not balanced.

Here's [the training code](experiment.html) and a notebook for training a switch transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/353770ce177c11ecaa5fb74452424f46)
"""

import torch
from torch import nn
import torch.nn.functional as F
import copy
from fairseq.modules import (LinearSuper)
import sys
import time

class SwitchFNN(nn.Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(self, *,
                 capacity_factor=None,
                 drop_tokens=None,
                 is_scale_prob=None,
                 super_n_experts=None,
                 super_d_model=None,
                 super_ffn_embed_dim_this_layer=None,
                 uniform_=None, ffn1_non_linear=None, ffn2_non_linear=None, activation_fn=None, expert_dropout_ratio=None, is_first_expert_identity=False):
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `super_n_experts` is the number of experts for super model
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `super_d_model` is the number of features in a token embedding for super model
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.is_scale_prob = is_scale_prob
        self.super_n_experts = super_n_experts
        self.super_d_model = super_d_model
        self.super_ffn_embed_dim_this_layer = super_ffn_embed_dim_this_layer
        self.uniform_ = uniform_
        self.ffn1_non_linear = ffn1_non_linear
        self.ffn2_non_linear = ffn2_non_linear
        self.activation_fn = activation_fn
        self.expert_dropout_ratio = expert_dropout_ratio
        self.is_first_expert_identity = is_first_expert_identity

        # make copies of the FFNs
        self.experts = nn.ModuleList([])
        for i in range(super_n_experts):
          cur_expert_layer = nn.ModuleList([])
          if self.is_first_expert_identity and i == 0:
            cur_expert_layer.append(nn.Identity())
            cur_expert_layer.append(nn.Identity())
            self.experts.append(cur_expert_layer)
            continue
          cur_expert_layer.append(LinearSuper(super_in_dim=self.super_d_model, super_out_dim=self.super_ffn_embed_dim_this_layer, uniform_=self.uniform_, non_linear=self.ffn1_non_linear))
          cur_expert_layer.append(LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_d_model, uniform_=self.uniform_, non_linear=self.ffn2_non_linear))
          self.experts.append(cur_expert_layer)

        # Routing layer and softmax
        self.switch = nn.Linear(super_d_model, super_n_experts)
        self.softmax = nn.Softmax(dim=-1)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_dropout = None
        self.sample_activation_dropout = None
        self.sample_n_experts = None
        
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def set_sample_config(self, sample_embed_dim=None, sample_ffn_embed_dim_this_layer=None, sample_dropout=None, sample_activation_dropout=None, sample_n_experts=None, sample_num_experts_to_route=None, sample_std_vs_dummy_experts=None, sample_each_expert_ffn_dim=None):
      self.sample_embed_dim = sample_embed_dim
      self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
      self.sample_dropout = min(1.0, sample_dropout*self.expert_dropout_ratio)
      self.sample_activation_dropout = min(1.0, sample_activation_dropout*self.expert_dropout_ratio)
      self.sample_n_experts = sample_n_experts
      self.sample_num_experts_to_route = sample_num_experts_to_route


      if sample_std_vs_dummy_experts is not None:
        if sample_std_vs_dummy_experts == 0:
          self.is_first_expert_identity = True
        else:
          self.is_first_expert_identity = False

      for i in range(self.super_n_experts):
        #if self.is_first_expert_identity and i == 0:
        #  continue
        if i < self.sample_n_experts:
          self.experts[i][0].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer if not sample_each_expert_ffn_dim else sample_each_expert_ffn_dim[i])
          self.experts[i][1].set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer if not sample_each_expert_ffn_dim else sample_each_expert_ffn_dim[i], sample_out_dim=self.sample_embed_dim)
        else:
          self.experts[i][0].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer if not sample_each_expert_ffn_dim or i >= len(sample_each_expert_ffn_dim) else sample_each_expert_ffn_dim[i])
          self.experts[i][1].set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer if not sample_each_expert_ffn_dim or i >= len(sample_each_expert_ffn_dim) else sample_each_expert_ffn_dim[i], sample_out_dim=self.sample_embed_dim)

    def router_forward(self, x):
      sample_router_weights = self.switch.weight[0:self.sample_n_experts, 0:self.sample_embed_dim]
      sample_router_bias = self.switch.bias[0:self.sample_n_experts]
      router_logits = F.linear(x, sample_router_weights, sample_router_bias)
      return router_logits

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape  
        """

        if self.sample_num_experts_to_route > 1:
          return self.forward_multiple_num_expert_to_route(x)

        # convert from `[batch_size, seq_len, d_model]` to `[seq_len, batch_size, d_model]`
        # x = x.permute(1, 0, 2)  # input is already [seq_len, batch_size, d_model]

        timer_set = False
        if timer_set:
          main_start_time = time.time()

        # Capture the shape to change shapes later
        seq_len, batch_size, d_model = x.shape
        # Flatten the sequence and batch dimensions
        if timer_set:
          start_time = time.time()
        x = x.reshape(seq_len*batch_size, d_model)
        if timer_set:
          print("--- Flatten the sequence and batch dimensions = %s seconds ---" % (time.time() - start_time))

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        if timer_set:
          start_time = time.time()
        route_prob = self.softmax(self.router_forward(x))
        if timer_set:
          print("--- Get routing probabilities = %s seconds ---" % (time.time() - start_time))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        if timer_set:
          start_time = time.time()
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        if timer_set:
          print("---  route to the expert with highest probability = %s seconds ---" % (time.time() - start_time))

        # Get indexes of tokens going to each expert
        if timer_set:
          start_time = time.time()
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.sample_n_experts)]
        if timer_set:
          print("---  Get indexes of tokens going to each expert = %s seconds ---" % (time.time() - start_time))

        if timer_set:
          start_time = time.time()
        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)
        if timer_set:
          print("---  Initialize an empty tensor = %s seconds ---" % (time.time() - start_time))

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        if timer_set:
          start_time = time.time()
        capacity = int(self.capacity_factor * len(x) / self.sample_n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.sample_n_experts)])
        if timer_set:
          print("---  Capacity of each expert = %s seconds ---" % (time.time() - start_time))

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.sample_n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        # expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.sample_n_experts)]
        if timer_set:
          start_time = time.time()
        expert_output = []
        for i in range(self.sample_n_experts): #TODO: slow in GPU. see deepseed impl. 
          if self.is_first_expert_identity and i == 0:
            expert_output.append(x[indexes_list[i], :])
            continue
          y = self.activation_fn(self.experts[i][0](x[indexes_list[i], :]))
          y = F.dropout(y, p=self.sample_activation_dropout, training=self.training)
          y = self.experts[i][1](y)
          y = F.dropout(y, p=self.sample_dropout, training=self.training)
          expert_output.append(y)

        # Assign to final output
        for i in range(self.sample_n_experts):
            final_output[indexes_list[i], :] = expert_output[i] #.float()
        if timer_set:
          print("--- Get outputs of the expert FFNs = %s seconds ---" % (time.time() - start_time))

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        final_output = final_output.view(seq_len, batch_size, d_model)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging

        # convert from `[seq_len, batch_size, d_model]` to `[batch_size, seq_len, d_model]`
        # final_output = final_output.permute(1, 0, 2)

        if timer_set:
          print("--- Overall main = %s seconds ---" % (time.time() - main_start_time))

        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max

    def forward_multiple_num_expert_to_route(self, x: torch.Tensor):
      seq_len, batch_size, d_model = x.shape
      x = x.reshape(seq_len*batch_size, d_model)
      route_prob = self.softmax(self.router_forward(x))
      global_route_prob_max, global_routes = torch.topk(route_prob, k=self.sample_num_experts_to_route, dim=-1)
      global_final_output = x.new_zeros(x.shape)
      for k in range(self.sample_num_experts_to_route):
        final_output = x.new_zeros(x.shape)
        route_prob_max, routes = global_route_prob_max[:, k], global_routes[:, k]
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.sample_n_experts)]
        capacity = int(self.capacity_factor * len(x) / self.sample_n_experts)
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.sample_n_experts)])
        expert_output = []
        for i in range(self.sample_n_experts):
          if self.is_first_expert_identity and i == 0:
            expert_output.append(x[indexes_list[i], :])
            continue
          y = self.activation_fn(self.experts[i][0](x[indexes_list[i], :]))
          y = F.dropout(y, p=self.sample_activation_dropout, training=self.training)
          y = self.experts[i][1](y)
          y = F.dropout(y, p=self.sample_dropout, training=self.training)
          expert_output.append(y)
        for i in range(self.sample_n_experts):
          final_output[indexes_list[i], :] = expert_output[i] 
        final_output = final_output * route_prob_max.view(-1, 1)
        global_final_output += final_output
      global_final_output = global_final_output.view(seq_len, batch_size, d_model)
      return global_final_output, None, None, None, None


def test():
  import fairseq.init as init
  from fairseq import options, utils
  activation_fn = F.relu
  switch_layer = SwitchFNN(capacity_factor=1.0, drop_tokens=False, is_scale_prob=True, super_n_experts=3, super_d_model=640, super_ffn_embed_dim_this_layer=3072, uniform_=init.uniform_, ffn1_non_linear='relu', ffn2_non_linear='linear', activation_fn=activation_fn, expert_dropout_ratio=1.5, is_first_expert_identity=True)
  #print(switch_layer)
  # supernet
  switch_layer.set_sample_config(640, 3072, 0.1, 0.2, 3, 2)
  print(switch_layer(torch.rand(3,2,640))[0].size())
  return
  # subnet
  switch_layer.set_sample_config(512, 1024, 0.0, 0.0, 2)
  # print(switch_layer(torch.rand(3,2,512))[0].size())
  output, counts, route_prob, num_dropped, route_prob_max = switch_layer(torch.rand(3,2,512))
  print(output.size())
  print(counts)
  print(route_prob)

# test()




