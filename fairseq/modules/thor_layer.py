'''
https://github.com/microsoft/Stochastic-Mixture-of-Experts
'''

import torch
from torch import nn
import torch.nn.functional as F
import copy
from fairseq.modules import (LinearSuper)
import sys

class ThorFNN(nn.Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(self, *,
                 super_n_experts=None,
                 super_d_model=None,
                 super_ffn_embed_dim_this_layer=None,
                 uniform_=None, ffn1_non_linear=None, ffn2_non_linear=None, activation_fn=None):
        """
        * `super_n_experts` is the number of experts for super model
        * `super_d_model` is the number of features in a token embedding for super model
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.super_n_experts = super_n_experts
        self.super_d_model = super_d_model
        self.super_ffn_embed_dim_this_layer = super_ffn_embed_dim_this_layer
        self.uniform_ = uniform_
        self.ffn1_non_linear = ffn1_non_linear
        self.ffn2_non_linear = ffn2_non_linear
        self.activation_fn = activation_fn

        # make copies of the FFNs
        self.experts = nn.ModuleList([])
        for i in range(super_n_experts):
          cur_expert_layer = nn.ModuleList([])
          cur_expert_layer.append(LinearSuper(super_in_dim=self.super_d_model, super_out_dim=self.super_ffn_embed_dim_this_layer, uniform_=self.uniform_, non_linear=self.ffn1_non_linear))
          cur_expert_layer.append(LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_d_model, uniform_=self.uniform_, non_linear=self.ffn2_non_linear))
          self.experts.append(cur_expert_layer)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_dropout = None
        self.sample_activation_dropout = None
        self.sample_n_experts = None
        
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def set_sample_config(self, sample_embed_dim=None, sample_ffn_embed_dim_this_layer=None, sample_dropout=None, sample_activation_dropout=None, sample_n_experts=None):
      self.sample_embed_dim = sample_embed_dim
      self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
      self.sample_dropout = sample_dropout
      self.sample_activation_dropout = sample_activation_dropout
      self.sample_n_experts = sample_n_experts

      for i in range(self.super_n_experts):
        self.experts[i][0].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.experts[i][1].set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_embed_dim)

    def forward(self, x: torch.Tensor):
      """
      * `x` is the input to the switching module with shape  
      most parts taken from: https://github.com/microsoft/Stochastic-Mixture-of-Experts/blob/main/thor/transformer_thor_layer.py
      """

      # x is (seq_len, batch_size, d_model)
      seq_len, batch_size, d_model = x.shape

      if self.training:
        # randomly chosen expert
        random_expert_i = torch.randint(low=0, high=self.sample_n_experts, size=(1,), device=x.device).item()
        # route all tokens to the same expert
        y = self.activation_fn(self.experts[random_expert_i][0](x))
        y = F.dropout(y, p=self.sample_activation_dropout, training=self.training)
        y = self.experts[random_expert_i][1](y)
        y = F.dropout(y, p=self.sample_dropout, training=self.training)
        return y

      # todo: optimize the code for FLOPs match by borrowing from switch layer
      result = []
      for expert in range(self.sample_n_experts):
        temp = self.activation_fn(self.experts[expert][0](x))
        temp = F.dropout(temp, p=self.sample_activation_dropout, training=self.training)
        temp = self.experts[expert][1](temp)
        temp = F.dropout(temp, p=self.sample_dropout, training=self.training)
        result.append(temp)
      result = torch.stack(result, dim=0)
      mask = torch.randint(0, self.sample_n_experts, size=(x.size(0), x.size(1)), device=result.device)
      for i in range(self.sample_n_experts):
        expert_mask = mask.eq(i)
        result[i] *= expert_mask.unsqueeze(-1)
      y = result.sum(0)
      return y

def test():
  import fairseq.init as init
  from fairseq import options, utils
  activation_fn = F.relu
  thor_layer = ThorFNN(super_n_experts=2, super_d_model=5, super_ffn_embed_dim_this_layer=8, uniform_=init.uniform_, ffn1_non_linear='relu', ffn2_non_linear='linear', activation_fn=activation_fn)
  #print(thor_layer)
  # supernet
  thor_layer.set_sample_config(5, 8, 0.0, 0.0, 2)
  thor_layer.eval()
  print(thor_layer(torch.rand(3,2,5)))
  # subnet
  #thor_layer.set_sample_config(512, 1024, 0.0, 0.0, 2)
  # print(thor_layer(torch.rand(3,2,512))[0].size())
  #output = thor_layer(torch.rand(3,2,512))
  # print(output.size())

# test()









