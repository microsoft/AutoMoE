# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

# AutoMoE: Neural Architecture Search for Efficient Sparsely Activated Transformers
# Ganesh Jawahar, Subhabrata Mukherjee, Xiaodong Liu, Young Jin Kim, Muhammad Abdul-Mageed, Laks V. S. Lakshmanan, Ahmed Hassan Awadallah, Sebastien Bubeck, Jianfeng Gao
# Paper: https://arxiv.org/abs/2210.07535

import torch
import random
import torchprofile
import numpy as np
import fairseq.utils as utils
import time

from fairseq import progress_bar, bleu
from fairseq.data import dictionary
from latency_predictor import LatencyPredictor


class Converter(object):
    def __init__(self, args):
        self.args = args
        self.super_encoder_layer_num = args.encoder_layers
        self.super_decoder_layer_num = args.decoder_layers

        self.encoder_embed_choice = args.encoder_embed_choice
        self.decoder_embed_choice = args.decoder_embed_choice

        self.encoder_ffn_embed_dim_choice = args.encoder_ffn_embed_dim_choice
        self.decoder_ffn_embed_dim_choice = args.decoder_ffn_embed_dim_choice

        self.encoder_layer_num_choice = args.encoder_layer_num_choice
        self.decoder_layer_num_choice = args.decoder_layer_num_choice

        self.encoder_self_attention_heads_choice = args.encoder_self_attention_heads_choice
        self.decoder_self_attention_heads_choice = args.decoder_self_attention_heads_choice
        self.decoder_ende_attention_heads_choice = args.decoder_ende_attention_heads_choice
        self.encoder_drop_mha_sublayer = args.encoder_drop_mha_sublayer
        self.encoder_drop_ffn_sublayer = args.encoder_drop_ffn_sublayer
        self.decoder_drop_mha_sublayer = args.decoder_drop_mha_sublayer
        self.decoder_drop_ffn_sublayer = args.decoder_drop_ffn_sublayer

        self.decoder_arbitrary_ende_attn_choice = args.decoder_arbitrary_ende_attn_choice

        if len(args.encoder_n_experts) > 0:
            self.encoder_n_experts = args.encoder_n_experts
            self.is_expert_in_encoder = True
            self.encoder_expert_all_fixed_num_experts = args.encoder_expert_all_fixed_num_experts
            self.encoder_num_experts_to_route = args.encoder_num_experts_to_route

            self.encoder_std_vs_dummy_experts = args.encoder_std_vs_dummy_experts
            self.encoder_each_expert_ffn_dim = args.encoder_each_expert_ffn_dim

        if len(args.decoder_n_experts) > 0:
            self.decoder_n_experts = args.decoder_n_experts
            self.is_expert_in_decoder = True
            self.decoder_expert_all_fixed_num_experts = args.decoder_expert_all_fixed_num_experts
            self.decoder_num_experts_to_route = args.decoder_num_experts_to_route

            self.decoder_std_vs_dummy_experts = args.decoder_std_vs_dummy_experts
            self.decoder_each_expert_ffn_dim = args.decoder_each_expert_ffn_dim

        self.gene2config_bothways = {} # decoder_each_expert_ffn_dim/encoder_each_expert_ffn_dim mutate/crossover help due to list 


    def config2gene(self, config):
        gene = []

        sample_encoder_layer_num = config['encoder']['encoder_layer_num']

        gene.append(config['encoder']['encoder_embed_dim'])
        gene.append(sample_encoder_layer_num)

        for i in range(self.super_encoder_layer_num):
            if i < sample_encoder_layer_num:
                gene.append(config['encoder']['encoder_ffn_embed_dim'][i])
            else:
                gene.append(config['encoder']['encoder_ffn_embed_dim'][0])

        for i in range(self.super_encoder_layer_num):
            if i < sample_encoder_layer_num:
                gene.append(config['encoder']['encoder_self_attention_heads'][i])
            else:
                gene.append(config['encoder']['encoder_self_attention_heads'][0])


        sample_decoder_layer_num = config['decoder']['decoder_layer_num']

        gene.append(config['decoder']['decoder_embed_dim'])
        gene.append(sample_decoder_layer_num)

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_ffn_embed_dim'][i])
            else:
                gene.append(config['decoder']['decoder_ffn_embed_dim'][0])

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_self_attention_heads'][i])
            else:
                gene.append(config['decoder']['decoder_self_attention_heads'][0])

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_ende_attention_heads'][i])
            else:
                gene.append(config['decoder']['decoder_ende_attention_heads'][0])

        for i in range(self.super_decoder_layer_num):
            gene.append(config['decoder']['decoder_arbitrary_ende_attn'][i])

        #if self.encoder_expert:
        #    gene.append()

        return gene

    def gene2config(self, gene):

        config = {
            'encoder': {
                'encoder_embed_dim': None,
                'encoder_layer_num': None,
                'encoder_ffn_embed_dim': None,
                'encoder_self_attention_heads': None,
            },
            'decoder': {
                'decoder_embed_dim': None,
                'decoder_layer_num': None,
                'decoder_ffn_embed_dim': None,
                'decoder_self_attention_heads': None,
                'decoder_ende_attention_heads': None,
                'decoder_arbitrary_ende_attn': None
            }
        }
        current_index = 0


        config['encoder']['encoder_embed_dim'] = gene[current_index]
        current_index += 1

        config['encoder']['encoder_layer_num'] = gene[current_index]
        current_index += 1

        config['encoder']['encoder_ffn_embed_dim'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num

        config['encoder']['encoder_self_attention_heads'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num


        config['decoder']['decoder_embed_dim'] = gene[current_index]
        current_index += 1

        config['decoder']['decoder_layer_num'] = gene[current_index]
        current_index += 1

        config['decoder']['decoder_ffn_embed_dim'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_self_attention_heads'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_ende_attention_heads'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_arbitrary_ende_attn'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        if hasattr(self, 'is_expert_in_encoder'):
            config['encoder']['encoder_n_experts'] = gene[current_index: current_index + self.super_encoder_layer_num]
            current_index += self.super_encoder_layer_num

        if hasattr(self, 'is_expert_in_decoder'):
            config['decoder']['decoder_n_experts'] = gene[current_index: current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

        if hasattr(self, 'is_expert_in_encoder'):
            config['encoder']['encoder_num_experts_to_route'] = gene[current_index: current_index + self.super_encoder_layer_num]
            current_index += self.super_encoder_layer_num

        if hasattr(self, 'is_expert_in_decoder'):
            config['decoder']['decoder_num_experts_to_route'] = gene[current_index: current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

        if len(self.encoder_drop_mha_sublayer) > 1:
            config['encoder']['encoder_drop_mha_sublayer'] = gene[current_index: current_index + self.super_encoder_layer_num]
            current_index += self.super_encoder_layer_num

        if len(self.encoder_drop_ffn_sublayer) > 1:
            config['encoder']['encoder_drop_ffn_sublayer'] = gene[current_index: current_index + self.super_encoder_layer_num]
            current_index += self.super_encoder_layer_num

        if len(self.decoder_drop_mha_sublayer) > 1:
            config['decoder']['decoder_drop_mha_sublayer'] = gene[current_index: current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

        if len(self.decoder_drop_ffn_sublayer) > 1:
            config['decoder']['decoder_drop_ffn_sublayer'] = gene[current_index: current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

        if hasattr(self, 'encoder_std_vs_dummy_experts') and len(self.encoder_std_vs_dummy_experts) > 1:
            config['encoder']['encoder_std_vs_dummy_experts'] = gene[current_index: current_index + self.super_encoder_layer_num]
            current_index += self.super_encoder_layer_num

        if hasattr(self, 'encoder_each_expert_ffn_dim') and len(self.encoder_each_expert_ffn_dim) > 1:
            config['encoder']['encoder_each_expert_ffn_dim'] = gene[current_index: current_index + self.super_encoder_layer_num]
            current_index += self.super_encoder_layer_num

        if hasattr(self, 'decoder_std_vs_dummy_experts') and len(self.decoder_std_vs_dummy_experts) > 1:
            config['decoder']['decoder_std_vs_dummy_experts'] = gene[current_index: current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

        if hasattr(self, 'decoder_each_expert_ffn_dim') and len(self.decoder_each_expert_ffn_dim) > 1:
            config['decoder']['decoder_each_expert_ffn_dim'] = gene[current_index: current_index + self.super_decoder_layer_num]
            current_index += self.super_decoder_layer_num

        return config


    def get_gene_choice(self):
        gene_choice = []

        gene_choice.append(self.encoder_embed_choice)
        gene_choice.append(self.encoder_layer_num_choice)

        for i in range(self.super_encoder_layer_num):
            gene_choice.append(self.encoder_ffn_embed_dim_choice)
        self.gene2config_bothways["encoder_ffn_embed_dim_choice"] = self.encoder_ffn_embed_dim_choice

        for i in range(self.super_encoder_layer_num):
            gene_choice.append(self.encoder_self_attention_heads_choice)


        gene_choice.append(self.decoder_embed_choice)
        gene_choice.append(self.decoder_layer_num_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_ffn_embed_dim_choice)
        self.gene2config_bothways["decoder_ffn_embed_dim_choice"] = self.decoder_ffn_embed_dim_choice

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_self_attention_heads_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_ende_attention_heads_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_arbitrary_ende_attn_choice)

        if hasattr(self, 'is_expert_in_encoder'):
            if len(self.encoder_expert_all_fixed_num_experts) == 0:
                self.gene2config_bothways["encoder_expert"] = []
                for i in range(self.super_encoder_layer_num):
                    self.gene2config_bothways["encoder_expert"].append(len(gene_choice)+i) 
                for i in range(self.super_encoder_layer_num):
                    gene_choice.append(self.encoder_n_experts)
            else:
                print('constraining encoder n-experts')
                self.gene2config_bothways["encoder_expert"] = []
                for i in range(self.encoder_expert_all_fixed_num_experts):
                    self.gene2config_bothways["encoder_expert"].append(len(gene_choice)+i) 
                for num_expert in self.encoder_expert_all_fixed_num_experts:
                    gene_choice.append([num_expert])

        if hasattr(self, 'is_expert_in_decoder'):
            if len(self.decoder_expert_all_fixed_num_experts) == 0:
                self.gene2config_bothways["decoder_expert"] = []
                for i in range(self.super_decoder_layer_num):
                    self.gene2config_bothways["decoder_expert"].append(len(gene_choice)+i) 
                for i in range(self.super_decoder_layer_num):
                    gene_choice.append(self.decoder_n_experts)
            else:
                print('constraining decoder n-experts')
                self.gene2config_bothways["decoder_expert"] = []
                for i in range(self.decoder_expert_all_fixed_num_experts):
                    self.gene2config_bothways["decoder_expert"].append(len(gene_choice)+i) 
                for num_expert in self.decoder_expert_all_fixed_num_experts:
                    gene_choice.append([num_expert])

        if hasattr(self, 'is_expert_in_encoder'):
            if len(self.encoder_expert_all_fixed_num_experts) == 0:
                for i in range(self.super_encoder_layer_num):
                    gene_choice.append(self.encoder_num_experts_to_route)
            else:
                print('constraining encoder n-experts')
                for num_expert in self.encoder_expert_all_fixed_num_experts:
                    gene_choice.append([num_expert for x in self.encoder_num_experts_to_route if x <= num_expert]) # todo: why x<=num_expert, not just x

        if hasattr(self, 'is_expert_in_decoder'):
            if len(self.decoder_expert_all_fixed_num_experts) == 0:
                for i in range(self.super_decoder_layer_num):
                    gene_choice.append(self.decoder_num_experts_to_route)
            else:
                print('constraining decoder n-experts')
                for num_expert in self.decoder_expert_all_fixed_num_experts:
                    gene_choice.append([num_expert for x in self.decoder_num_experts_to_route if x <= num_expert]) 

        if len(self.encoder_drop_mha_sublayer) > 1:
            for i in range(self.super_encoder_layer_num):
                gene_choice.append(self.encoder_drop_mha_sublayer)

        if len(self.encoder_drop_ffn_sublayer) > 1:
            for i in range(self.super_encoder_layer_num):
                gene_choice.append(self.encoder_drop_ffn_sublayer)

        if len(self.decoder_drop_mha_sublayer) > 1:
            for i in range(self.super_decoder_layer_num):
                gene_choice.append(self.decoder_drop_mha_sublayer)

        if len(self.decoder_drop_ffn_sublayer) > 1:
            for i in range(self.super_decoder_layer_num):
                gene_choice.append(self.decoder_drop_ffn_sublayer)

        if hasattr(self, 'encoder_std_vs_dummy_experts') and len(self.encoder_std_vs_dummy_experts) > 1:
            for i in range(self.super_encoder_layer_num):
                gene_choice.append(self.encoder_std_vs_dummy_experts)

        self.gene2config_bothways["encoder_each_expert_ffn_dim"] = {}
        if hasattr(self, 'encoder_each_expert_ffn_dim') and len(self.encoder_each_expert_ffn_dim) > 1: 
            for i in range(self.super_encoder_layer_num):
                self.gene2config_bothways["encoder_each_expert_ffn_dim"][len(gene_choice)+i] = self.gene2config_bothways["encoder_expert"][i] 
            for i in range(self.super_encoder_layer_num):
                gene_choice.append([self.encoder_ffn_embed_dim_choice])

        if hasattr(self, 'decoder_std_vs_dummy_experts') and len(self.decoder_std_vs_dummy_experts) > 1:
            for i in range(self.super_decoder_layer_num):
                gene_choice.append(self.decoder_std_vs_dummy_experts)

        self.gene2config_bothways["decoder_each_expert_ffn_dim"] = {}
        if hasattr(self, 'decoder_each_expert_ffn_dim') and  len(self.decoder_each_expert_ffn_dim) > 1:
            for i in range(self.super_decoder_layer_num):
                self.gene2config_bothways["decoder_each_expert_ffn_dim"][len(gene_choice)+i] = self.gene2config_bothways["decoder_expert"][i]
            for i in range(self.super_decoder_layer_num):
                gene_choice.append([self.decoder_ffn_embed_dim_choice])

        print("search space = ", gene_choice)
        print("search space length = %d"%len(gene_choice))
        print("gene2config_bothways = ", self.gene2config_bothways)

        return gene_choice


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Evolution(object):
    def __init__(self, args, trainer, task, epoch_iter, generator=None):
        self.population_size = args.population_size
        self.args = args
        self.parent_size = args.parent_size
        self.mutation_size = args.mutation_size
        self.mutation_prob = args.mutation_prob
        self.crossover_size = args.crossover_size
        assert self.population_size == self.parent_size + self.mutation_size + self.crossover_size
        self.converter = Converter(args)
        self.gene_choice = self.converter.get_gene_choice()
        self.gene_len = len(self.gene_choice)
        self.evo_iter = args.evo_iter
        self.trainer = trainer
        self.task=task
        self.epoch_iter = epoch_iter
        if args.latency_compute == "predictor":
            self.latency_predictor = LatencyPredictor(
                feature_norm=args.feature_norm,
                lat_norm=args.lat_norm,
                ckpt_path=args.ckpt_path,
                feature_dim=len(args.feature_norm)
            )
            self.latency_predictor.load_ckpt()
        self.latency_constraint = args.latency_constraint
        self.generator = generator

        self.best_config = None

        self.validation_metric = args.validation_metric
        self.flops_constraint_giga = args.flops_constraint_giga
        self.latency_compute = args.latency_compute

        # specify the length of the dummy input for profile
        # for iwslt, the average length is 23, for wmt, that is 30
        dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
        if 'iwslt' in args.arch:
            dummy_sentence_length = dummy_sentence_length_dict['iwslt']
        elif 'wmt' in args.arch:
            dummy_sentence_length = dummy_sentence_length_dict['wmt']
        else:
            raise NotImplementedError

        self.dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
        self.dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
        self.dummy_src_tokens_tensor = torch.tensor([self.dummy_src_tokens], dtype=torch.long).to('cuda')
        self.dummy_src_len_tensor = torch.tensor([30]).to('cuda')
        self.dummy_prev_tokens_tensor = torch.tensor([self.dummy_prev], dtype=torch.long).to('cuda')
        self.dummy_sentence_length = dummy_sentence_length

        # todo make it modifable from command line
        # todo max iterations = 5
        self.extra_args = {"latcpu": False, "latgpu": True, "beam": 5, "latiter": args.latiter}

    def run_evo_search(self):
        start_time = time.time()
        popu = self.random_sample(self.population_size)

        all_scores_list = []

        '''
        # remove duplicates from popu
        existing_candidates_hash = {}
        new_popu = []
        for cand in popu:
            if str(cand) in existing_candidates_hash:
                continue
            existing_candidates_hash[str(cand)] = True
            new_popu.append(cand)
        '''
        final_population = {}
        for i in range(self.evo_iter):
            print(f"| Start Iteration {i}:")
            popu_scores = self.get_scores(popu)
            print(f"| Iteration {i}, Lowest loss: {min(popu_scores)}")

            if self.validation_metric == "bleu" or self.validation_metric == "active_nonemb_params" or self.validation_metric == "nonemb_params":
                sorted_ind = np.array(popu_scores).argsort()[-self.parent_size:] # changed for BLEU
            elif self.validation_metric == "loss":
                sorted_ind = np.array(popu_scores).argsort()[:self.parent_size]

            self.best_config = self.converter.gene2config(popu[sorted_ind[0]])
            print(f"| Config for lowest loss model: {self.best_config}")
            if self.latency_compute == "predictor":
                print(f"| Predicted latency for lowest loss model: {self.latency_predictor.predict_lat(self.converter.gene2config(popu[sorted_ind[0]]))}")
            print(f"| Real latency for lowest loss model: {self.get_real_latency(self.converter.gene2config(popu[sorted_ind[0]]))}")
            print(self.converter.gene2config(popu[sorted_ind[0]]))
            print(f"| FLOPs for lowest loss model: {self.get_macs(self.converter.gene2config(popu[sorted_ind[0]]))}")

            parents_popu = [popu[m] for m in sorted_ind]

            parents_score = [popu_scores[m] for m in sorted_ind]

            if i == self.evo_iter-1:
                print("--- Time taken to do evolutionary search = %s seconds ---" % (time.time() - start_time))
                idx = 0
                for m in sorted_ind:
                    final_population[idx] = {}
                    final_population[idx]["supernet_val_score"] = popu_scores[m]
                    final_population[idx]["gene_info"] = popu[m]
                    cur_config = self.converter.gene2config(popu[m])
                    if self.latency_compute == "predictor":
                        final_population[idx]["predicted_latency"] = self.latency_predictor.predict_lat(cur_config)
                    final_population[idx]["real_latency"] = self.get_real_latency(cur_config)
                    final_population[idx]["macs"] = self.get_macs(cur_config)
                    final_population[idx]["model_size"] = self.trainer.model.get_sampled_params_numel(cur_config)
                    final_population[idx]["config_info"] = cur_config                    
                    idx += 1

            all_scores_list.append(parents_score)

            mutate_popu = []

            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutate(random.choices(parents_popu)[0])
                if self.satisfy_constraints(mutated_gene): # and str(mutated_gene) not in existing_candidates_hash:
                    mutate_popu.append(mutated_gene)
                    k += 1


            crossover_popu = []

            k = 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(random.sample(parents_popu, 2))
                if self.satisfy_constraints(crossovered_gene):
                    crossover_popu.append(crossovered_gene)
                    k += 1

            popu = parents_popu + mutate_popu + crossover_popu

        return self.best_config, final_population


    def crossover(self, genes):
        crossovered_gene = []
        for i in range(self.gene_len):
            if i not in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"] and i not in self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"]:
                if np.random.uniform() < 0.5:
                    crossovered_gene.append(genes[0][i])
                else:
                    crossovered_gene.append(genes[1][i])
            else:
                assert(isinstance(genes[0][i], list))
                assert(isinstance(genes[1][i], list))
                cur_parent, cur_child = None, None
                if np.random.uniform() < 0.5:
                    cur_parent = 0
                    cur_child = genes[0][i]
                else:
                    cur_parent = 1
                    cur_child = genes[1][i]
                num_experts_in_this_layer = crossovered_gene[self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"][i]] if i in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"] else crossovered_gene[self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"][i]]
                possible_ffn_dims = self.converter.gene2config_bothways["encoder_ffn_embed_dim_choice"] if i in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"] else self.converter.gene2config_bothways["decoder_ffn_embed_dim_choice"]
                if len(cur_child) != num_experts_in_this_layer:
                    if len(cur_child) > num_experts_in_this_layer:
                        cur_child = cur_child[0:num_experts_in_this_layer]
                    else:
                        cur_child += [random.choice(possible_ffn_dims) for j in range(num_experts_in_this_layer-len(cur_child))]
                assert(len(cur_child)==num_experts_in_this_layer)
                crossovered_gene.append(cur_child)

        self.gene_checker(crossovered_gene)
        return crossovered_gene


    def mutate(self, gene):
        mutated_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < self.mutation_prob:
                if i not in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"] and i not in self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"]:
                    mutated_gene.append(random.choices(self.gene_choice[i])[0])
                else:
                    assert(isinstance(gene[i], list))
                    num_experts_in_this_layer = mutated_gene[self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"][i]] if i in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"] else mutated_gene[self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"][i]]
                    possible_ffn_dims = self.converter.gene2config_bothways["encoder_ffn_embed_dim_choice"] if i in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"] else self.converter.gene2config_bothways["decoder_ffn_embed_dim_choice"]
                    # assert(num_experts_in_this_layer == len(mutated_gene[i]))  # gene might've been mutated as well
                    cur_gene = []
                    for j in range(num_experts_in_this_layer):
                        cur_gene.append(random.choice(possible_ffn_dims))
                    mutated_gene.append(cur_gene)
            else:
                mutated_gene.append(gene[i])

        for gene_id in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"]:
            num_experts_in_this_layer = mutated_gene[self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"][gene_id]]
            if len(mutated_gene[gene_id]) != num_experts_in_this_layer:
                # need to mutate this as well
                possible_ffn_dims = self.converter.gene2config_bothways["encoder_ffn_embed_dim_choice"]
                cur_child = mutated_gene[gene_id]
                if len(cur_child) != num_experts_in_this_layer:
                    if len(cur_child) > num_experts_in_this_layer:
                        cur_child = cur_child[0:num_experts_in_this_layer]
                    else:
                        cur_child += [random.choice(possible_ffn_dims) for j in range(num_experts_in_this_layer-len(cur_child))]
                mutated_gene[gene_id] = cur_child
                
        for gene_id in self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"]:
            num_experts_in_this_layer =  mutated_gene[self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"][gene_id]]
            if len(mutated_gene[gene_id]) != num_experts_in_this_layer:
                # need to mutate this as well
                possible_ffn_dims = self.converter.gene2config_bothways["decoder_ffn_embed_dim_choice"]
                cur_child = mutated_gene[gene_id]
                if len(cur_child) != num_experts_in_this_layer:
                    if len(cur_child) > num_experts_in_this_layer:
                        cur_child = cur_child[0:num_experts_in_this_layer]
                    else:
                        cur_child += [random.choice(possible_ffn_dims) for j in range(num_experts_in_this_layer-len(cur_child))]
                mutated_gene[gene_id] = cur_child

        self.gene_checker(mutated_gene)
        return mutated_gene


    def get_scores(self, genes):
        configs = []
        for gene in genes:
            configs.append(self.converter.gene2config(gene))

        scores = validate_all(self.args, self.trainer, self.task, self.epoch_iter, configs, self.generator)

        return scores

    def satisfy_constraints(self, gene):
        satisfy = True

        config = self.converter.gene2config(gene)

        if self.latency_constraint != -1: 
            cur_latency = None
            if self.latency_compute == "predictor":
                cur_latency = self.latency_predictor.predict_lat(config)
            elif self.latency_compute == "gold":
                # cur_latency = utils.measure_latency_during_search(self.args, self.trainer.model, self.dummy_src_tokens, self.dummy_prev, config, self.extra_args)
                cur_latency = self.get_real_latency(config)
            if cur_latency > self.latency_constraint:
                satisfy = False
        elif self.flops_constraint_giga != -1:
            macs = self.get_macs(config)
            # self.trainer.model.set_sample_config(config)
            # self.trainer.model.profile(mode=True)
            # macs = torchprofile.profile_macs(self.trainer.model, args=(self.dummy_src_tokens_tensor, self.dummy_src_len_tensor, self.dummy_prev_tokens_tensor))
            # self.trainer.model.profile(mode=False)
            # last_layer_macs = config['decoder']['decoder_embed_dim'] * self.dummy_sentence_length * len(self.task.tgt_dict)
            # total_flops_without_last_layer = (macs - last_layer_macs) * 2
            if macs > self.flops_constraint_giga:
                satisfy = False
        if self.args.ind_bias_encoder_layers_greater_than_equal_to_decoder_layers:
            if config["encoder"]["encoder_layer_num"] < config["decoder"]["decoder_layer_num"]:
                satisfy = False

        return satisfy

    def get_macs(self, config):
        self.trainer.model.set_sample_config(config)
        self.trainer.model.profile(mode=True)
        macs = torchprofile.profile_macs(self.trainer.model, args=(self.dummy_src_tokens_tensor, self.dummy_src_len_tensor, self.dummy_prev_tokens_tensor))
        self.trainer.model.profile(mode=False)
        return macs*2

    def get_real_latency(self, config):
        return utils.measure_latency_during_search(self.args, self.trainer.model, self.dummy_src_tokens, self.dummy_prev, config, self.extra_args)

    def random_sample(self, sample_num):
        popu = []
        i = 0
        while i < sample_num:
            samp_gene = []
            for k in range(self.gene_len):
                if not isinstance(self.gene_choice[k][0], list):
                    samp_gene.append(random.choices(self.gene_choice[k])[0])
                else:
                    assert k in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"] or k in self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"]
                    # encoder_each_expert_ffn_dim, decoder_each_expert_ffn_dim
                    num_experts_in_this_layer = samp_gene[self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"][k]] if k in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"] else samp_gene[self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"][k]]
                    samp_gene.append(random.choices(self.gene_choice[k][0], k=num_experts_in_this_layer))
            if self.satisfy_constraints(samp_gene):
                self.gene_checker(samp_gene)
                popu.append(samp_gene)
                i += 1

        return popu

    def gene_checker(self, gene):
        for gene_id in self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"]:
            num_experts_expected =  gene[self.converter.gene2config_bothways["encoder_each_expert_ffn_dim"][gene_id]]
            assert(len(gene[gene_id]) == num_experts_expected)
        for gene_id in self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"]:
            num_experts_expected =  gene[self.converter.gene2config_bothways["decoder_each_expert_ffn_dim"][gene_id]]
            assert(len(gene[gene_id]) == num_experts_expected)


def test():
    config = {
        'encoder': {
            'encoder_embed_dim': 512,
            'encoder_layer_num': 4,
            'encoder_ffn_embed_dim': [1024, 1025, 1026, 1027],
            'encoder_self_attention_heads': [4, 5, 6, 7],
        },
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_layer_num': 5,
            'decoder_ffn_embed_dim': [2048, 2049, 2050, 2051, 2052],
            'decoder_self_attention_heads': [4, 6, 7, 8, 9],
            'decoder_ende_attention_heads': [3, 4, 5, 6, 7],
            'decoder_arbitrary_ende_attn': [1, 2, 3, 4, 5, 6, 7]
        }
    }

    args = Namespace(encoder_layers=6,
                     decoder_layers=7,
                     encoder_embed_choice=[768, 512],
                     decoder_embed_choice=[768, 512],
                     encoder_ffn_embed_dim_choice=[3072, 2048],
                     decoder_ffn_embed_dim_choice=[3072, 2048],
                     encoder_layer_num_choice=[6, 5],
                     decoder_layer_num_choice=[6, 5, 4, 3],
                     encoder_self_attention_heads_choice=[8, 4],
                     decoder_self_attention_heads_choice=[8, 4],
                     decoder_ende_attention_heads_choice=[8],
                     decoder_arbitrary_ende_attn_choice=[1, 2]
                     )



    converter = Converter(args)
    gene_get = converter.config2gene(config)

    print(gene_get)
    print(len(gene_get))

    config_get = converter.gene2config(gene_get)

    print(config_get)

    print(len(converter.get_gene_choice()))
    print(converter.get_gene_choice())


def validate_all(args, trainer, task, epoch_itr, configs, generator):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
        # Initialize data iterator
    def get_itr():
        itr = task.get_batch_iterator(
            dataset=task.dataset('valid'),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format('valid'),
        )
        return progress


    for config in configs:
        trainer.set_sample_config(config)
        if args.validation_metric == "nonemb_params":
            nonemb_params = 0
            for name, param in trainer.model.named_parameters():
                if 'embed' not in name:
                    nonemb_params += param.numel()
            valid_losses.append(nonemb_params)
            continue

        progress = get_itr()

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        valid_cnt = 0
        if args.validation_metric == "bleu":
            bleu_syss, bleu_refs = [], []
        for sample in progress:
            valid_cnt += 1
            if valid_cnt > args.valid_cnt_max:
                break
            if args.validation_metric == "bleu":
                log_output, bleu_input = trainer.valid_step(sample, generator)
                bleu_syss += bleu_input[1]
                bleu_refs += bleu_input[0]
            else:
                log_output = trainer.valid_step(sample)

        if args.validation_metric == "bleu":
            # compute valid bleu score
            dict_obj = dictionary.Dictionary()
            scorer = bleu.Scorer(dict_obj.pad(), dict_obj.eos(), dict_obj.unk())
            for (ref_tok, sys_tok) in zip(bleu_refs, bleu_syss):
                sys_tok = dict_obj.encode_line(sys_tok)
                ref_tok = dict_obj.encode_line(ref_tok)
                scorer.add(ref_tok, sys_tok)
            bleu_score = scorer.score(4) # consider ngrams up to this order 

            valid_losses.append(bleu_score)
        elif args.validation_metric == "loss":
            valid_losses.append(trainer.get_meter('valid_loss').avg)

    return valid_losses


# if __name__=='__main__':
#    test()
