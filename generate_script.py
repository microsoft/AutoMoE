# AutoMoE: Neural Architecture Search for Efficient Sparsely Activated Transformers
# Ganesh Jawahar, Subhabrata Mukherjee, Xiaodong Liu, Young Jin Kim, Muhammad Abdul-Mageed, Laks V. S. Lakshmanan, Ahmed Hassan Awadallah, Sebastien Bubeck, Jianfeng Gao
# Paper: https://arxiv.org/abs/2210.07535

'''
script to generate commands to run AutoMoE pipeline
'''

import sys, json, glob, os
import argparse

def run_automoe(args):
  on_device_hardware = args.hardware_spec
  trial_run = (args.trial_run == 1)
  task_name = args.task
  cur_work_dir = args.output_dir
  experts_possible_vals = " ".join(str(i+1) for i in range(args.max_experts)).strip()
  exp_new_space = "--encoder-n-experts %s --decoder-n-experts %s"%(experts_possible_vals, experts_possible_vals)
  extra_lat_feats = "640 6 2048 6 640 6 2048 6 6 2 %d %d"%(args.max_experts, args.max_experts)
  if args.frac_experts == 1:
    exp_new_space += " --encoder-each-expert-ffn-dim 0 1 --decoder-each-expert-ffn-dim 0 1"
    extra_lat_feats += " 3072 3072"

  # (1) train SuperNet
  cur_supernet_dir = cur_work_dir + "/supernet"
  os.makedirs(cur_supernet_dir, exist_ok=True)
  train_super_transformer_cmd = "python -B train.py --configs=configs/%s/supertransformer/space0.yml --save-dir %s %s"%(task_name, cur_supernet_dir, exp_new_space)
  train_super_transformer_cmd += " --update-freq=%d --no-epoch-checkpoints"%(int(128/args.num_gpus))
  supernet_ckpt_f = cur_supernet_dir + "/checkpoint_best.pt"
  if trial_run:
    train_super_transformer_cmd += " --max-update 5 --save-interval-updates 5"
  train_super_transformer_cmd += " > %s/run.out"%(cur_supernet_dir)
  print(train_super_transformer_cmd.strip())

  # (2) generate latency dataset (optional stage)
  cur_genlatdata_dir = cur_work_dir + "/genlatdata"
  os.makedirs(cur_genlatdata_dir, exist_ok=True)
  genlatdata_datatset_f = "%s/%s_%s.csv"%(cur_genlatdata_dir, task_name, on_device_hardware)
  gen_latency_data_cmd = "CUDA_VISIBLE_DEVICES=0 python latency_dataset.py --configs=configs/%s/latency_dataset/%s.yml %s --lat-dataset-path %s"%(task_name, on_device_hardware, exp_new_space, genlatdata_datatset_f)
  if trial_run:
    gen_latency_data_cmd += " --lat-dataset-size 10"
  gen_latency_data_cmd += " > %s/run.out"%(cur_genlatdata_dir)
  print(gen_latency_data_cmd.strip())

  # (3) train latency predictor (optional stage)
  cur_latpred_dir = cur_work_dir + "/latpred"
  os.makedirs(cur_latpred_dir, exist_ok=True)
  latpred_ckpt_f = "%s/%s_%s.pt"%(cur_latpred_dir, task_name, on_device_hardware)
  feature_norm = extra_lat_feats
  feature_dim = len(feature_norm.split())
  train_latency_predictor_cmd = "python latency_predictor.py --configs=configs/%s/latency_predictor/%s.yml --feature-norm %s --feature-dim %d --lat-dataset-path %s --ckpt-path %s"%(task_name, on_device_hardware, feature_norm, feature_dim, genlatdata_datatset_f, latpred_ckpt_f)
  if trial_run:
    train_latency_predictor_cmd += " --bsz 2 --train-steps 10"
  train_latency_predictor_cmd += " > %s/run.out"%(cur_latpred_dir)
  print(train_latency_predictor_cmd)

  # (4) run evolutionary search
  cur_evosearch_dir = cur_work_dir + "/evosearch"
  os.makedirs(cur_evosearch_dir, exist_ok=True)
  efficient_config_f = "%s/%s_%s.yml"%(cur_evosearch_dir, task_name, on_device_hardware)
  feature_norm = extra_lat_feats
  feature_dim = len(feature_norm.split())
  evosearch_cmd = "CUDA_VISIBLE_DEVICES=0 python evo_search.py --configs=configs/%s/supertransformer/space0.yml --evo-configs=configs/%s/evo_search/%s_titanxp.yml --restore-file %s --ckpt-path %s --feature-norm %s --write-config-path %s %s --validation-metric loss --latency-constraint %d --latency-compute %s --latiter %d --evo-iter %d"%(task_name, task_name, task_name.replace(".", "").replace("-", ""), supernet_ckpt_f, latpred_ckpt_f, feature_norm, efficient_config_f, exp_new_space, args.latency_constraint, args.latency_compute, args.latiter, args.evo_iter)
  if trial_run:
    evosearch_cmd += " --evo-iter 1 --parent-size 2 --mutation-size 2 --crossover-size 2 --population-size 6"
  evosearch_cmd += " > %s/run.out"%(cur_evosearch_dir)
  print(evosearch_cmd)

  # (5) hardware latency & flops  
  latency_flops_inherited_test_bleu_dir = cur_work_dir + "/latency_flops_inherited_test"
  os.makedirs(latency_flops_inherited_test_bleu_dir, exist_ok=True)
  latency_flops_inherited_test_result_f = "%s/%s_%s.txt"%(latency_flops_inherited_test_bleu_dir, task_name, on_device_hardware)
  latency_flops_inherited_test_cmd = "CUDA_VISIBLE_DEVICES=0 python train.py --configs=%s --sub-configs=configs/%s/subtransformer/common.yml --latgpu --save-dir %s"%(efficient_config_f, task_name, latency_flops_inherited_test_bleu_dir)
  latency_flops_inherited_test_cmd += " >> %s"%latency_flops_inherited_test_result_f
  latency_flops_inherited_test_cmd += "\nCUDA_VISIBLE_DEVICES=0 python train.py --configs=%s --sub-configs=configs/%s/subtransformer/common.yml --profile-flops --save-dir %s"%(efficient_config_f, task_name, latency_flops_inherited_test_bleu_dir)
  latency_flops_inherited_test_cmd += " >> %s"%latency_flops_inherited_test_result_f
  print(latency_flops_inherited_test_cmd.strip())

  # (6) train searched transformer from scratch
  cur_effnet_dir = cur_work_dir + "/effnet"
  os.makedirs(cur_effnet_dir, exist_ok=True)
  train_effnet_transformer_cmd = "python -B train.py --configs=%s --save-dir %s --sub-configs=configs/%s/subtransformer/common.yml --no-epoch-checkpoints"%(efficient_config_f, cur_effnet_dir, task_name)  
  train_effnet_transformer_cmd += " --update-freq=%d"%(int(128/args.num_gpus))
  effnet_ckpt_f = cur_effnet_dir + "/checkpoint_best.pt"
  if trial_run:
    train_effnet_transformer_cmd += " --max-update 5 --save-interval-updates 5"
  train_effnet_transformer_cmd += " > %s/run.out"%(cur_effnet_dir)
  print(train_effnet_transformer_cmd.strip())

  # (7) inference of best subnet
  cur_perf_scratch_test_bleu_dir = cur_work_dir + "/perf_scratch_test"
  os.makedirs(cur_perf_scratch_test_bleu_dir, exist_ok=True)
  perf_scratch_test_result_f = "%s/%s_%s.txt"%(cur_perf_scratch_test_bleu_dir, task_name, on_device_hardware)
  test_sh_f = "configs/%s/test.sh"%(task_name)
  perf_scratch_test_cmd = "bash %s %s %s %s normal >> %s"%(test_sh_f, effnet_ckpt_f, efficient_config_f, cur_perf_scratch_test_bleu_dir, perf_scratch_test_result_f)
  print(perf_scratch_test_cmd.replace("normal >> ", "normal 0 valid >> ").strip())
  print(perf_scratch_test_cmd.strip())


parser = argparse.ArgumentParser(description="Script to run AutoMoE pipeline")
parser.add_argument("--task", type=str, default="wmt14.en-de", help="MT dataset? wmt14.en-de or wmt14.en-fr or wmt19.en-de")
parser.add_argument("--output_dir", type=str, default="/tmp", help="Output directory to write files generated during experiment?")
parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use?")
parser.add_argument("--trial_run", type=int, default=0, help="Trial run?")
parser.add_argument("--hardware_spec", type=str, default="gpu_titanxp", help="Hardware specification?")
parser.add_argument("--max_experts", type=int, default=6, help="Maximum experts (for Supernet) to use?")
parser.add_argument("--frac_experts", type=int, default=1, help="Fractional (varying FFN. intermediate size) experts to use?")
parser.add_argument("--supernet_ckpt", type=str, default=None, help="Skip supernet training by specifiying checkpoint from https://1drv.ms/u/s!AlflMXNPVy-wgb9w-aq0XZypZjqX3w?e=VmaK4n")
parser.add_argument('--latency_compute', type=str, default="gold", help='gold or predictor latency')
parser.add_argument('--latiter', type=int, default=100, help='number of latency iterations for using real latency? only used when latency-compute is gold')
parser.add_argument('--latency_constraint', type=float, default=200, help='latency constraint in terms of milliseconds?')
parser.add_argument('--evo_iter', type=int, default=10, help='# iterations for evolutionary search')
args = parser.parse_args()

run_automoe(args)





