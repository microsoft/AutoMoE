# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import pdb
import json
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.trainer import Trainer
from fairseq.evolution import Evolution


def main(args):
    utils.import_user_module(args)
    utils.handle_save_path(args)
    print(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    generator = None
    if args.validation_metric == "bleu":
        generator = task.build_generator(args)
    print(model)

    # Build trainer
    trainer = Trainer(args, task, model, criterion)

    # Load the latest checkpoint if one is available and restore the corresponding train iterator
    args.train_subset = 'valid' # no need to train, so just set a small subset to save loading time
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # run evolutionary search to find the model with lowest loss and satisfies the latency requirement
    evolver = Evolution(args, trainer, task, epoch_itr, generator=generator)
    best_config, final_popu = evolver.run_evo_search()
    print('search done')
    print(best_config)

    with open(args.write_config_path.replace(".yml", ".json"), 'w') as outfile:
        json.dump(final_popu, outfile)

    with open(args.write_config_path, 'w') as fid:
        encoder_layer_num = best_config['encoder']['encoder_layer_num']
        decoder_layer_num = best_config['decoder']['decoder_layer_num']

        fid.write(f"encoder-embed-dim-subtransformer: {best_config['encoder']['encoder_embed_dim']}\n")
        fid.write(f"decoder-embed-dim-subtransformer: {best_config['decoder']['decoder_embed_dim']}\n\n")

        fid.write(f"encoder-ffn-embed-dim-all-subtransformer: {best_config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num]}\n")
        fid.write(f"decoder-ffn-embed-dim-all-subtransformer: {best_config['decoder']['decoder_ffn_embed_dim'][:decoder_layer_num]}\n\n")

        fid.write(f"encoder-layer-num-subtransformer: {best_config['encoder']['encoder_layer_num']}\n")
        fid.write(f"decoder-layer-num-subtransformer: {best_config['decoder']['decoder_layer_num']}\n\n")

        fid.write(f"encoder-self-attention-heads-all-subtransformer: {best_config['encoder']['encoder_self_attention_heads'][:encoder_layer_num]}\n")
        fid.write(f"decoder-self-attention-heads-all-subtransformer: {best_config['decoder']['decoder_self_attention_heads'][:decoder_layer_num]}\n")
        fid.write(f"decoder-ende-attention-heads-all-subtransformer: {best_config['decoder']['decoder_ende_attention_heads'][:decoder_layer_num]}\n\n")

        fid.write(f"decoder-arbitrary-ende-attn-all-subtransformer: {best_config['decoder']['decoder_arbitrary_ende_attn'][:decoder_layer_num]}\n\n")

        if 'encoder_n_experts' in best_config['encoder']:
            fid.write(f"encoder-n-experts: {best_config['encoder']['encoder_n_experts'][:encoder_layer_num]}\n")

        if 'decoder_n_experts' in best_config['decoder']:
            fid.write(f"decoder-n-experts: {best_config['decoder']['decoder_n_experts'][:decoder_layer_num]}\n\n")

        #if len(args.encoder_num_experts_to_route) > 1:
        if 'encoder_num_experts_to_route' in best_config['encoder']:
            fid.write(f"encoder-num-experts-to-route: {best_config['encoder']['encoder_num_experts_to_route'][:encoder_layer_num]}\n")

        #if len(args.decoder_num_experts_to_route) > 1:
        if 'decoder_num_experts_to_route' in best_config['decoder']:
            fid.write(f"decoder-num-experts-to-route: {best_config['decoder']['decoder_num_experts_to_route'][:decoder_layer_num]}\n\n")

        if 'encoder_drop_mha_sublayer' in best_config['encoder']:
            fid.write(f"encoder-drop-mha-sublayer: {best_config['encoder']['encoder_drop_mha_sublayer'][:encoder_layer_num]}\n\n")
        if 'encoder_drop_ffn_sublayer' in best_config['encoder']:
            fid.write(f"encoder-drop-ffn-sublayer: {best_config['encoder']['encoder_drop_ffn_sublayer'][:encoder_layer_num]}\n\n")

        if 'decoder_drop_mha_sublayer' in best_config['decoder']:
            fid.write(f"decoder-drop-mha-sublayer: {best_config['decoder']['decoder_drop_mha_sublayer'][:decoder_layer_num]}\n\n")
        if 'decoder_drop_ffn_sublayer' in best_config['decoder']:
            fid.write(f"decoder-drop-ffn-sublayer: {best_config['decoder']['decoder_drop_ffn_sublayer'][:decoder_layer_num]}\n\n")

        if 'encoder_std_vs_dummy_experts' in best_config['encoder']:
            fid.write(f"encoder-std-vs-dummy-experts: {best_config['encoder']['encoder_std_vs_dummy_experts'][:encoder_layer_num]}\n")
        if 'encoder_each_expert_ffn_dim' in best_config['encoder']:
            listoflist = []
            for item in best_config['encoder']['encoder_each_expert_ffn_dim'][:encoder_layer_num]:
                listoflist.append("_".join([str(it) for it in item]))
            fid.write(f"encoder-each-expert-ffn-dim-listoflist: {listoflist}\n\n")

        if 'decoder_std_vs_dummy_experts' in best_config['decoder']:
            fid.write(f"decoder-std-vs-dummy-experts: {best_config['decoder']['decoder_std_vs_dummy_experts'][:decoder_layer_num]}\n")
        if 'decoder_each_expert_ffn_dim' in best_config['decoder']:
            listoflist = []
            for item in best_config['decoder']['decoder_each_expert_ffn_dim'][:decoder_layer_num]:
                listoflist.append("_".join([str(it) for it in item]))
            fid.write(f"decoder-each-expert-ffn-dim-listoflist: {listoflist}\n\n")
            

def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--evo-configs', required=True, is_config_file=True)
    parser.add_argument('--evo-iter', type=int, default=30)
    parser.add_argument('--population-size', type=int, default=125)
    parser.add_argument('--parent-size', type=int, default=25)
    parser.add_argument('--mutation-size', type=int, default=50)
    parser.add_argument('--crossover-size', type=int, default=50)
    parser.add_argument('--mutation-prob', type=float, default=0.3)

    parser.add_argument('--feature-norm', type=float, nargs='+', help='normalizing factor for each feature')
    parser.add_argument('--lat-norm', type=float, help='normalizing factor for latency')
    parser.add_argument('--ckpt-path', type=str, help='path to load latency predictor weights')

    parser.add_argument('--latency-constraint', type=float, default=-1, help='latency constraint')
    parser.add_argument('--valid-cnt-max', type=int, default=1e9, help='max number of sentences to use in validation set')

    parser.add_argument('--write-config-path', type=str, help='path to write out the searched best SubTransformer')

    parser.add_argument('--validation-metric', type=str, default="loss", help='loss or bleu or active_nonemb_params or nonemb_params')
    parser.add_argument('--ind-bias-encoder-layers-greater-than-equal-to-decoder-layers', action='store_true', default=False)

    parser.add_argument('--flops-constraint-giga', type=float, default=-1, help='flops constraint in giga flops. -1 means no FLOPs constraint')
    parser.add_argument('--latency-compute', type=str, default="predictor", help='predictor or gold')
    parser.add_argument('--latiter', type=int, default=50, help='number of latency iterations for using real latency. only used when latency-compute is gold')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    if args.pdb:
        pdb.set_trace()

    # one GPU is fast enough to do the search
    args.distributed_world_size = 1
                  
    # if search on CPU, use fp32 as default
    if args.cpu:
        args.fp16 = False

    main(args)


if __name__ == '__main__':
    cli_main()
