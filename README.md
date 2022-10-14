# AutoMoE: Neural Architecture Search for Efficient Sparsely Activated Transformers

This repository contains code, data and pretrained models used in AutoMoE work. This repository builds on [Hardware Aware Transformer (HAT)'s repository](https://github.com/mit-han-lab/hardware-aware-transformers).

## Quick Setup

### (1) Install
Run the following commands to install AutoMoE:
```bash
git clone https://github.com/UBC-NLP/AutoMoE.git
cd AutoMoE
pip install --editable .
```

### (2) Prepare Data
Run the following commands to download preprocessed MT data:
```bash
bash configs/[task_name]/get_preprocessed.sh
```
where `[task_name]` can be `wmt14.en-de` or `wmt14.en-fr` or `wmt19.en-de`.

### (3) Run full AutoMoE pipeline
Run the following commands to start AutoMoE pipeline:
```bash
python generate_script.py --task wmt14.en-de --output_dir /tmp --num_gpus 4 --trial_run 0 --hardware_spec gpu_titanxp --max_experts 6 --frac_experts 1 > automoe.sh
bash automoe.sh
```
where,
* `task` - MT dataset to use: `wmt14.en-de` or `wmt14.en-fr` or `wmt19.en-de` (default: `wmt14.en-de`)
* `output_dir` - Output directory to write files generated during experiment (default: `/tmp`)
* `num_gpus` - Number of GPUs to use (default: `4`)
* `trial_run` - Run trial run (useful to quickly check if everything runs fine without errors.): 0 (final run), 1 (dry/dummy/trial run) (default: `0`)
* `hardware_spec` - Hardware specification: `gpu_titanxp` (For GPU) (default: `gpu_titanxp`)
* `max_experts` - Maximum experts (for Supernet) to use (default: `6`)
* `frac_experts` - Fractional (varying FFN. intermediate size) experts: 0 (Standard experts) or 1 (Fractional) (default: `1`)
* `supernet_ckpt` - Skip supernet training by specifiying checkpoint from [pretrained models](https://1drv.ms/u/s!AlflMXNPVy-wgb9w-aq0XZypZjqX3w?e=VmaK4n) (default: `None`)
* `latency_compute` - Use (partially) gold or predictor latency (default: `gold`)
* `latiter` - Number of latency measurements for using (partially) gold latency (default: `100`)
* `latency_constraint` - Latency constraint in terms of milliseconds (default: `200`)
* `evo_iter` - Number of iterations for evolutionary search (default: `10`)

## Contact
If you have questions, contact Ganesh (`ganeshjwhr@gmail.com`), Subho (`Subhabrata.Mukherjee@microsoft.com`) and/or create GitHub issue.

## License
See LICENSE.txt for license information.

## Acknowledgements
* [Hardware Aware Transformer](https://github.com/mit-han-lab/hardware-aware-transformers) from `mit-han-lab`
* [fairseq](https://github.com/facebookresearch/fairseq) from `facebookresearch`

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
