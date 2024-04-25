# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os
import sys
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trainer.efficient_trainer import EfficientTrainer
from models.l0_module import L0Module
from args import AdditionalArguments, DataTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from utils.nuteprune_utils import load_zs, load_l0_module
from models.model_args import ModelArguments
import torch
import math

logger = logging.getLogger(__name__)

ALPACA_TASK = ["alpaca", "alpaca-gpt4", "alpaca-gpt4-zh", "unnatural_instruction_gpt4", "math", "open_orca", "alpaca-cleaned"]

def set_lora_args(config, modeling_args):
    config.use_lora = modeling_args.use_lora
    config.lora_rank = modeling_args.lora_rank
    config.lora_train_bias = modeling_args.lora_train_bias
    config.lora_alpha = modeling_args.lora_alpha
    config.lora_param = modeling_args.lora_param
    config.lora_layers = modeling_args.lora_layers
    return config


def main():
    # # Used for profiling, usage:
    # #   [install] sudo env "PATH=$PATH" pip install viztracer
    # #   [profile] sudo env "PATH=$PATH" viztracer --attach_installed [PID]
    # from viztracer import VizTracer
    # tracer = VizTracer()
    # tracer.install()

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    training_args.report_to = []

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args} \n {additional_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # model initialize
    CONFIG, TOKENIZER, CAUSALLM = None, None, None
    if 'llama' in model_args.model_name_or_path.lower():
        from models.modeling_llama import LlamaForCausalLM
        from transformers.models.llama import LlamaConfig
        from transformers import AutoTokenizer
        CONFIG, TOKENIZER, CAUSALLM = LlamaConfig, AutoTokenizer, LlamaForCausalLM
    elif 'mistral' in model_args.model_name_or_path.lower():
        from transformers.models.mistral import MistralConfig
        from models.modelling_mistral import MistralForCausalLM
        from transformers import AutoTokenizer
        CONFIG, TOKENIZER, CAUSALLM = MistralConfig, AutoTokenizer, MistralForCausalLM
    else:
        raise NotImplementedError

    config = CONFIG.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    config.use_cache = False
    lora_ckpt = None
    config = set_lora_args(config, model_args)
    if additional_args.do_layer_distill:
        config.output_hidden_states = True
    if additional_args.pretrained_pruned_model is not None:
        lora_ckpt = os.path.join(additional_args.pretrained_pruned_model, 'lora_weights.pt')
        if os.path.exists(lora_ckpt):
            logger.info(f"Load lora ckpt from {lora_ckpt}")
        else:
            lora_ckpt = None
            logger.info(f"No lora ckpt found")
    tokenizer = TOKENIZER.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_max_length=512,
    )

    model = CAUSALLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    if lora_ckpt is not None:
        model.load_state_dict(torch.load(lora_ckpt), strict=False)
    else:  # init lora
        for n, p in model.named_parameters():
            if 'lora_A' in n:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    model.half()  # accelerate
    model.enable_input_require_grads()
    training_args.fp16 = True

    l0_module = None
    if additional_args.pruning_type is not None:
        if additional_args.pretrained_pruned_model is not None:
            l0_module = load_l0_module(os.path.join(additional_args.pretrained_pruned_model, 'l0_module.pt'))
            logger.info(f"Load pretrained l0_module!")
        else:
            l0_module = L0Module(config=config,
                                droprate_init=additional_args.droprate_init,
                                layer_gate_init_open=additional_args.layer_gate_init_open,
                                layer_gate_open_0=additional_args.layer_gate_open_0,
                                block_layer_start=additional_args.block_layer_start,
                                block_layer_end=additional_args.block_layer_end,
                                sparsity_scheduler=additional_args.sparsity_scheduler,
                                temperature=additional_args.temperature,
                                target_sparsity=additional_args.target_sparsity,
                                pruning_type=additional_args.pruning_type)

    zs = None
    if additional_args.pretrained_pruned_model is not None:
        zs = load_zs(os.path.join(additional_args.pretrained_pruned_model, 'zs.pt'))
        logger.info(f"Load pretrained zs!")
        for key in zs:
            zs[key] = zs[key].detach()

    # dataset initialize
    from tasks import get_data_module
    if data_args.dataset_name in ALPACA_TASK:
        data_module = get_data_module(data_args.dataset_name)(tokenizer, model_args, data_args, training_args, model)
    else:
        data_module = get_data_module(data_args.dataset_name)(tokenizer, model_args, data_args, training_args)

    # Initialize our Trainer
    trainer = EfficientTrainer(
        model=model,
        args=training_args,
        additional_args=additional_args,
        tokenizer=tokenizer,
        use_lora=model_args.use_lora,
        lora_train_bias=model_args.lora_train_bias,
        l0_module=l0_module,
        **data_module
    )

    if additional_args.pretrained_pruned_model is not None:
        if l0_module is None:  # continue train zs instead of load fixed zs
            trainer.zs = zs

    # Training
    if training_args.do_train:
        trainer.train(None)

if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()
