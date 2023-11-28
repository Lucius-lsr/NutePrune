# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import logging
from itertools import chain
import transformers
from transformers import DataCollatorWithPadding, default_data_collator
from transformers.testing_utils import CaptureLogger
from models.tokenization_llama import LlamaTokenizer
from datasets import load_from_disk
from datasets import load_dataset

logger = logging.getLogger(__name__)

pruning_prompt = (
        "###System Instruction:\nAttention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Here's what you need to know:\n"
    "Improved Efficiency: The pruning algorithm streamlines your operations, enhancing overall performance.\n"
    "Resource Utilization: Reducing unnecessary parameters optimizes computational resources and memory requirements.\n"
    "Maintaining Quality: The algorithm selectively prunes to preserve language generation capabilities while minimizing impact on performance.\n"
    "Iterative Refinement: Pruning occurs gradually, with fine-tuning after each step to ensure a smooth transition.\n"
    "Collaboration: Embrace changes, continue generating high-quality language outputs to contribute to the optimization process.\n"
    "Continuous Learning: Pruning helps you evolve, focusing on essential aspects of language generation.\n"
    "Together, we'll create a leaner and more efficient version of you. Let's unlock your full potential through pruning.\n\n"
        "Below is an pre-training text, "
        "please take full consideration of the System Instruction mentioned above, and then predict the next token as you undergo pre-training.\n\n"
        "###Input:\n"
    ),



def get_c4_prompt_data_module(tokenizer, model_args, data_args, training_args):
    
    #load samples from c4 dataset
    # raw_datasets = load_from_disk(data_args.train_file)
    raw_datasets = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    print('data count: ', raw_datasets.num_rows)
    raw_datasets = raw_datasets.select(range(20000))
    print('data count: ', raw_datasets.num_rows)
    
    column_names=['timestamp','url','text']
    text_column_name = "text"

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.max_seq_length is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        tokenized_prompt = tokenizer(pruning_prompt)
        len_prompt = len(tokenized_prompt["input_ids"][0])
        block_size = min(data_args.max_seq_length, tokenizer.model_max_length) - len_prompt
        assert block_size > 0, "total length of prompt is larger than max_seq_length"

        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        prompted_result = {k:[] for k in result.keys()}
        IGNORE_INDEX = -100
        for i in range(len(result["input_ids"])):
            prompted_result["input_ids"].append(tokenized_prompt["input_ids"][0] + result["input_ids"][i])
            prompted_result["labels"].append( ([IGNORE_INDEX] * len_prompt) + result["labels"][i])
            prompted_result["attention_mask"].append(tokenized_prompt["attention_mask"][0] + result["attention_mask"][i])
        return prompted_result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        
    ##############################################3
    train_dataset = None
    if training_args.do_train:
        train_dataset = lm_datasets

    # if training_args.do_eval:
    #     raise ValueError("our c4 dataset dosen't have evaluation set")
    
    
    #################################################    
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    return dict(
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics= None,
        preprocess_logits_for_metrics=None,
    )


def evaluate_c4(model, model_args, data_args, training_args):
    from trainer.compresso_trainer import CompressoTrainer

    if "llama" in model_args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            padding_side="left",
            truncation_side="left",
        )
    else:
        raise ValueError("Tokenizer is not set.")

    data_module = get_c4_prompt_data_module(tokenizer, model_args, data_args, training_args)

    trainer = CompressoTrainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        **data_module
    )

    metrics = trainer.evaluate()

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    return metrics

