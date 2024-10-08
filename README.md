<h1 align="center"> 
<p> NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models </p>
</h1>

## Install dependencies
```bash
pip install -r requirements.txt
```

## Pruning

**Step 1**: Run ``bash ./scripts/cotrain.sh`` to try NutePrune pruning.

(Modify ``pruning_type`` ``target_sparsity`` ``model_name_or_path`` ``lagrangian_warmup_epochs``to run different tasks.)

## Post Fine-tuning

**Step 1**: After pruning, prepare the pruned output folder as ``$baseline_pruned_model``, it should consist of LoRA weights file ``lora_weights.pt`` and pruning mask file ``zs.pt``.

**Step 2**: Prepare dataset for training: Download official [Alpaca dataset](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json) and put the dataset into ``./data``.

**Step 3**: Run ``bash ./scripts/finetune_alpaca.sh`` to try post fine-tuning on alpaca. (Modify ``baseline_pruned_model`` ``model_name_or_path`` to run different tasks)

## Evaluation

**1. PPL**

Run ``bash ./scripts/eval_ppl.sh``

**2. Zero-shot Commonsense Reasoning**

First, install lm-evaluation-harness:
```
cd lm-evaluation-harness
conda create -n lm-eval python==3.9
conda activate lm-eval
pip install -e .
```
Install other packages:
```
pip install deepspeed
pip install sentencepiece
```
Then Run ``bash ./scripts/eval_commonsense.sh``

**3. Benchmarks**

To evaluate MMLU, BBH, GSM8K and other LLM benchmarks, we recommond using the latest lm-evaluation-harness:
```
cd ~
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
conda create -n leh python==3.9
conda activate leh
pip install -e .
pip install sentencepiece
pip install protobuf
```

Then merge lora weights and masks by runing ``bash ./scripts/merge_weights.sh``

Then Run ``bash ./scripts/eval_benchmark.sh``
