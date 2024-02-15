## Evaluation

We evaluate NutePrune on the LLaMA family. We follow the original LLaMA paper to measure the effectiveness of pruned LLMs across three key application domains:

- [Zero-shot Commonsense Reasoning/Reading Comprehension](./lm-evaluation-harness/). We evaluate the 0-shot results for 7 commonsense reasoning benchmarks: PIQA, HellaSwag, WinoGrande, ARC easy and challenge, OpenBookQA (OBQA), and BoolQ. [[Example link](../scripts/eval_commonsense_lora.sh)]

- [Popular Aggregated Benchmarks](./instruct-eval/). Besides, we evaluate the in-context learning ability under a few-shot setting. We report the results on MMLU (5 shot), which consists of 57 tasks covering STEM, humanities, social science, etc.
    > Note: instruct-eval repository requires to create new virtual environment, and within the new environment, ``transformers==4.25.1`` will to be updated to ``transformers==4.29.0``.

For commonsense reasoning and reading comprehension, we use [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master)
to carry out the evaluations. For MMLU, we use [InstructEval](https://github.com/declare-lab/instruct-eval/tree/main). We saved the repo versions used in our work within this folder in case any changes in the results due to future updates in these repos.

- lm-evaluation-harness setup:

``` bash
cd evaluation/lm-evaluation-harness
pip install -e .
```

- InstructEval setup:

``` bash
conda create -n instruct-eval python=3.8 -y
conda activate instruct-eval
pip install -r requirements.txt
mkdir -p data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
```
