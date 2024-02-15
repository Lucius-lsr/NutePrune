import json
import signal
import time
from pathlib import Path
from typing import Optional, Tuple

# import openai
# import rwkv
# import tiktoken
import torch
import torch.nn as nn
import transformers
from fire import Fire
# from peft import PeftModel
from pydantic import BaseModel
# from rwkv.model import RWKV
# from rwkv.utils import PIPELINE
# from torchvision.datasets.utils import download_url
from transformers import AutoTokenizer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    # LlamaForCausalLM,
    # LlamaTokenizer,
    AutoModel,
    # LlamaConfig,
)

import quant


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    max_input_length: int = 512
    max_output_length: int = 512

    def run(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def count_text_length(self, text: str) -> int:
        raise NotImplementedError

    def check_valid_length(self, text: str) -> bool:
        return self.count_text_length(text) <= self.max_input_length


# class OpenAIModel(EvalModel):
#     model_path: str
#     engine: str = ""
#     use_azure: bool = False
#     tokenizer: Optional[tiktoken.Encoding]
#     api_endpoint: str = "https://research.openai.azure.com/"
#     api_version: str = "2023-03-15-preview"
#     timeout: int = 60
#     temperature: float = 0.0

#     def load(self):
#         if self.tokenizer is None:
#             self.tokenizer = tiktoken.get_encoding("cl100k_base")  # chatgpt/gpt-4

#         with open(self.model_path) as f:
#             info = json.load(f)
#             openai.api_key = info["key"]
#             self.engine = info["engine"]

#         if self.use_azure:
#             openai.api_type = "azure"
#             openai.api_base = self.api_endpoint
#             openai.api_version = self.api_version

#     def run(self, prompt: str, **kwargs) -> str:
#         self.load()
#         output = ""
#         error_message = "The response was filtered"

#         while not output:
#             try:
#                 key = "engine" if self.use_azure else "model"
#                 kwargs = {key: self.engine}
#                 response = openai.ChatCompletion.create(
#                     messages=[{"role": "user", "content": prompt}],
#                     timeout=self.timeout,
#                     request_timeout=self.timeout,
#                     temperature=0,  # this is the degree of randomness of the model's output
#                     **kwargs,
#                 )
#                 if response.choices[0].finish_reason == "content_filter":
#                     raise ValueError(error_message)
#                 output = response.choices[0].message.content
#             except Exception as e:
#                 print(e)
#                 if error_message in str(e):
#                     output = error_message

#             if not output:
#                 print("OpenAIModel request failed, retrying.")

#         return output

#     def count_text_length(self, text: str) -> int:
#         self.load()
#         return len(self.tokenizer.encode(text))

#     def get_choice(self, prompt: str, **kwargs) -> str:
#         self.load()

#         def handler(signum, frame):
#             raise Exception("Timeout")

#         signal.signal(signal.SIGALRM, handler)

#         for i in range(3):  # try 5 times
#             signal.alarm(2)  # 5 seconds
#             try:
#                 response = openai.ChatCompletion.create(
#                     engine=self.model_path,
#                     messages=[{"role": "user", "content": prompt}],
#                 )
#                 return response.choices[0].message.content
#             except Exception as e:
#                 if "content management policy" in str(e):
#                     break
#                 else:
#                     time.sleep(3)
#         return "Z"


class SeqToSeqModel(EvalModel):
    model_path: str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer|str]
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    do_sample: bool = False

    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            do_sample=self.do_sample,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        start_token = torch.tensor(
            [[self.tokenizer.pad_token_id]], dtype=torch.long
        ).to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                decoder_input_ids=start_token,
                **kwargs,
            ).logits[0, 0]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


class CausalModel(SeqToSeqModel):
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, **args
            )
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            do_sample=self.do_sample,
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


class LlamaModel(SeqToSeqModel):
    use_template: bool = False
    zs: Optional[dict] = None
    """
    Not officially supported by AutoModelForCausalLM, so we need the specific class
    Optionally, we can use the prompt template from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
    However, initial MMLU experiments indicate that the template is not useful for few-shot settings
    """

    def load(self):
        import os
        from models.modeling_llama import LlamaForCausalLM
        from utils.compresso_utils import load_zs
        from models.modeling_llama import LlamaConfig
        from models.tokenization_llama import LlamaTokenizer

        # if self.tokenizer is None:
        #     self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        # elif isinstance(self.tokenizer, str):
        #     self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer)
        if self.model is None:
            args = {}
            # if self.load_8bit:
            #     args.update(device_map="auto", load_in_8bit=True)
            # self.model = LlamaForCausalLM.from_pretrained(self.model_path, **args)
            # if self.lora_path:
            #     self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            # self.model.eval()
            # self.model.half()
            # if not self.load_8bit:
            #     self.model.to(self.device)
                # model initialize
            def set_lora_args(config, use_lora, lora_param):
                config.use_lora = True
                config.lora_rank = 8
                config.lora_train_bias = None
                config.lora_alpha = 8.0
                config.lora_param = lora_param
                config.lora_layers = config.num_hidden_layers
                return config
            config = LlamaConfig.from_pretrained(
                self.model_path,
            )
            config.use_cache = False
            lora_ckpt = None
            self.zs = None

            # pretrained_pruned_model = None
            pretrained_pruned_model = 'output/Compresso-finetune-s0-lr0.0001-reglr0.1-warmup0/alpaca_mmlu_fulllora/epoch1'

            config = set_lora_args(config, False, '')
            if pretrained_pruned_model is not None:
                config = set_lora_args(config, True, 'Q.K.V.O.F')
                peft = pretrained_pruned_model
                lora_ckpt = os.path.join(peft, 'lora_weights.pt')
                if not os.path.exists(lora_ckpt):
                    print('No lora module found, ignored!')
                    lora_ckpt = None
                    config.lora_param = ''
            # lora_ckpt = None  # no lora
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.model_path
            )
            self.model = LlamaForCausalLM.from_pretrained(
                LlamaForCausalLM,
                self.model_path,
                from_tf=False,
                config=config,
                lora_ckpt = lora_ckpt
            )
            self.model.half()
            self.model.eval()
            self.model.to(self.device)

            if pretrained_pruned_model is not None:
                self.zs = load_zs(os.path.join(pretrained_pruned_model, 'zs.pt'))
                for key in self.zs:
                    self.zs[key] = self.zs[key].detach().to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        if self.use_template:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            )
            text = template.format_map(dict(instruction=prompt))
        else:
            text = prompt

        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if "65b" in self.model_path.lower():
            self.max_input_length = 1024
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.device)

        if self.zs is not None:
            inputs["zs"] = self.zs

        from transformers import GenerationConfig
        class GenerationConfig_my(GenerationConfig): 
            def validate(self):
                if self.early_stopping not in {True, False, "never"}:
                    raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")

            def update(self, **kwargs):
                to_remove = []
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        to_remove.append(key)

                # remove all the attributes that were updated, without modifying the input dict
                unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
                return unused_kwargs
        generation_config = GenerationConfig_my()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            generation_config=generation_config,
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        outputs = outputs.sequences
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if self.zs is not None:
            inputs["zs"] = self.zs
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def noop(*args, **kwargs):
    assert args is not None
    assert kwargs is not None


def load_quant(
    model,
    checkpoint,
    wbits,
    groupsize=-1,
    fused_mlp=True,
    warmup_autotune=True,
):
    config = LlamaConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()

    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]

    quant.make_quant_linear(model, layers, wbits, groupsize)
    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)

    model.seqlen = 2048
    print("Done.")
    return model

def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        seq_to_seq=SeqToSeqModel,
        causal=CausalModel,
        llama=LlamaModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Write an email about an alpaca that likes flan.",
    model_name: str = "seq_to_seq",
    model_path: str = "google/flan-t5-base",
    **kwargs,
):
    model = select_model(model_name, model_path=model_path, **kwargs)
    print(locals())
    print(model.run(prompt))


"""
p modeling.py test_model --model_name causal --model_path gpt2
p modeling.py test_model --model_name llama --model_path decapoda-research/llama-7b-hf
p modeling.py test_model --model_name llama --model_path chavinlo/alpaca-native
p modeling.py test_model --model_name chatglm --model_path THUDM/chatglm-6b
p modeling.py test_model --model_name llama --model_path TheBloke/koala-7B-HF
p modeling.py test_model --model_name llama --model_path eachadea/vicuna-13b --load_8bit
p modeling.py test_model --model_name causal --model_path togethercomputer/GPT-NeoXT-Chat-Base-20B --load_8bit
p modeling.py test_model --model_name llama --model_path huggyllama/llama-7b --lora_path tloen/alpaca-lora-7b
p modeling.py test_model --model_name seq_to_seq --model_path google/flan-t5-xl --lora_path declare-lab/flan-alpaca-xl-lora
p modeling.py test_model --model_name openai --model_path openai_info.json
p modeling.py test_model --model_name rwkv --model_path https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-7B-v11-Eng99%25-Other1%25-20230427-ctx8192.pth
p modeling.py test_model --model_name causal --model_path mosaicml/mpt-7b-instruct
p modeling.py test_model --model_name gptq --model_path TheBloke/alpaca-lora-65B-GPTQ-4bit --quantized_path alpaca-lora-65B-GPTQ-4bit-128g.safetensors
"""


if __name__ == "__main__":
    Fire()
