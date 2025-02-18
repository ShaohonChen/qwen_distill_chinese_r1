import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Union, Mapping
from functools import partial

import swanlab.integration.transformers
import torch
import datasets
import transformers
import peft
import swanlab
import swanlab.integration
import trl

IGNORE_INDEX = -100
DEFAULT_THINK_START_TOKEN = "<|think_start|>"
DEFAULT_THINK_END_TOKEN = "<|think_end|>"
PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
)
THINKING_RETURN_TEMPLATE = (
    "<|think_end|>\n{reasoning_content}<|think_start|>\n{content}<|im_end|>"
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = (None,)


@dataclass
class DataArguments:
    dataset_id_or_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_files: str = None


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class LoraArguments:
    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_bias: str = "none"


@dataclass
class SwanlabArguments:
    """SwanLab参数的数据类"""

    # 是否使用 SwanLab
    swanlab: bool = field(default=True)
    # SwanLab 用户名
    workspace: str = field(default=None)
    # SwanLab 的项目名
    project: str = field(default="SFT_Chinese_R1")
    # SwanLab 的实验名
    experiment_name: str = field(default="Qwen2.5")
    # SwanLab 工作模式
    mode: str = field(default="cloud")


def smart_tokenizer_and_embedding_resize(
    special_tokens_list: List,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    num_new_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens_list},
        replace_additional_special_tokens=False,
    )

    if model.vocab_size < tokenizer.vocab_size + len(tokenizer.get_added_vocab()):
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def process_datasets(examples, tokenizer):
    # 使用tokenizer的apply_chat_template方法
    prompt_text = PROMPT_TEMPLATE.format_map(examples)
    target_text = THINKING_RETURN_TEMPLATE.format_map(examples)
    prompt_tokens = tokenizer.encode(prompt_text, padding="longest", truncation=True)
    target_tokens = tokenizer.encode(target_text, padding="longest", truncation=True)

    max_len = tokenizer.model_max_length
    inputs_len = len(prompt_tokens) + len(target_tokens)
    padding_len = max_len - inputs_len
    input_tokens = (
        prompt_tokens + target_tokens + [tokenizer.pad_token_id] * padding_len
    )
    labels = [IGNORE_INDEX] * len(prompt_tokens) + target_tokens
    labels = labels[1:] + [IGNORE_INDEX] * (padding_len + 1)
    attention_mask = [1] * inputs_len + [0] * padding_len

    input_tokens = input_tokens[:max_len]
    labels = labels[:max_len]
    attention_mask = attention_mask[:max_len]

    return {
        "input_ids": input_tokens,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def train():
    parser = trl.TrlParser(  # 比transformers的好用多了
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            LoraArguments,
            SwanlabArguments,
        )
    )
    model_args, data_args, training_args, lora_args, swanlab_args = (
        parser.parse_args_and_config()
    )

    distill_data = datasets.load_dataset(
        data_args.dataset_id_or_path, data_files=data_args.data_files
    )

    train_dataset = distill_data["train"].shuffle(seed=42)
    train_dataset = train_dataset.select(range(10000))  # for speed

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding="max_length",
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    process_func = partial(process_datasets, tokenizer=tokenizer)

    train_dataset = train_dataset.map(process_func, num_proc=4)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    # 添加LoRA adaptor
    lora_config = peft.LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias=lora_args.lora_bias,
    )
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    smart_tokenizer_and_embedding_resize(
        special_tokens_list=[DEFAULT_THINK_START_TOKEN, DEFAULT_THINK_END_TOKEN],
        tokenizer=tokenizer,
        model=model,
    )

    callback = []
    if swanlab_args.swanlab:
        swanlab_callback = swanlab.integration.transformers.SwanLabCallback(
            project=swanlab_args.project,
            workspace=swanlab_args.workspace,
            experiment_name=swanlab_args.experiment_name,
            mode=swanlab_args.mode,
        )
        callback.append(swanlab_callback)

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callback,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=None,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
