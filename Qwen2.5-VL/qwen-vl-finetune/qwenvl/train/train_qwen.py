# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
import math
from typing import Dict, List
import shutil
import sys
from pathlib import Path
import contextlib, torch
import torch.distributed as dist
import deepspeed
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from deepspeed import zero as ds_zero

local_rank = None
import contextlib, torch, torch.distributed as dist

import contextlib, torch
import torch.distributed as dist
from deepspeed import zero as ds_zero
def are_embeddings_tied(model):
    """Check if input and output embeddings are tied (share memory)."""
    return model.config.tie_word_embeddings
   

def reinitialize_new_tokens(model, old_len, new_len):
    """Reinitialize only the embeddings for new tokens."""
    is_tied =  are_embeddings_tied(model)
    if is_deepspeed_zero3_enabled():
        params_to_gather = [model.get_input_embeddings().weight]
        if not is_tied:
            params_to_gather.append(model.get_output_embeddings().weight)
        
        with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
            if dist.get_rank() == 0:
                # Initialize only new tokens
                print("^^^^^^^^^^^^^^",is_tied, model.get_input_embeddings().weight.data.shape, model.get_output_embeddings().weight.data.shape, model.lm_head.weight.data.shape, old_len, new_len)
                print(model.get_input_embeddings().weight.shape)
                print(model.get_output_embeddings().weight.shape)
                model.get_input_embeddings().weight.data[old_len:new_len].normal_(mean=0.0, std=0.02)
                if is_tied:
                    model.get_output_embeddings().weight.data[old_len:new_len] = model.get_input_embeddings().weight.data[old_len:new_len]
                else:
                    model.get_output_embeddings().weight.data[old_len:new_len].normal_(mean=0.0, std=0.02)
    
    else:
        model.get_input_embeddings().weight.data[old_len:new_len].normal_(mean=0.0, std=0.02)
        
        if is_tied:
            model.get_output_embeddings().weight.data[old_len:new_len] = model.get_input_embeddings().weight.data[old_len:new_len]
        else:
            model.get_output_embeddings().weight.data[old_len:new_len].normal_(mean=0.0, std=0.02)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def load_new_tokens(tokens_file: str) -> List[str]:
    """Load new tokens from a text file (one token per line)."""
    tokens = []
    with open(tokens_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token and not token.startswith('#'):  # Skip empty lines and comments
                tokens.append(token)
    return tokens

def add_tokens_to_tokenizer(tokenizer, new_tokens: List[str]) -> int:
    """Add new tokens to tokenizer and return number of added tokens."""
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens_filtered = [token for token in new_tokens if token not in existing_tokens]
    
    if new_tokens_filtered:
        tokenizer.add_tokens(new_tokens_filtered)
        rank0_print(f"Added {len(new_tokens_filtered)} new tokens to tokenizer")
        rank0_print(f"Skipped {len(new_tokens) - len(new_tokens_filtered)} existing tokens")
    else:
        rank0_print("All tokens already exist in the vocabulary!")
    
    return len(new_tokens_filtered)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
    
    if model_args.tune_embeddings:
        model.get_input_embeddings().weight.requires_grad = True
    else:
        model.get_input_embeddings().weight.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    ## Newly added ##
    if model_args.new_tokens_file:
        old_len = len(tokenizer)
        new_tokens = load_new_tokens(model_args.new_tokens_file)
        num_added = add_tokens_to_tokenizer(tokenizer, new_tokens)
        new_len = len(tokenizer)
        num_emb = model.get_input_embeddings().num_embeddings
        
        if new_len > num_emb:
            model.resize_token_embeddings(new_len, pad_to_multiple_of=128)
            model.config.vocab_size = math.ceil(new_len / 128) * 128
            print(f"Resized model embeddings from {num_emb} to {new_len}")
        reinitialize_new_tokens(model, old_len, new_len)
    ## Newly added ##            

    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    if model_args.new_tokens_file:
        rank0_print("Saving updated tokenizer...")
        tokenizer.save_pretrained(training_args.output_dir)
    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
