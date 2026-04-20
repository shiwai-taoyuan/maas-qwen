#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:
import torch
from typing import List, Tuple, Dict
# from .plugin_manager import load_origin_model, load_model_with_plugin
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
from maas_model_source.template import get_conv_template
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
from configs import logger
from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict, PeftModel
import platform
import os


def chat(model, tokenizer, query: str, history: List[Dict] = None, **kwargs):
    if history is None:
        history = []
    system_prompt = ""
    if history:
        if history[0]["role"] == "system":
            system_prompt = history[0]["content"]
            history = history[1:]
        history = [[history[i]["content"], history[i + 1]["content"]] for i in range(0, len(history), 2)]
    prompt_template = get_conv_template("qwen")
    history_messages = history + [[query, ""]]
    text = prompt_template.get_prompt(messages=history_messages, system_prompt=system_prompt)

    # history.append({"role": "user", "content": query})
    #
    # text = tokenizer.apply_chat_template(
    #     history,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # logger.info(text)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    input_ids = tokenizer.encode(text,return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device="cuda")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def check_lora_model(lora_model_dir):
    """
    检查lora模型文件完整性
    :lora_model_dir:   lora模型路径
    """
    all_lora_files = ['adapter_config.json', 'adapter_model.safetensors']
    now_lora_files = os.listdir(lora_model_dir)
    return all(i in now_lora_files for i in all_lora_files)


def init_model_and_tokenizer(qwen2_7b_model_path, lora_model_dir=None):
    if lora_model_dir and not check_lora_model(lora_model_dir):
        raise Exception(f"{lora_model_dir}中的lora模型文件不完整！")
    
    origin_model = AutoModelForCausalLM.from_pretrained(
            qwen2_7b_model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(qwen2_7b_model_path)
    if lora_model_dir:
        # peft_config = LoraConfig.from_pretrained(lora_model_dir)
        # print(peft_config)
        # model = get_peft_model(model, peft_config)
        # set_peft_model_state_dict(model, torch.load(lora_model_path))
        model = PeftModel.from_pretrained(origin_model, lora_model_dir)
        model.base_model.enable_adapter_layers()
    else:
        model = origin_model
    model = model.eval()
    model = model.cuda()
    return model, tokenizer, origin_model


if __name__ == '__main__':

    model_path = "/path/to/local/models/Qwen2-7B-Instruct"
    lora_model_path = "./outputs-sft-qwen-v2-little-T-100epoch/checkpoint-200"
    model, tokenizer, origin_model = init_model_and_tokenizer(model_path, lora_model_path)
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    
    print("欢迎使用 千问2 大模型，clear 清空，stop 终止程序")
    while True:
        query = input("\n输入：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            os.system(clear_command)
            print("欢迎使用 千问2 大模型，clear 清空，stop 终止程序")
            continue
        print("\n千问2：", end="")
        response = chat(model, tokenizer, query)
        print(response, end="", flush=True)


