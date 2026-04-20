#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:
import torch
from typing import List, Tuple, Dict
from .plugin_manager import load_origin_model, load_model_with_plugin
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
from .template import get_conv_template
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
from vllm import SamplingParams

def qwen2_72b_query(query, history=None, do_sample=True, max_length=8192, top_p=0.8, temperature=0.95):
    """

    :param query:
    :param do_sample:
    :param history:
    :param max_length:
    :param top_p:
    :param temperature:
    :return:
    """

    model, tokenizer = load_origin_model()
    response, history = vllm_chat(model, tokenizer, query, history=history,
                             do_sample=do_sample,
                             max_length=max_length,
                             top_p=top_p,
                             temperature=temperature)
    return response


def qwen2_72b_query_with_plugin(query, history=None, plugin_id=None, do_sample=True, max_length=8192,
                               top_p=0.8, temperature=0.95):
    """
    根据插件id，决定调用哪个微调模型
    :param query:
    :param do_sample:
    :param history:
    :param plugin_id: None表示加载原始大模型
    :param max_length:
    :param top_p:
    :param temperature:
    :return:
    """

    model, tokenizer = load_model_with_plugin(plugin_id=plugin_id)
    response, history = chat(model, tokenizer, query, history=history,
                             do_sample=do_sample,
                             max_length=max_length,
                             top_p=top_p,
                             temperature=temperature)
    return response


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def chat(model, tokenizer, query: str, history: List[Dict] = None, do_sample=True, max_length=8192, top_p=0.8,
         temperature=0.95, **kwargs):
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
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = tokenizer.encode(text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, None


def vllm_chat(model, tokenizer, query: str, history: List[Dict] = None, do_sample=True, max_length=8192, top_p=0.8,
         temperature=0.95, **kwargs):
    if history is None:
        history = []
    history.append({"role": "user", "content": query})

    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=1.05, max_tokens=max_length)
    # generate outputs
    outputs = model.generate([text], sampling_params)
    response = outputs[0].outputs[0].text

    return response, None
