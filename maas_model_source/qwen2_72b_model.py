#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:

from configs import (
    QWEN2_72B_INSTRUCT_MODEL_ABS_DIR,
    TORCH_DTYPE,
    VLLM_TENSOR_PARALLEL_SIZE,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_SEQ_LEN_TO_CAPTURE,
    VLLM_MAX_MODEL_LEN,
)
from transformers import AutoTokenizer
from vllm import LLM

qwen2_72b_base_model = None
qwen2_72b_base_tokenizer = None
qwen2_72b_model_path = QWEN2_72B_INSTRUCT_MODEL_ABS_DIR


def get_qwen2_72b_model(model_path=qwen2_72b_model_path):
    # global qwen2_72b_base_model, qwen2_72b_base_tokenizer
    # if not qwen2_72b_base_model:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         torch_dtype=TORCH_DTYPE,
    #         device_map="auto"
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     model = model.eval()
    #     qwen2_72b_base_tokenizer = tokenizer
    #     qwen2_72b_base_model = model
    # return qwen2_72b_base_model, qwen2_72b_base_tokenizer

    # 使用vllm加速推理
    global qwen2_72b_base_model, qwen2_72b_base_tokenizer
    if not qwen2_72b_base_model:
        qwen2_72b_base_tokenizer = AutoTokenizer.from_pretrained(model_path)
        qwen2_72b_base_model = LLM(
            model=model_path,
            dtype=TORCH_DTYPE,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_seq_len_to_capture=VLLM_MAX_SEQ_LEN_TO_CAPTURE,
            max_model_len=VLLM_MAX_MODEL_LEN,
        )

    return qwen2_72b_base_model, qwen2_72b_base_tokenizer


