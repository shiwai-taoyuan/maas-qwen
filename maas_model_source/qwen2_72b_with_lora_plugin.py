#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:

import os
from configs import ABS_LORA_PLUGINS_DIR, logger
from peft import PeftModel
import torch

plugins = dict()


def find_all_lora_plugins():
    """
    按照目录，据此找到所有的 lora plugin
    """
    global plugins
    plugins = dict()
    for plugin_id in os.listdir(ABS_LORA_PLUGINS_DIR):
        if not os.path.isdir(os.path.join(ABS_LORA_PLUGINS_DIR, plugin_id)):
            continue
        plugins[int(plugin_id)] = os.path.join(ABS_LORA_PLUGINS_DIR, plugin_id)


cache_plugin_params = dict()


def load_lora_model(origin_model, lora_plugin_id):
    """
    加载lora模型
    :param origin_model:   原始模型位置
    :param lora_plugin_id:   lora模型目录
    """
    if lora_plugin_id in plugins:
        lora_model_dir = plugins[lora_plugin_id]
    else:
        raise Exception(
            f"plugin initialize fail ,not found the lora plugin {lora_plugin_id} for qwen2-7b")
    if not check_lora_model(lora_model_dir):
        raise Exception(
            f"plugin initialize fail , the lora plugin {lora_plugin_id} for qwen is incomplete！")

    logger.info(f"loading lora plugin = {lora_plugin_id}")
    model = PeftModel.from_pretrained(origin_model, lora_model_dir)
    model.base_model.enable_adapter_layers()
    model = model.eval()
    model = model.cuda()
    logger.info(f"finish loading lora plugin = {lora_plugin_id}")
    return model


def unload_lora_model(peft_model):
    """
    移除lora模型
    :param peft_model:   peft模型
    """
    logger.info(f"unload lora plugin")
    peft_model.base_model.disable_adapter_layers()
    origin_model = peft_model.get_base_model()

    origin_model = origin_model.eval()
    origin_model = origin_model.cuda()

    return origin_model


def check_lora_model(lora_model_dir):
    """
    检查lora模型文件完整性
    :lora_model_dir:   lora模型路径
    """
    all_lora_files = ['adapter_config.json', 'adapter_model.safetensors']
    now_lora_files = os.listdir(lora_model_dir)
    return all(i in now_lora_files for i in all_lora_files)
