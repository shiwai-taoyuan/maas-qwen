#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:
from .qwen2_72b_model import get_qwen2_72b_model
from .qwen2_72b_with_lora_plugin import unload_lora_model, load_lora_model
from .qwen2_72b_with_lora_plugin import plugins as lora_plugins
from tools.plugin_tool import plugin_conflict_check, download_plugins
from configs import ABS_PROJECT_PLUGINS_DIR, ALLOW_PLUGIN_TYPES, logger
import configs

now_status = "origin"  # 当前的模型状态，有两种：lora，origin
now_model = None
now_tokenizer = None


def init_now_model():
    global now_model, now_status, now_tokenizer
    if not now_model:
        qwen2_72b_base_model, qwen2_72b_base_tokenizer = get_qwen2_72b_model()
        now_model = qwen2_72b_base_model
        now_tokenizer = qwen2_72b_base_tokenizer
        now_status = "origin"
    return now_model, now_tokenizer


def load_model_with_plugin(plugin_id):
    if plugin_id in lora_plugins:
        return load_lora_plugins(plugin_id)
    else:
        raise Exception(f"plugin initialize fail ,not found the plugin {plugin_id} for qwen")


def load_origin_model():
    """
    加载 origin 的微调模型参数

    :return:
    """
    global now_model, now_status, now_tokenizer
    if now_status in ["origin"]:
        pass
    elif now_status in ["lora"]:
        logger.info("unloading lora model for origin model")
        now_model = unload_lora_model(now_model)
    else:
        raise NotImplemented
    now_status = "origin"
    return now_model, now_tokenizer


def load_lora_plugins(plugin_id):
    """
    加载 lora 的微调模型参数

    :param plugin_id:
    :return:
    """
    global now_model, now_status, now_tokenizer
    if now_status in ["origin"]:
        now_model = load_lora_model(now_model, plugin_id)
    elif now_status in ["lora"]:
        now_model = unload_lora_model(now_model)
        now_status = 'origin'
        now_model = load_lora_model(now_model, plugin_id)
    else:
        raise NotImplemented
    now_status = "lora"
    return now_model, now_tokenizer


def plugin_init():
    """
    根据plugins的配置信息进行下载和更新
    """
    download_plugin_configs = plugin_conflict_check(ABS_PROJECT_PLUGINS_DIR,
                                                    configs.PLUGINS,
                                                    allow_plugin_types=ALLOW_PLUGIN_TYPES)
    download_plugins(ABS_PROJECT_PLUGINS_DIR, download_plugin_configs)
