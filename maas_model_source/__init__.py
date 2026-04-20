#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:

from .qwen2_72b_service import qwen2_72b_query, qwen2_72b_query_with_plugin
from .qwen2_72b_with_lora_plugin import plugins as lora_plugins
from functools import partial

model_function_register = {
    0: qwen2_72b_query,  # 原始模型
}


def init_model_function():
    # 先重置基座模型路由，再按当前插件列表重建，避免脏插件映射残留
    model_function_register.clear()
    model_function_register[0] = qwen2_72b_query
    for plugin_id in list(lora_plugins.keys()):
        model_function_register[plugin_id] = partial(qwen2_72b_query_with_plugin, plugin_id=plugin_id)
