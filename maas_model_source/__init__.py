#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:

from .qwen2_72b_service import qwen2_72b_query, qwen2_72b_query_with_plugin
from functools import partial
import threading

model_function_register = {
    0: qwen2_72b_query,  # 原始模型
}
_model_function_register_lock = threading.Lock()


def init_model_function():
    from .qwen2_72b_with_lora_plugin import plugins as lora_plugins

    new_register = {0: qwen2_72b_query}
    for plugin_id in lora_plugins.keys():
        new_register[plugin_id] = partial(qwen2_72b_query_with_plugin, plugin_id=plugin_id)
    with _model_function_register_lock:
        model_function_register.clear()
        model_function_register.update(new_register)
