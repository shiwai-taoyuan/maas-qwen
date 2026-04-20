#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:

import os
import signal
from tools.shared import cmd_opts
from tools import timer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from configs import logger
import threading
from tools.redis_tool import RedisFactory
import json
from configs import (
    REDIS_DB_CORE,
    REDIS_PLUGIN_CONFIG_KEY,
    QWEN2_72B_INSTRUCT_MODEL_ABS_DIR,
    PLUGIN_REFRESH_INTERVAL_SECONDS,
    API_MAX_CONCURRENCY,
)
import time
from server.runtime_state import set_service_state, resize_concurrency

plugin_refresh_signature = ""

mode = os.getenv("MODE", default="prd")
logger.info(f"running in the mode {mode}")

startup_timer = timer.Timer()


def create_api(app):
    from server.api.api import Api
    api = Api(app)
    return api


def setup_middleware(app: FastAPI):
    # reset current middleware to allow modifying user provided list
    app.middleware_stack = None
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if cmd_opts.cors_allow_origins and cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','),
                           allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=[
            '*'],
            allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_methods=['*'],
                           allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'],
                           allow_credentials=True, allow_headers=['*'])
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly


def initialize():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    from maas_model_source import init_model_function
    from maas_model_source.qwen2_72b_model import get_qwen2_72b_model
    from maas_model_source.plugin_manager import init_now_model, plugin_init
    from maas_model_source.qwen2_72b_with_lora_plugin import find_all_lora_plugins

    # 模型文件挂载，不进行下载
    # logger.info("begin download  model checkpoints")
    # download_models()
    logger.info("begin loading base model")

    get_qwen2_72b_model(QWEN2_72B_INSTRUCT_MODEL_ABS_DIR)

    init_now_model()
    plugin_init()
    find_all_lora_plugins()
    init_model_function()
    resize_concurrency(API_MAX_CONCURRENCY)
    set_service_state(ready=True, model_loaded=True, last_error="")
    plugin_thread = threading.Thread(target=init_and_refresh_plugin_config, daemon=True)
    plugin_thread.start()


def is_same_compare_plugin(one_plugin, another_plugin):
    """
    比较两个插件是否一致
    """
    type_compare = one_plugin["type"] == another_plugin["type"]
    one_urls = list(sorted(one_plugin["urls"], key=lambda x: x["url"]))
    another_urls = list(sorted(another_plugin["urls"], key=lambda x: x["url"]))
    url_compare = all(one_url["url"] == another_url["url"] and one_url["file_name"] == another_url["file_name"]
                      for one_url, another_url in zip(one_urls, another_urls))
    return type_compare and url_compare


def init_and_refresh_plugin_config():
    import configs
    global plugin_refresh_signature
    while True:
        try:
            rc = RedisFactory.get_or_create(REDIS_DB_CORE)
            redis_value = rc.get(REDIS_PLUGIN_CONFIG_KEY)
            # logger.info(f"find plugin redis {redis_value}")
            plugins = []
            if redis_value:
                plugins = json.loads(str(redis_value, 'utf-8'))['qwen2d5-72b']
            already_plugins = dict()
            for origin_plugin in configs.PLUGINS:
                plugin_id = origin_plugin["plugin_id"]
                already_plugins[plugin_id] = origin_plugin
            for new_plugin in plugins:
                plugin_id = new_plugin["plugin_id"]
                if plugin_id not in already_plugins:
                    logger.info(f"find new plugin = {str(new_plugin)}")
                    already_plugins[plugin_id] = new_plugin
                else:
                    origin_plugin = already_plugins[plugin_id]
                    overwrite = new_plugin.get("overwrite")
                    if overwrite:  # 只有重写的情况下，需要进行比对，否则无需比对
                        is_same = is_same_compare_plugin(origin_plugin, new_plugin)
                        if not is_same:  # 不相同，则需要新的
                            logger.info(
                                f"origin plugin is not same as new plugin, use new plugin:"
                                f" origin={str(origin_plugin)},new={str(new_plugin)} ")
                            already_plugins[plugin_id] = new_plugin
                        else:
                            already_plugins[plugin_id] = origin_plugin
                    else:
                        already_plugins[plugin_id] = origin_plugin
            new_plugins = list(already_plugins.values())
            signature = json.dumps(new_plugins, sort_keys=True, ensure_ascii=False)
            if signature != plugin_refresh_signature:
                logger.info("plugin config changed, refreshing plugin registry")
                configs.PLUGINS = new_plugins
                from maas_model_source.plugin_manager import plugin_init
                plugin_init()
                from maas_model_source.qwen2_72b_with_lora_plugin import find_all_lora_plugins
                find_all_lora_plugins()
                from maas_model_source import init_model_function
                init_model_function()
                plugin_refresh_signature = signature
        except:
            logger.exception("")
            set_service_state(last_error="plugin refresh failed")
        time.sleep(PLUGIN_REFRESH_INTERVAL_SECONDS)


def api_only():
    print('程序开始运行!')
    app = create_app()
    api = app.state.api

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name=cmd_opts.server_name if cmd_opts.server_name else "0.0.0.0",
               port=cmd_opts.port if cmd_opts.port else 9001,
               log_level=cmd_opts.uvicorn_log_level)


def create_app():
    skip_model_init = os.getenv("SKIP_MODEL_INITIALIZATION", "0").lower() in {"1", "true", "yes"}
    set_service_state(ready=False, model_loaded=False, last_error="")
    if skip_model_init:
        logger.warning("skip model initialization for smoke test mode")
        set_service_state(ready=False, model_loaded=False, last_error="model initialization skipped")
    else:
        initialize()
    app = FastAPI()
    setup_middleware(app)
    api = create_api(app)
    app.state.api = api
    return app


if __name__ == "__main__":
    api_only()
