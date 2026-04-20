#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:
import json
import os
import shutil
from tqdm import tqdm
from configs import logger
from tools.file_tool import save_file_from_url


def plugin_conflict_check(plugin_dir, plugin_configs, allow_plugin_types):
    """
    插件模型冲突检测
    返回需要下载或者覆盖的模型
    """
    already_plugins = set()
    for type_name in os.listdir(plugin_dir):
        if not os.path.isdir(os.path.join(plugin_dir, type_name)):
            continue
        for plugin_id in os.listdir(os.path.join(plugin_dir, type_name)):
            if not os.path.isdir(os.path.join(plugin_dir, type_name, plugin_id)):
                continue
            already_plugins.add(int(plugin_id))
    # logger.info(f"already find plugins in plugin dir = {str(already_plugins)}")
    config_plugins = set()
    download_plugin_configs = []  # 需要下载的模型配置
    for plugin_config in plugin_configs:
        plugin_id = plugin_config["plugin_id"]
        overwrite = plugin_config.get("overwrite", False)
        plugin_type = plugin_config["type"]
        if plugin_type not in allow_plugin_types:
            logger.info(f"not allow plugin type， type={plugin_type}")
            raise Exception(f"conflict plugin id = {plugin_id}")
        if plugin_id in config_plugins:
            logger.info(f"在插件配置中发现冲突的插件id，请重新处理配置，冲突的插件id={plugin_id}")
            raise Exception(f"在插件配置中发现冲突的插件id，请重新处理配置，冲突的插件id={plugin_id}")
        config_plugins.add(plugin_id)
        if plugin_id in already_plugins:
            if overwrite:
                download_plugin_configs.append(plugin_config)
                plugin_config["overwrite"] = False  # 更新操作也会用到这个方法，不需要再重新下载了
                # logger.info(f"plugin id={plugin_id}已经存在,overwrite参数为True，将覆盖原先插件模型文件")
            # else:
            #     logger.info(f"plugin id={plugin_id}已经存在,overwrite参数为False，不进行调整")
        else:
            download_plugin_configs.append(plugin_config)
    # logger.info(f"need to download plugins = {json.dumps(download_plugin_configs)}")
    return download_plugin_configs


def download_plugins(plugin_dir, download_plugin_configs):
    """
    下载模型插件
    """
    for plugin_config in download_plugin_configs:
        try:
            logger.info(f"download plugin = {str(plugin_config)}")
            plugin_type = plugin_config["type"]
            plugin_id = plugin_config["plugin_id"]
            logger.info(f"download plugin ={plugin_id}")
            new_plugin_dir = os.path.join(plugin_dir, plugin_type, str(plugin_id))
            if os.path.isdir(new_plugin_dir):
                shutil.rmtree(new_plugin_dir)
            os.makedirs(new_plugin_dir, exist_ok=True)
            urls = plugin_config["urls"]
            for url_info in tqdm(urls):
                url = url_info["url"]
                file_name = url_info["file_name"]
                save_file_from_url(url, os.path.join(new_plugin_dir, file_name))
        except:
            logger.exception(f"plugin download fail, plugin id={str(plugin_config)}")
