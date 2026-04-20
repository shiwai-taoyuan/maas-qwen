#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:
import os.path
from configs import QWEN2_72B_INSTRUCT_MODEL_ABS_DIR, PROJECT_DIR, logger
from tools.file_tool import check_or_download_model_file

finish_download = False


def is_finish_download():
    global finish_download
    return finish_download


def download_models():
    """
    将需要提前下载的模型在此处配置，服务启动时将自动检查
    :return:
    """
    model_dir = QWEN2_72B_INSTRUCT_MODEL_ABS_DIR
    urls_and_relative_paths = [
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/checksum.md5"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/config.json"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/merges.txt"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/model-00001-of-00004.safetensors"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/model-00002-of-00004.safetensors"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/model-00003-of-00004.safetensors"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/model-00004-of-00004.safetensors"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/model.safetensors.index.json"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/tokenizer_config.json"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/tokenizer.json"
        ],
        [
            "https://placeholder.invalid/REDACTED",
            f"{model_dir}/vocab.json"
        ],
    ]
    check_or_download_model_file(urls_and_relative_paths, root_dir=PROJECT_DIR)
    global finish_download
    finish_download = True
