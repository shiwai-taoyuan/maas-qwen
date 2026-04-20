#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:

LOGGER_DIR = "logs/"
FILE_LOG_LEVEL = "DEBUG"
OPEN_FILE_LOG = 1
STREAM_LOG_LEVEL = "DEBUG"
OPEN_STREAM_LOG = 1

QWEN2_72B_INSTRUCT_MODEL_DIR = "checkpoints/Qwen2-72B-Instruct"

PROJECT_PLUGINS_DIR = "plugins"
PROJECT_LORA_PLUGINS_DIR = "plugins/lora"

# redis 配置信息
REDIS_HOST = "***********************"
REDIS_PORT = 0
REDIS_PASSWORD = "***********************"
REDIS_TIMEOUT = 60
REDIS_DB_CORE = 0
REDIS_PLUGIN_CONFIG_KEY = "***********************"


# lora 模型会自动下载到 plugins/lora方法下
# overwrite 为True代表覆盖本地数据
PLUGINS = [
]

TMP_OUTPUT_DIR = "output"

MAAS_SERVER = "******************************"

FTP_DIR = "ftp_dir"

# API 与服务参数
API_MAX_CONCURRENCY = 4
API_ACQUIRE_TIMEOUT_SECONDS = 30
PLUGIN_REFRESH_INTERVAL_SECONDS = 30
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_UPLOAD_EXTENSIONS = [".json", ".txt", ".csv"]
ENABLE_DOCS = 1

# vLLM 参数
TORCH_DTYPE = "bfloat16"
VLLM_TENSOR_PARALLEL_SIZE = 2
VLLM_GPU_MEMORY_UTILIZATION = 0.98
VLLM_MAX_MODEL_LEN = 8192
VLLM_MAX_SEQ_LEN_TO_CAPTURE = 8192

