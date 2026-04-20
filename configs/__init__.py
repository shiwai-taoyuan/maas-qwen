#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:

import importlib
import os
from tools.log_tool import get_logger

mode = os.getenv("MODE", default="prd").lower()
_mode_to_module = {
    "prd": "configs.prd.config",
    "pre": "configs.pre.config",
    "dev": "configs.dev.config",
}
if mode not in _mode_to_module:
    mode = "dev"

try:
    config = importlib.import_module(_mode_to_module[mode])
except ModuleNotFoundError:
    # pre 环境配置缺失时回退到 prd，避免启动即失败
    mode = "prd"
    config = importlib.import_module(_mode_to_module[mode])

LOGGER_DIR = config.LOGGER_DIR
OPEN_FILE_LOG = config.OPEN_FILE_LOG
FILE_LOG_LEVEL = config.FILE_LOG_LEVEL
OPEN_STREAM_LOG = config.OPEN_STREAM_LOG
STREAM_LOG_LEVEL = config.STREAM_LOG_LEVEL

logger = get_logger(logger_dir=LOGGER_DIR, open_file_log=OPEN_FILE_LOG, file_log_level=FILE_LOG_LEVEL,
                    open_stream_log=OPEN_STREAM_LOG, stream_log_level=STREAM_LOG_LEVEL)

logger.info(f"running in the mode {mode}")

# 项目路径
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
# 模型路径
QWEN2_72B_INSTRUCT_MODEL_DIR = os.getenv("QWEN2_MODEL_DIR", config.QWEN2_72B_INSTRUCT_MODEL_DIR)
if os.path.isabs(QWEN2_72B_INSTRUCT_MODEL_DIR):
    QWEN2_72B_INSTRUCT_MODEL_ABS_DIR = QWEN2_72B_INSTRUCT_MODEL_DIR
else:
    QWEN2_72B_INSTRUCT_MODEL_ABS_DIR = os.path.join(PROJECT_DIR, QWEN2_72B_INSTRUCT_MODEL_DIR)
os.makedirs(QWEN2_72B_INSTRUCT_MODEL_ABS_DIR, exist_ok=True)

# 微调参数插件的绝对路径
ABS_PROJECT_PLUGINS_DIR = os.path.join(PROJECT_DIR, config.PROJECT_PLUGINS_DIR)
ABS_LORA_PLUGINS_DIR = os.path.join(PROJECT_DIR, config.PROJECT_LORA_PLUGINS_DIR)
os.makedirs(ABS_PROJECT_PLUGINS_DIR, exist_ok=True)
os.makedirs(ABS_LORA_PLUGINS_DIR, exist_ok=True)
ALLOW_PLUGIN_TYPES = [
    folder for folder in os.listdir(ABS_PROJECT_PLUGINS_DIR)
    if os.path.isdir(os.path.join(ABS_PROJECT_PLUGINS_DIR, folder))
]

# 插件模型
PLUGINS = config.PLUGINS

# redis 配置信息
REDIS_DB_CORE = config.REDIS_DB_CORE
REDIS_PLUGIN_CONFIG_KEY = config.REDIS_PLUGIN_CONFIG_KEY
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
REDIS_TIMEOUT = config.REDIS_TIMEOUT

# 输出的路径
TMP_OUTPUT_DIR = os.path.join(PROJECT_DIR, config.TMP_OUTPUT_DIR)
os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)

# 消息队列结构,如果没有配置，则为空
MESSAGE_MIDDLEWARE = getattr(config, 'MESSAGE_MIDDLEWARE', [])
CONNECTION_CONFIG = getattr(config, 'CONNECTION_CONFIG', None)

MAAS_SERVER = config.MAAS_SERVER

FTP_DIR = os.path.join(PROJECT_DIR, config.FTP_DIR)
os.makedirs(FTP_DIR, exist_ok=True)

TORCH_DTYPE = os.getenv("TORCH_DTYPE", getattr(config, "TORCH_DTYPE", "bfloat16"))

# API 并发/请求控制
API_MAX_CONCURRENCY = int(os.getenv("API_MAX_CONCURRENCY", getattr(config, "API_MAX_CONCURRENCY", 4)))
API_ACQUIRE_TIMEOUT_SECONDS = float(
    os.getenv("API_ACQUIRE_TIMEOUT_SECONDS", getattr(config, "API_ACQUIRE_TIMEOUT_SECONDS", 30))
)
PLUGIN_REFRESH_INTERVAL_SECONDS = int(
    os.getenv("PLUGIN_REFRESH_INTERVAL_SECONDS", getattr(config, "PLUGIN_REFRESH_INTERVAL_SECONDS", 30))
)

# 接口安全参数
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", getattr(config, "MAX_UPLOAD_SIZE_MB", 50)))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = tuple(
    ext.lower() for ext in getattr(config, "ALLOWED_UPLOAD_EXTENSIONS", [".json", ".txt", ".csv"])
)

# vLLM 参数
VLLM_TENSOR_PARALLEL_SIZE = int(
    os.getenv("VLLM_TENSOR_PARALLEL_SIZE", getattr(config, "VLLM_TENSOR_PARALLEL_SIZE", 2))
)
VLLM_GPU_MEMORY_UTILIZATION = float(
    os.getenv("VLLM_GPU_MEMORY_UTILIZATION", getattr(config, "VLLM_GPU_MEMORY_UTILIZATION", 0.98))
)
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", getattr(config, "VLLM_MAX_MODEL_LEN", 8192)))
VLLM_MAX_SEQ_LEN_TO_CAPTURE = int(
    os.getenv("VLLM_MAX_SEQ_LEN_TO_CAPTURE", getattr(config, "VLLM_MAX_SEQ_LEN_TO_CAPTURE", 8192))
)
ENABLE_DOCS = os.getenv("ENABLE_DOCS", str(getattr(config, "ENABLE_DOCS", 1))).lower() in {"1", "true", "yes"}