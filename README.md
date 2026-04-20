# qwen2.5-72b大模型服务

## 项目依赖

```text
python 3.9 或以上
ubuntu
```

## Quick Install

With pip:

```bash
# 安装完整依赖（运行 + 训练）
pip install -r requirements.txt

# 仅安装线上 API 运行依赖
pip install -r requirements.runtime.txt

# 仅安装训练依赖（需先具备运行依赖）
pip install -r requirements.train.txt
```

## Quick Start

```bash
启动
bash run.sh 

不同的配置文件可以采用MODE环境变量激活
MODE的不同值 dev pre prd 分别对应configs文件夹下的同名配置文件
```

也支持使用 uvicorn 的 factory 方式启动：

```bash
uvicorn maas_qwen2_72b_app:create_app --factory --host 0.0.0.0 --port 9001
```

## 健康检查与指标

- `GET /healthz`：进程存活检查
- `GET /readyz`：模型与服务就绪检查（未就绪返回 503）
- `GET /metrics`：基础请求指标（总请求、成功/失败、平均延迟、按插件统计）

## 常用环境变量

- `QWEN2_MODEL_DIR`：模型目录（支持绝对路径）
- `API_MAX_CONCURRENCY`：API 最大并发槽位
- `API_ACQUIRE_TIMEOUT_SECONDS`：请求排队超时
- `PLUGIN_REFRESH_INTERVAL_SECONDS`：插件配置刷新间隔
- `VLLM_TENSOR_PARALLEL_SIZE`、`VLLM_GPU_MEMORY_UTILIZATION`
- `VLLM_MAX_MODEL_LEN`、`VLLM_MAX_SEQ_LEN_TO_CAPTURE`
- `MAX_UPLOAD_SIZE_MB`、`ENABLE_DOCS`

可参考根目录 `.env.example` 进行配置。

## 自检（不加载模型）

用于验证接口可用性与部署连通性：

```bash
set SKIP_MODEL_INITIALIZATION=1
uvicorn maas_qwen2_72b_app:create_app --factory --host 0.0.0.0 --port 9001
```

预期：
- `GET /healthz` 返回 200
- `GET /readyz` 返回 503（因为跳过了模型初始化）

