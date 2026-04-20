FROM cuda12.1.0-cudnn8-ubuntu22.04:base
ENV TZ "Asia/Shanghai"

WORKDIR /workspace

COPY requirements.runtime.txt requirements.runtime.txt
RUN pip install --no-cache-dir -r requirements.runtime.txt

COPY /output /workspace

# 设置环境变量
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python3", "-m", "maas_qwen2_72b_app", "--port=9001"]
