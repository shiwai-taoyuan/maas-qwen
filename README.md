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


