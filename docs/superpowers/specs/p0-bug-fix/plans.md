# P0 问题修复实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复三个 P0 级代码问题：Logger init 传参错误、模型路由表线程不安全、硬编码 `.cuda()`

**Architecture:** 三个问题互不依赖，可并行修复。均在同一文件内局部修改，不改变接口和数据流。

**Tech Stack:** Python 3.9+, FastAPI, PyTorch, vLLM, PEFT

---

### Task 1: 修复 Logger 初始化传参错误

**Files:**
- Modify: `tools/log_tool.py:33-36`

- [ ] **Step 1: 修改 `__init__` 中的 `super()` 调用**

```python
# 修改前
super().__init__(self)

# 修改后
super().__init__("maas-qwen")
```

**预期效果：** `MyLogger` 记录器 name 变为字符串 `"maas-qwen"` 而非 Logger 实例引用。

### Task 2: 修复模型路由表线程安全

**Files:**
- Modify: `maas_model_source/__init__.py:1-20`
- Modify: `maas_model_source/qwen2_72b_with_lora_plugin.py:13-24`

- [ ] **Step 1: `qwen2_72b_with_lora_plugin.py` — `find_all_lora_plugins()` 改为 copy-on-write**

先构建 `new_plugins`，最后原子赋值 `plugins = new_plugins`：

```python
def find_all_lora_plugins():
    global plugins
    new_plugins = dict()
    for plugin_id in os.listdir(ABS_LORA_PLUGINS_DIR):
        if not os.path.isdir(os.path.join(ABS_LORA_PLUGINS_DIR, plugin_id)):
            continue
        new_plugins[int(plugin_id)] = os.path.join(ABS_LORA_PLUGINS_DIR, plugin_id)
    plugins = new_plugins
```

- [ ] **Step 2: `__init__.py` — 模块级 import 改为函数内 import，route 重建加锁保护**

```python
import threading
from functools import partial
from .qwen2_72b_service import qwen2_72b_query, qwen2_72b_query_with_plugin

model_function_register = {0: qwen2_72b_query}
_model_function_register_lock = threading.Lock()

def init_model_function():
    from .qwen2_72b_with_lora_plugin import plugins as lora_plugins
    new_register = {0: qwen2_72b_query}
    for plugin_id in lora_plugins.keys():
        new_register[plugin_id] = partial(qwen2_72b_query_with_plugin, plugin_id=plugin_id)
    with _model_function_register_lock:
        model_function_register.clear()
        model_function_register.update(new_register)
```

**预期效果：** 后台线程重建路由表时，API 请求线程要么看到完整旧表，要么等待锁释放后看到完整新表。

### Task 3: 修复硬编码 `.cuda()`

**Files:**
- Modify: `maas_model_source/qwen2_72b_with_lora_plugin.py`

- [ ] **Step 1: 新增 `_get_device()` 辅助函数**

在 `cache_plugin_params` 变量之后添加：

```python
def _get_device(model):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        pass
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 2: 替换 `load_lora_model` 中的 `.cuda()`**

```python
model = model.to(_get_device(origin_model))
```

- [ ] **Step 3: 替换 `unload_lora_model` 中的 `.cuda()`**

```python
origin_model = origin_model.to(_get_device(peft_model))
```

**预期效果：** LoRA 模型加载/卸载时使用基座模型的设备而非硬编码 GPU 0。
