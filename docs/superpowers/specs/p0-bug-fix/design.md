# P0 级问题修复方案

## 需求描述

修复 maas-qwen 项目中分析出的三个 P0 级代码问题，涉及并发安全、设备兼容性和 Logger 初始化。

## 问题 1：Logger 初始化传参错误

### 现状

`tools/log_tool.py:36` `super().__init__(self)` 将 Logger 实例作为 `name` 参数传给 `logging.Logger.__init__`，name 字段会变成 `<MyLogger object at 0x...>` 之类的对象引用。

### 修复方案

- **修改**：`super().__init__(self)` → `super().__init__("maas-qwen")`
- **原理**：`logging.Logger.__init__(name, level)` 的第一个参数 `name` 是字符串标识名，用于日志层次结构
- **影响范围**：仅修改 `tools/log_tool.py` 一行，对外接口完全不变
- **复杂度**：S

## 问题 2：模型路由表线程不安全

### 现状

`maas_model_source/__init__.py` 中 `model_function_register` 在后台插件刷新线程中通过 `clear()` + 逐条 `[]=` 重建，API 请求线程同时通过 `dict.get()` 读取。clear 之后重建完成之前，请求可能读到空字典或部分字典，导致 `plugin_id not found` 错误。

同时 `qwen2_72b_with_lora_plugin.py` 的 `plugins` dict 通过 `plugins = dict()` + 逐条赋值重建，持有引用者可能读到空字典。

### 修复方案

**`model_function_register`（方案 A：加锁保护）：**

1. 新增 `threading.Lock` 保护重建操作
2. 在外面用局部变量构建好完整新字典，再在锁内 `clear()` + `update()`；缩小临界区
3. `api.py` 在模块顶层的 `from ... import model_function_register` 引用保持不变，始终能读到最新数据

**`plugins`（方案 B：copy-on-write）：**

1. `find_all_lora_plugins()` 改为先构建 `new_plugins = dict()`，最后原子赋值 `plugins = new_plugins`
2. `maas_model_source/__init__.py` 的 `init_model_function()` 将 `from ... import plugins as lora_plugins` 移到函数内部，每次重建时重新读取最新值

### 影响范围

- `maas_model_source/__init__.py`：添加 `threading` import + lock 变量 + 修改 `init_model_function()`
- `maas_model_source/qwen2_72b_with_lora_plugin.py`：修改 `find_all_lora_plugins()` 为 copy-on-write
- 对外接口完全不变
- **复杂度**：M

## 问题 3：硬编码 `.cuda()`

### 现状

`maas_model_source/qwen2_72b_with_lora_plugin.py` 中 `load_lora_model()` 和 `unload_lora_model()` 使用 `model.cuda()` 和 `origin_model.cuda()`，强制将模型移到 GPU 0。在非 CUDA 环境崩溃，在有多 GPU + device_map 的环境可能破坏模型的设备分布。

### 修复方案

新增 `_get_device(model)` 辅助函数：

1. 优先从 model parameters 推断设备（`next(model.parameters()).device`）
2. 若无法推断（无参数 / 非 torch Module），回退到 `torch.cuda.is_available()`
3. 将 `.cuda()` 替换为 `.to(_get_device(...))`

### 影响范围

- `maas_model_source/qwen2_72b_with_lora_plugin.py`：新增 `_get_device()` 函数 + 修改两处 `.cuda()` 调用
- 对外接口完全不变
- **复杂度**：S

## 修改清单

| 文件 | 改动类型 | 复杂度 |
|------|---------|--------|
| `tools/log_tool.py` | 1 行修改 `super().__init__()` | S |
| `maas_model_source/__init__.py` | 添加 `threading` import、lock、重写 `init_model_function()` | M |
| `maas_model_source/qwen2_72b_with_lora_plugin.py` | 新增 `_get_device()`、copy-on-write 重构 `find_all_lora_plugins()`、替换 2 处 `.cuda()` | M |

## 不涉及的部分

- 不需要新增文件或依赖
- 不需要修改 API 接口或配置
- 不需要修改数据模型
- 数据库/SQL 无变更
- 不需要引入新依赖
