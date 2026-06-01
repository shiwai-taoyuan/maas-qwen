"""测试线程安全修复 — copy-on-write + threading.Lock

注意: 该项目的模块导入链会触发 vllm/transformers/torch 等重型依赖。
本测试采用 mock 方式隔离这些依赖,仅测试目标函数。
"""
import importlib.util
import os
import sys
import threading
from unittest.mock import patch, MagicMock, PropertyMock

# =========================================================================
# 在 import 项目模块之前,先 mock 掉所有重型 ML 依赖
# =========================================================================
_MOCK_VLLM = MagicMock()
_MOCK_VLLM.SamplingParams = MagicMock
sys.modules["vllm"] = _MOCK_VLLM

_MOCK_TRANSFORMERS = MagicMock()
sys.modules["transformers"] = _MOCK_TRANSFORMERS
# transformers.generation.utils.LogitsProcessorList
_MOCK_TGEN = MagicMock()
_MOCK_TGEN_UTILS = MagicMock()
_MOCK_TGEN_UTILS.LogitsProcessorList = MagicMock
_MOCK_TGEN.logits_process = MagicMock()
_MOCK_TGEN.logits_process.LogitsProcessor = MagicMock
sys.modules["transformers.generation"] = _MOCK_TGEN
sys.modules["transformers.generation.utils"] = _MOCK_TGEN_UTILS
sys.modules["transformers.generation.logits_process"] = _MOCK_TGEN.logits_process
# transformers classes
for _cls in ["AutoModel", "AutoTokenizer", "AutoModelForCausalLM",
             "BloomForCausalLM", "BloomTokenizerFast", "LlamaForCausalLM",
             "GenerationConfig", "TextIteratorStreamer"]:
    setattr(_MOCK_TRANSFORMERS, _cls, MagicMock)

_MOCK_TORCH = MagicMock()
_MOCK_TORCH.cuda.is_available = MagicMock(return_value=False)
_MOCK_TORCH.LongTensor = MagicMock
_MOCK_TORCH.FloatTensor = MagicMock
sys.modules["torch"] = _MOCK_TORCH
sys.modules["torch.nn"] = MagicMock()

_MOCK_PEFT = MagicMock()
_MOCK_PEFT.PeftModel = MagicMock
sys.modules["peft"] = _MOCK_PEFT

# mock peft submodules
sys.modules["peft.peft_model"] = MagicMock()

# =========================================================================
# 使用 importlib 直接加载目标模块,不经过 maas_model_source/__init__.py
# =========================================================================
_PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
_MODULE_PATH = os.path.join(_PROJECT_DIR, "maas_model_source", "qwen2_72b_with_lora_plugin.py")
_SPEC = importlib.util.spec_from_file_location(
    "maas_model_source.qwen2_72b_with_lora_plugin",
    _MODULE_PATH,
    submodule_search_locations=[]
)

# 预创建 package package 的 namespace 避免运行 __init__.py
_MMS = type(sys)("maas_model_source")
_MMS.__path__ = [os.path.join(_PROJECT_DIR, "maas_model_source")]
_MMS.__file__ = os.path.join(_PROJECT_DIR, "maas_model_source", "__init__.py")
_MMS.__package__ = "maas_model_source"
sys.modules["maas_model_source"] = _MMS

# 加载目标模块
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["maas_model_source.qwen2_72b_with_lora_plugin"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class TestFindAllLoraPlugins:
    """测试 qwen2_72b_with_lora_plugin.py 的 copy-on-write 机制"""

    def test_find_all_replaces_plugins_atomically(self):
        """验证 find_all_lora_plugins() 原子性替换 plugins 字典"""
        ftp = _MODULE  # alias

        with patch.object(ftp, "os") as mock_os:
            mock_os.listdir.return_value = ["1", "2", "3"]
            mock_os.path.isdir.return_value = True
            mock_os.path.join.side_effect = os.path.join

            ftp.plugins.clear()
            assert len(ftp.plugins) == 0

            ftp.find_all_lora_plugins()

            assert len(ftp.plugins) == 3
            assert ftp.plugins[1].endswith("/1")
            assert ftp.plugins[2].endswith("/2")
            assert ftp.plugins[3].endswith("/3")

    def test_find_all_filters_non_dirs(self):
        """验证 find_all_lora_plugins() 跳过非目录条目"""
        ftp = _MODULE

        with patch.object(ftp, "os") as mock_os:
            mock_os.listdir.return_value = ["1", "file.txt", "2", ".hidden"]
            mock_os.path.isdir.side_effect = lambda p: p.endswith("/1") or p.endswith("/2")
            mock_os.path.join.side_effect = os.path.join

            ftp.plugins.clear()
            ftp.find_all_lora_plugins()

            assert len(ftp.plugins) == 2
            assert 1 in ftp.plugins
            assert 2 in ftp.plugins

    def test_concurrent_reads_see_consistent_state(self):
        """并发读取 plugins 字典时不会看到空字典"""
        ftp = _MODULE
        ftp.plugins = {1: "/fake/1", 2: "/fake/2"}

        read_results = []

        def reader():
            read_results.append(dict(ftp.plugins))

        with patch.object(ftp, "os") as mock_os:
            mock_os.listdir.return_value = ["3", "4"]
            mock_os.path.isdir.return_value = True
            mock_os.path.join.side_effect = os.path.join

            t1 = threading.Thread(target=reader)
            t2 = threading.Thread(target=ftp.find_all_lora_plugins)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            for r in read_results:
                assert len(r) > 0, "读取到空字典！存在线程安全问题"

    def test_check_lora_model_detects_incomplete(self):
        with patch.object(_MODULE, "os") as mock_os:
            mock_os.listdir.return_value = ["adapter_config.json"]
            assert _MODULE.check_lora_model("/fake/dir") is False

    def test_check_lora_model_detects_complete(self):
        with patch.object(_MODULE, "os") as mock_os:
            mock_os.listdir.return_value = [
                "adapter_config.json", "adapter_model.safetensors"]
            assert _MODULE.check_lora_model("/fake/dir") is True
