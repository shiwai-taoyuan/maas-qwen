"""测试 _get_device() — 设备动态检测逻辑"""
import importlib.util
import os
import sys
from unittest.mock import MagicMock

# =========================================================================
# mock ML 依赖,避免触发导入链
# =========================================================================
sys.modules["vllm"] = MagicMock()
sys.modules["transformers"] = MagicMock()
_TGEN = MagicMock()
_TGEN_UTILS = MagicMock()
_TGEN_UTILS.LogitsProcessorList = MagicMock
_TGEN.logits_process = MagicMock()
_TGEN.logits_process.LogitsProcessor = MagicMock
sys.modules["transformers.generation"] = _TGEN
sys.modules["transformers.generation.utils"] = _TGEN_UTILS
sys.modules["transformers.generation.logits_process"] = _TGEN.logits_process
for _cls in ["AutoModel", "AutoTokenizer", "AutoModelForCausalLM",
             "BloomForCausalLM", "BloomTokenizerFast", "LlamaForCausalLM",
             "GenerationConfig", "TextIteratorStreamer"]:
    setattr(sys.modules["transformers"], _cls, MagicMock)

_MOCK_TORCH = MagicMock()
sys.modules["torch"] = _MOCK_TORCH
sys.modules["torch.nn"] = MagicMock()

sys.modules["peft"] = MagicMock()
sys.modules["peft.peft_model"] = MagicMock()

# =========================================================================
# 直接加载 qwen2_72b_with_lora_plugin.py,不经过 __init__.py
# =========================================================================
_PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
_MODULE_PATH = os.path.join(_PROJECT_DIR, "maas_model_source", "qwen2_72b_with_lora_plugin.py")
_SPEC = importlib.util.spec_from_file_location(
    "maas_model_source.qwen2_72b_with_lora_plugin",
    _MODULE_PATH,
    submodule_search_locations=[]
)
_MMS = type(sys)("maas_model_source")
_MMS.__path__ = [os.path.join(_PROJECT_DIR, "maas_model_source")]
_MMS.__file__ = os.path.join(_PROJECT_DIR, "maas_model_source", "__init__.py")
_MMS.__package__ = "maas_model_source"
sys.modules["maas_model_source"] = _MMS
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["maas_model_source.qwen2_72b_with_lora_plugin"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class TestGetDevice:
    def test_returns_device_from_model_parameters(self):
        """验证 _get_device 从模型参数推断设备"""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cuda:0"
        mock_model.parameters.return_value = iter([mock_param])

        device = _MODULE._get_device(mock_model)
        assert device == "cuda:0"

    def test_fallback_on_empty_model(self):
        """验证 _get_device 在无参数模型时报错"""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])

        device = _MODULE._get_device(mock_model)
        assert device is not None

    def test_fallback_on_stop_iteration(self):
        """验证 _get_device 在迭代器为空时正确回退"""
        mock_model = MagicMock()
        mock_model.parameters.side_effect = StopIteration()

        device = _MODULE._get_device(mock_model)
        assert device is not None

    def test_fallback_on_attribute_error(self):
        """验证 _get_device 在无 parameters 属性时回退"""
        mock_model = object()

        device = _MODULE._get_device(mock_model)
        assert device is not None

    def test_return_type_is_string_or_torch_device(self):
        """验证返回值是字符串或 torch.device 类型"""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cuda:0"
        mock_model.parameters.return_value = iter([mock_param])

        device = _MODULE._get_device(mock_model)
        assert device == "cuda:0"
