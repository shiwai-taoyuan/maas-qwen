#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态加载并切换 base / sft / dpo 推理模式。

使用示例:
python train_as_tools/lora/dynamic_adapter_infer.py \
  --base_model_path /path/to/base \
  --sft_lora_path /path/to/sft \
  --dpo_lora_path /path/to/dpo \
  --prompt "你好，请介绍一下人工智能。" \
  --mode dpo
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def infer_input_device(model: torch.nn.Module) -> torch.device:
    """
    在 device_map='auto' 等场景下，尽量推断输入应放置的设备。
    """
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        for dev in hf_device_map.values():
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
        for dev in hf_device_map.values():
            if isinstance(dev, str):
                return torch.device(dev)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def disable_all_adapters(model: PeftModel) -> None:
    """
    兼容不同 PEFT 版本的“关闭所有适配器”行为。
    """
    if hasattr(model, "disable_adapters"):
        model.disable_adapters()
        return
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "disable_adapter_layers"):
        base_model.disable_adapter_layers()
        return
    raise RuntimeError("当前 peft 版本不支持 disable_adapters/disable_adapter_layers。")


def set_active_adapter(model: PeftModel, adapter) -> None:
    """
    兼容不同 PEFT 版本的 set_adapter 调用。
    """
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter)
        return
    raise RuntimeError("当前 peft 版本不支持 set_adapter。")


class DynamicLoraRunner:
    def __init__(
        self,
        base_model_path: str,
        sft_lora_path: str,
        dpo_lora_path: str,
        trust_remote_code: bool = True,
    ) -> None:
        self.base_model_path = base_model_path
        self.sft_lora_path = sft_lora_path
        self.dpo_lora_path = dpo_lora_path
        self.trust_remote_code = trust_remote_code
        self.model: Optional[PeftModel] = None
        self.tokenizer = None

    def load(self) -> None:
        print("正在加载基座模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print("正在加载 SFT-LoRA...")
        model = PeftModel.from_pretrained(
            base_model,
            self.sft_lora_path,
            adapter_name="sft",
        )

        print("正在加载 DPO-LoRA...")
        if hasattr(model, "load_adapter"):
            model.load_adapter(self.dpo_lora_path, adapter_name="dpo")
        else:
            raise RuntimeError("当前 peft 版本不支持 load_adapter，无法动态加载第二个 LoRA。")

        model.eval()
        self.model = model
        print("模型与适配器加载完成。")

    def set_model_mode(self, mode: str) -> None:
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load()。")

        if mode == "base":
            disable_all_adapters(self.model)
            print("已切换模式: base")
            return

        if mode == "sft":
            set_active_adapter(self.model, "sft")
            print("已切换模式: sft")
            return

        if mode == "dpo":
            # 优先尝试与训练一致的双适配器叠加；若当前版本不支持则回退到 dpo 单适配器。
            try:
                set_active_adapter(self.model, ["sft", "dpo"])
                print("已切换模式: dpo (sft + dpo)")
            except Exception:
                set_active_adapter(self.model, "dpo")
                print("已切换模式: dpo (当前 peft 不支持双适配器叠加，已回退为 dpo 单适配器)")
            return

        raise ValueError("mode 必须是 base / sft / dpo")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        mode: str = "dpo",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型或 tokenizer 尚未加载，请先调用 load()。")

        self.set_model_mode(mode)
        input_device = infer_input_device(self.model)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def unload_lora(self, adapter_name: str) -> None:
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load()。")
        peft_cfg = getattr(self.model, "peft_config", {})
        if adapter_name not in peft_cfg:
            print(f"未找到适配器: {adapter_name}")
            return
        if hasattr(self.model, "unload_adapter"):
            self.model.unload_adapter(adapter_name)
            print(f"已卸载 LoRA 适配器: {adapter_name}")
        else:
            print("当前 peft 版本不支持 unload_adapter，跳过卸载。")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="动态切换 base/sft/dpo 的 LoRA 推理脚本")
    parser.add_argument("--base_model_path", required=True, help="基座模型路径")
    parser.add_argument("--sft_lora_path", required=True, help="SFT LoRA 路径")
    parser.add_argument("--dpo_lora_path", required=True, help="DPO LoRA 路径")
    parser.add_argument("--prompt", default="你好，请介绍一下人工智能。", help="输入提示词")
    parser.add_argument("--mode", default="dpo", choices=["base", "sft", "dpo"], help="推理模式")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成长度")
    return parser.parse_args()


def main() -> None:
    args = build_args()
    runner = DynamicLoraRunner(
        base_model_path=args.base_model_path,
        sft_lora_path=args.sft_lora_path,
        dpo_lora_path=args.dpo_lora_path,
    )
    runner.load()
    output = runner.generate(
        prompt=args.prompt,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n===== 输出 =====")
    print(output)


if __name__ == "__main__":
    main()
