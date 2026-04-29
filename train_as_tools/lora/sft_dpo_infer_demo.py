#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Chat inference for SFT/DPO LoRA adapters")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sft_adapter_path", type=str, default="./sft-lora-model/final-lora")
    parser.add_argument("--dpo_adapter_path", type=str, default="./dpo-lora-model/final-lora")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--system_prompt", type=str, default="你是一位专业、礼貌、简洁的电商客服助手。")
    return parser.parse_args()


def load_model_and_tokenizer(model_name):
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_adapters(base_model, args):
    model = PeftModel.from_pretrained(
        base_model,
        args.sft_adapter_path,
        adapter_name="sft",
        is_trainable=False,
    )
    model.load_adapter(args.dpo_adapter_path, adapter_name="dpo", is_trainable=False)
    model.base_model.set_adapter(["sft", "dpo"])
    mode = "dual_active(sft+dpo)"

    model.eval()
    return model, mode


def print_param_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    lora_sft_params = 0
    lora_dpo_params = 0

    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        if ".sft." in name:
            lora_sft_params += param.numel()
        elif ".dpo." in name:
            lora_dpo_params += param.numel()

    lora_total_params = lora_sft_params + lora_dpo_params
    lora_ratio = (lora_total_params / total_params * 100) if total_params > 0 else 0.0

    print(f"Model total params: {total_params:,}")
    print(f"LoRA(sft) params: {lora_sft_params:,}")
    print(f"LoRA(dpo) params: {lora_dpo_params:,}")
    print(f"LoRA total params: {lora_total_params:,} ({lora_ratio:.4f}%)")


def build_model_inputs(tokenizer, messages, model_device):
    if hasattr(tokenizer, "apply_chat_template"):
        # 对支持该参数的 Qwen tokenizer，显式关闭 thinking 模式。
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if isinstance(inputs, torch.Tensor):
            return {"input_ids": inputs.to(model_device)}
        if hasattr(inputs, "items"):
            return {
                k: v.to(model_device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }
        raise TypeError(f"Unsupported chat template output type: {type(inputs)}")

    prompt = ""
    for msg in messages:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += "assistant:"
    encoded = tokenizer(prompt, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"].to(model_device),
        "attention_mask": encoded["attention_mask"].to(model_device),
    }


def generate_reply(model, tokenizer, messages, args):
    model_device = next(model.parameters()).device
    model_inputs = build_model_inputs(tokenizer, messages, model_device)
    prompt_len = model_inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    base_model, tokenizer = load_model_and_tokenizer(args.model_name)
    model, active_mode = load_adapters(base_model, args)

    print(f"Loaded base model: {args.model_name}")
    print(f"Loaded SFT adapter: {args.sft_adapter_path}")
    print(f"Loaded DPO adapter: {args.dpo_adapter_path}")
    print(f"Active adapter mode: {active_mode}")
    print_param_stats(model)
    print("进入对话模式，输入 exit 或 quit 结束。")

    messages = [{"role": "system", "content": args.system_prompt}]
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        answer = generate_reply(model, tokenizer, messages, args)
        print(f"助手: {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
