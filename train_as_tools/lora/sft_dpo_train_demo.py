#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-script pipeline: build SFT data -> train SFT LoRA -> freeze base+SFT -> train DPO LoRA"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sft_output_dir", type=str, default="./sft-lora-model")
    parser.add_argument("--dpo_output_dir", type=str, default="./dpo-lora-model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--sft_learning_rate", type=float, default=5e-5)
    parser.add_argument("--dpo_learning_rate", type=float, default=5e-5)
    parser.add_argument("--sft_max_steps", type=int, default=10)
    parser.add_argument("--dpo_max_steps", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=5)
    return parser.parse_args()


def to_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
                if role and content:
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(content or item))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(value, dict):
        return str(value.get("content", value)).strip()
    return str(value).strip()


def extract_prompt_text(value):
    if isinstance(value, list):
        user_texts = []
        for item in value:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if content:
                    user_texts.append(str(content).strip())
        if user_texts:
            return "\n".join(user_texts).strip()
    return to_text(value)


def extract_response_text(value):
    if isinstance(value, list):
        assistant_texts = []
        for item in value:
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content")
                if content:
                    assistant_texts.append(str(content).strip())
        if assistant_texts:
            return "\n".join(assistant_texts).strip()
    if isinstance(value, dict) and value.get("role") == "assistant":
        return str(value.get("content", "")).strip()
    return to_text(value)


def add_prompt_separator(prompt_text):
    prompt_text = prompt_text.rstrip()
    return f"{prompt_text}\n"


def get_model_kwargs():
    return {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }


def load_base_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, **get_model_kwargs())
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def build_demo_sft_and_dpo_datasets():
    demo_rows = [
        {
            "prompt": "用户：我下单后发现收货地址填错了，订单还没发货，怎么处理？",
            "chosen": "客服：您好，未发货订单可以修改地址。请在订单详情点击“修改地址”，若按钮不可用请把正确地址发给我，我马上帮您提交仓库拦截修改。",
            "rejected": "客服：地址错了就没办法了，你重新买一个吧。",
        },
        {
            "prompt": "用户：快递显示已签收，但我没收到包裹怎么办？",
            "chosen": "客服：非常抱歉给您带来困扰。我先为您登记异常签收核查，请您提供订单号和联系电话。我们会联系快递站点核实，24小时内给您处理结果。",
            "rejected": "客服：系统显示签收了，和我们无关。",
        },
        {
            "prompt": "用户：这件衣服尺码不合适，退货运费谁承担？",
            "chosen": "客服：您好，若因尺码不合适申请七天无理由退货，通常需要您先垫付寄回运费；若商品存在质量问题，我们会承担运费并优先处理退款。",
            "rejected": "客服：退货运费永远都是你出，别问了。",
        },
        {
            "prompt": "用户：我需要开电子发票，抬头和税号在哪里填？",
            "chosen": "客服：您可以在下单页勾选“电子发票”，填写发票抬头和税号；若已下单，可在“我的订单-申请开票”补填信息，审核后会发送到您的邮箱。",
            "rejected": "客服：发票开不了，我们没有这个功能。",
        },
    ]

    sft_rows = []
    dpo_rows = []
    for ex in demo_rows:
        prompt_text = add_prompt_separator(extract_prompt_text(ex["prompt"]))
        chosen_text = extract_response_text(ex["chosen"]).lstrip()
        rejected_text = extract_response_text(ex["rejected"]).lstrip()

        sft_rows.append(
            {
                "prompt": prompt_text,
                "completion": chosen_text,
                "text": f"{prompt_text}{chosen_text}",
            }
        )
        dpo_rows.append(
            {
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
        )

    sft_dataset = Dataset.from_list(sft_rows)
    dpo_dataset = Dataset.from_list(dpo_rows)
    print("Use built-in demo samples only.")
    print(f"SFT dataset size: {len(sft_dataset)}")
    print(f"DPO dataset size: {len(dpo_dataset)}")
    print("First SFT sample:", sft_dataset[0])
    print("First DPO sample:", dpo_dataset[0])
    return sft_dataset, dpo_dataset


def build_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def train_sft_lora(args, sft_dataset):
    print("\n===== Stage 1: Train SFT LoRA =====")
    model, tokenizer = load_base_model_and_tokenizer(args.model_name)
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    sft_args = SFTConfig(
        output_dir=args.sft_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.sft_learning_rate,
        max_steps=args.sft_max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=False,
        bf16=True,
    )

    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": sft_dataset,
    }
    sft_signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sft_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in sft_signature.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_signature.parameters:
        trainer_kwargs["max_seq_length"] = args.max_length

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    final_sft_adapter_dir = f"{args.sft_output_dir}/final-lora"
    trainer.model.save_pretrained(final_sft_adapter_dir)
    tokenizer.save_pretrained(args.sft_output_dir)
    print(f"SFT adapter saved to: {final_sft_adapter_dir}")
    return final_sft_adapter_dir, tokenizer


def freeze_for_dpo(model, train_adapter_name):
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if f".{train_adapter_name}." in name and "lora_" in name:
            param.requires_grad = True


def train_dpo_lora(args, dpo_dataset, sft_adapter_path, tokenizer):
    print("\n===== Stage 2: Freeze Base + SFT, then Train DPO LoRA =====")
    base_model, _ = load_base_model_and_tokenizer(args.model_name)
    model = PeftModel.from_pretrained(base_model, sft_adapter_path, adapter_name="sft", is_trainable=False)
    model.add_adapter("dpo", build_lora_config())
    model.set_adapter("dpo")
    freeze_for_dpo(model, train_adapter_name="dpo")
    model.print_trainable_parameters()

    dpo_args = DPOConfig(
        output_dir=args.dpo_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.dpo_learning_rate,
        max_steps=args.dpo_max_steps,
        beta=args.beta,
        fp16=False,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
    )

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": dpo_dataset,
    }

    ref_base_model, _ = load_base_model_and_tokenizer(args.model_name)
    ref_model = PeftModel.from_pretrained(
        ref_base_model,
        sft_adapter_path,
        adapter_name="sft",
        is_trainable=False,
    )
    ref_model.set_adapter("sft")
    for _, param in ref_model.named_parameters():
        param.requires_grad = False
    ref_model.eval()
    ref_model.print_trainable_parameters()
    trainer_kwargs["ref_model"] = ref_model

    dpo_signature = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in dpo_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in dpo_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "ref_model" not in dpo_signature.parameters:
        trainer_kwargs.pop("ref_model", None)

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()

    final_dpo_adapter_dir = f"{args.dpo_output_dir}/final-lora"
    trainer.model.save_pretrained(final_dpo_adapter_dir)
    tokenizer.save_pretrained(args.dpo_output_dir)
    print(f"DPO adapter saved to: {final_dpo_adapter_dir}")


def main():
    args = parse_args()
    sft_dataset, dpo_dataset = build_demo_sft_and_dpo_datasets()
    sft_adapter_path, tokenizer = train_sft_lora(args, sft_dataset)
    train_dpo_lora(args, dpo_dataset, sft_adapter_path, tokenizer)
    print("\nPipeline done: SFT LoRA + frozen-base/SFT DPO LoRA.")


if __name__ == "__main__":
    main()
