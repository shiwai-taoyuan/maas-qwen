import argparse
import inspect
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Three-stage pipeline: SFT LoRA -> RM LoRA -> GRPO LoRA (all data built-in)"
    )
    parser.add_argument("--model_name", type=str, default="checkpoints/Qwen3-0d6B")
    parser.add_argument("--sft_output_dir", type=str, default="./sft-lora-model")
    parser.add_argument("--rm_output_dir", type=str, default="./rm-lora-model")
    parser.add_argument("--grpo_output_dir", type=str, default="./grpo-lora-model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=256)

    parser.add_argument("--sft_learning_rate", type=float, default=5e-5)
    parser.add_argument("--rm_learning_rate", type=float, default=3e-5)
    parser.add_argument("--grpo_learning_rate", type=float, default=2e-5)
    parser.add_argument("--sft_max_steps", type=int, default=40)
    parser.add_argument("--rm_max_steps", type=int, default=40)
    parser.add_argument("--grpo_max_steps", type=int, default=80)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--reward_clip_low", type=float, default=-5.0)
    parser.add_argument("--reward_clip_high", type=float, default=5.0)
    parser.add_argument("--rm_margin", type=float, default=0.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=20)
    return parser.parse_args()


def should_use_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def get_model_kwargs():
    return {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if should_use_bf16() else torch.float32,
    }


def build_lora_config(task_type):
    kwargs = {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": task_type,
    }
    # RM 任务需要同时训练并保存打分头，避免 score.weight 缺失
    if task_type == TaskType.SEQ_CLS:
        kwargs["modules_to_save"] = ["score"]
    return LoraConfig(**kwargs)


def print_trainable_stats(model, stage_name):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable / total if total > 0 else 0.0
    print(f"[{stage_name}] trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def disable_thinking_mode(model=None, tokenizer=None):
    for obj in [
        tokenizer,
        getattr(model, "config", None),
        getattr(model, "generation_config", None),
    ]:
        if obj is None:
            continue
        for key in ("enable_thinking", "use_thinking", "thinking"):
            if hasattr(obj, key):
                setattr(obj, key, False)


def build_demo_datasets():
    # 固定 20 条客服偏好样本，全部代码内构造
    rows = [
        {
            "prompt": "用户：我下单后发现地址写错了，订单还没发货，怎么办？",
            "chosen": "客服：您好，未发货订单可在订单详情修改地址；若按钮不可用，请提供正确地址，我马上为您提交拦截修改。",
            "rejected": "客服：地址错了就没办法了，你重新买吧。",
        },
        {
            "prompt": "用户：包裹显示已签收，但我没收到。",
            "chosen": "客服：非常抱歉，我先为您登记异常签收，请提供订单号和联系电话，我们会联系站点核查并在24小时内回复。",
            "rejected": "客服：系统写签收就是签收，和我们没关系。",
        },
        {
            "prompt": "用户：衣服尺码不合适想退货，运费谁承担？",
            "chosen": "客服：您好，七天无理由退货通常需您先垫付寄回运费；如商品存在质量问题，运费由我们承担。",
            "rejected": "客服：运费你自己出，不用再问。",
        },
        {
            "prompt": "用户：我想开电子发票，在哪里填写抬头和税号？",
            "chosen": "客服：您可在下单页勾选电子发票并填写抬头税号；若已下单，可在我的订单里申请开票补填信息。",
            "rejected": "客服：不能开发票。",
        },
        {
            "prompt": "用户：优惠券显示不可用，明明没过期。",
            "chosen": "客服：抱歉给您带来不便，请把优惠券截图和商品链接发我，我帮您核对适用范围并尽快处理。",
            "rejected": "客服：不能用就是不能用。",
        },
        {
            "prompt": "用户：商品少发了一件，怎么补发？",
            "chosen": "客服：非常抱歉，麻烦提供开箱照片和订单号，我先为您登记少件，核实后优先补发或退款。",
            "rejected": "客服：少件你自己找快递。",
        },
        {
            "prompt": "用户：下单后多久能发货？",
            "chosen": "客服：您好，现货通常24小时内发出，预售商品以详情页承诺时间为准，发货后会同步物流单号。",
            "rejected": "客服：不知道，等着吧。",
        },
        {
            "prompt": "用户：我想取消订单但找不到按钮。",
            "chosen": "客服：您好，未发货订单可在订单详情点击取消；若按钮不可用，请把订单号发我，我帮您人工取消。",
            "rejected": "客服：取消不了。",
        },
        {
            "prompt": "用户：收到的杯子有裂痕。",
            "chosen": "客服：非常抱歉影响使用，请提供商品照片和外包装照片，我们将优先安排换新或退款。",
            "rejected": "客服：碎了就自己处理。",
        },
        {
            "prompt": "用户：发错颜色了，我买的是黑色收到白色。",
            "chosen": "客服：抱歉给您添麻烦了，请提供订单号和实物照片，我立即为您安排换货并跟进进度。",
            "rejected": "客服：颜色随机发的。",
        },
        {
            "prompt": "用户：退款什么时候到账？",
            "chosen": "客服：您好，退款成功后原路返回，一般1到5个工作日到账，具体以支付渠道处理时效为准。",
            "rejected": "客服：等着就行。",
        },
        {
            "prompt": "用户：我可以修改收货人手机号吗？",
            "chosen": "客服：可以的，未发货订单可在订单详情修改手机号；若已发货，我可协助您联系快递尝试更新。",
            "rejected": "客服：改不了。",
        },
        {
            "prompt": "用户：这个商品支持七天无理由吗？",
            "chosen": "客服：您好，若商品页面无特殊说明，支持七天无理由退货，且商品需保持完好不影响二次销售。",
            "rejected": "客服：不支持。",
        },
        {
            "prompt": "用户：可以开发票到公司抬头吗？",
            "chosen": "客服：可以的，请在开票信息中填写公司名称和税号，审核通过后会开具电子发票发送到您的邮箱。",
            "rejected": "客服：公司票不能开。",
        },
        {
            "prompt": "用户：物流一直不更新，停了三天。",
            "chosen": "客服：抱歉久等了，我这边立即为您发起物流催件并跟进节点，有结果会第一时间通知您。",
            "rejected": "客服：物流慢很正常。",
        },
        {
            "prompt": "用户：想换尺码，流程怎么走？",
            "chosen": "客服：您好，可在订单详情申请换货并选择目标尺码，提交后按页面指引寄回，我们收到后尽快换发。",
            "rejected": "客服：换不了尺码。",
        },
        {
            "prompt": "用户：商品有色差，不想要了。",
            "chosen": "客服：抱歉未达预期，若在七天无理由范围内可申请退货退款，提交后我们会尽快审核处理。",
            "rejected": "客服：色差正常，不退。",
        },
        {
            "prompt": "用户：买了两件，能不能只退一件？",
            "chosen": "客服：可以的，您可在售后申请中选择需要退的那一件提交，我们会按对应商品为您处理退款。",
            "rejected": "客服：只能全退。",
        },
        {
            "prompt": "用户：客服不在线时怎么留言？",
            "chosen": "客服：您可在会话窗口留下订单号和问题描述，系统会自动排队，客服上线后会第一时间回复您。",
            "rejected": "客服：没人就别留了。",
        },
        {
            "prompt": "用户：会员价怎么没有生效？",
            "chosen": "客服：抱歉给您带来困扰，请提供账号和商品链接，我帮您核对会员状态及活动规则后尽快处理。",
            "rejected": "客服：没生效就原价买。",
        },
    ]
    sft_rows = []
    rm_rows = []
    grpo_rows = []
    for ex in rows:
        prompt = ex["prompt"].strip() + "\n"
        chosen = ex["chosen"].strip()
        rejected = ex["rejected"].strip()
        sft_rows.append({"text": f"{prompt}{chosen}"})
        rm_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        grpo_rows.append({"prompt": prompt})
    sft_dataset = Dataset.from_list(sft_rows)
    rm_dataset = Dataset.from_list(rm_rows)
    grpo_dataset = Dataset.from_list(grpo_rows)
    print("Use built-in demo samples only.")
    print(f"SFT dataset size: {len(sft_dataset)}")
    print(f"RM dataset size: {len(rm_dataset)}")
    print(f"GRPO dataset size: {len(grpo_dataset)}")
    print("First RM sample:", rm_dataset[0])
    return sft_dataset, rm_dataset, grpo_dataset


def train_sft_lora(args, sft_dataset):
    print("\n===== Stage 1: Train SFT LoRA =====")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **get_model_kwargs())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    disable_thinking_mode(model=model, tokenizer=tokenizer)

    model = get_peft_model(model, build_lora_config(TaskType.CAUSAL_LM))
    print_trainable_stats(model, stage_name="SFT")

    sft_args = SFTConfig(
        output_dir=args.sft_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.sft_learning_rate,
        max_steps=args.sft_max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=False,
        bf16=should_use_bf16(),
        dataloader_pin_memory=False,
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
    return final_sft_adapter_dir


def build_reward_text(prompt, response):
    return f"Instruction:\n{prompt.strip()}\n\nResponse:\n{response.strip()}"


def preprocess_rm_sample(example, tokenizer, max_length):
    chosen_tokens = tokenizer(build_reward_text(example["prompt"], example["chosen"]), truncation=True, max_length=max_length)
    rejected_tokens = tokenizer(
        build_reward_text(example["prompt"], example["rejected"]), truncation=True, max_length=max_length
    )
    return {
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
    }


@dataclass
class PairwiseRewardDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids = [f["chosen_input_ids"] for f in features]
        chosen_attention_mask = [f["chosen_attention_mask"] for f in features]
        rejected_input_ids = [f["rejected_input_ids"] for f in features]
        rejected_attention_mask = [f["rejected_attention_mask"] for f in features]

        chosen_batch = self.tokenizer.pad(
            {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask}, return_tensors="pt"
        )
        rejected_batch = self.tokenizer.pad(
            {"input_ids": rejected_input_ids, "attention_mask": rejected_attention_mask}, return_tensors="pt"
        )
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }


class PairwiseRewardTrainer(Trainer):
    def __init__(self, *args, rm_margin=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rm_margin = rm_margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_scores = model(
            input_ids=inputs["chosen_input_ids"], attention_mask=inputs["chosen_attention_mask"]
        ).logits.view(-1)
        rejected_scores = model(
            input_ids=inputs["rejected_input_ids"], attention_mask=inputs["rejected_attention_mask"]
        ).logits.view(-1)
        loss = -F.logsigmoid(chosen_scores - rejected_scores - self.rm_margin).mean()
        if return_outputs:
            return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
        return loss


def train_rm_lora(args, rm_dataset):
    print("\n===== Stage 2: Train Reward Model LoRA =====")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if should_use_bf16() else torch.float32,
    )
    rm_base.config.pad_token_id = tokenizer.pad_token_id
    rm_model = get_peft_model(rm_base, build_lora_config(TaskType.SEQ_CLS))
    print_trainable_stats(rm_model, stage_name="RM")

    train_ds = rm_dataset.map(
        lambda x: preprocess_rm_sample(x, tokenizer, args.max_length),
        remove_columns=rm_dataset.column_names,
    )
    collator = PairwiseRewardDataCollator(tokenizer=tokenizer)
    rm_args = TrainingArguments(
        output_dir=args.rm_output_dir,
        learning_rate=args.rm_learning_rate,
        max_steps=args.rm_max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=should_use_bf16(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = PairwiseRewardTrainer(
        model=rm_model,
        args=rm_args,
        train_dataset=train_ds,
        data_collator=collator,
        rm_margin=args.rm_margin,
    )
    trainer.train()
    final_rm_adapter_dir = f"{args.rm_output_dir}/final-lora"
    trainer.model.save_pretrained(final_rm_adapter_dir)
    tokenizer.save_pretrained(args.rm_output_dir)
    print(f"RM adapter saved to: {final_rm_adapter_dir}")
    return final_rm_adapter_dir


def load_rm_for_scoring(args, rm_adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(args.rm_output_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        # 评分模型不要用 auto/offload，避免 peft+accelerate 的 meta device 错误
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if should_use_bf16() else torch.float32,
        low_cpu_mem_usage=False,
    )
    rm_base.config.pad_token_id = tokenizer.pad_token_id
    rm_model = PeftModel.from_pretrained(rm_base, rm_adapter_path, adapter_name="rm", is_trainable=False)
    rm_model.set_adapter("rm")
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rm_model.to(target_device)
    rm_model.eval()
    for _, param in rm_model.named_parameters():
        param.requires_grad = False
    return rm_model, tokenizer


def freeze_for_adapter(peft_model, train_adapter_name):
    for _, param in peft_model.named_parameters():
        param.requires_grad = False
    for name, param in peft_model.named_parameters():
        if f".{train_adapter_name}." in name and "lora_" in name:
            param.requires_grad = True


def _extract_text_from_completion(completion):
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    return str(completion).strip()


def build_rm_reward_func(rm_model, rm_tokenizer, max_length, clip_low, clip_high):
    rm_device = next(rm_model.parameters()).device

    def rm_reward_func(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            response_text = _extract_text_from_completion(completion)
            rm_inputs = rm_tokenizer(
                build_reward_text(prompt, response_text),
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            rm_inputs = {k: v.to(rm_device) for k, v in rm_inputs.items()}
            with torch.no_grad():
                score = rm_model(**rm_inputs).logits[:, 0].detach().float().cpu().item()
            score = max(clip_low, min(clip_high, float(score)))
            rewards.append(score)
        return rewards

    return rm_reward_func


def build_grpo_config_compat(args):
    # 不同 trl 版本的 GRPOConfig 字段差异较大，这里做兼容映射
    base_kwargs = {
        "output_dir": args.grpo_output_dir,
        "learning_rate": args.grpo_learning_rate,
        "max_steps": args.grpo_max_steps,
        "num_generations": args.num_generations,
        "beta": args.beta,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "bf16": should_use_bf16(),
        "fp16": False,
        "remove_unused_columns": False,
    }
    sig = inspect.signature(GRPOConfig.__init__)
    params = sig.parameters
    compat_kwargs = {k: v for k, v in base_kwargs.items() if k in params}

    # prompt 长度参数
    if "max_prompt_length" in params:
        compat_kwargs["max_prompt_length"] = args.max_prompt_length
    elif "prompt_max_length" in params:
        compat_kwargs["prompt_max_length"] = args.max_prompt_length
    elif "max_length" in params and "max_length" not in compat_kwargs:
        compat_kwargs["max_length"] = args.max_length

    # completion 长度参数
    if "max_completion_length" in params:
        compat_kwargs["max_completion_length"] = args.max_completion_length
    elif "response_length" in params:
        compat_kwargs["response_length"] = args.max_completion_length
    elif "max_new_tokens" in params:
        compat_kwargs["max_new_tokens"] = args.max_completion_length

    return GRPOConfig(**compat_kwargs)


def train_grpo_lora(args, grpo_dataset, sft_adapter_path, rm_adapter_path):
    print("\n===== Stage 3: Train GRPO LoRA =====")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **get_model_kwargs())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(base_model, sft_adapter_path, adapter_name="sft", is_trainable=False)
    # 一些 trl 版本在 GRPOTrainer 内部强依赖 "default" 适配器名
    model.add_adapter("default", build_lora_config(TaskType.CAUSAL_LM))
    model.set_adapter("default")
    freeze_for_adapter(model, train_adapter_name="default")
    disable_thinking_mode(model=model, tokenizer=tokenizer)
    print_trainable_stats(model, stage_name="GRPO")

    instruction_prefix = "你是电商客服助手，请给出礼貌、清晰、可执行的答复。\n问题："
    train_rows = []
    for sample in grpo_dataset:
        train_rows.append({"prompt": f"{instruction_prefix}{sample['prompt'].strip()}"})
    train_dataset = Dataset.from_list(train_rows)

    rm_model, rm_tokenizer = load_rm_for_scoring(args, rm_adapter_path)
    rm_reward_func = build_rm_reward_func(
        rm_model=rm_model,
        rm_tokenizer=rm_tokenizer,
        max_length=args.max_length,
        clip_low=args.reward_clip_low,
        clip_high=args.reward_clip_high,
    )

    grpo_args = build_grpo_config_compat(args)
    trainer_kwargs = {
        "model": model,
        "reward_funcs": [rm_reward_func],
        "args": grpo_args,
        "train_dataset": train_dataset,
    }
    grpo_signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in grpo_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in grpo_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    final_grpo_adapter_dir = f"{args.grpo_output_dir}/final-lora"
    trainer.model.save_pretrained(final_grpo_adapter_dir)
    tokenizer.save_pretrained(args.grpo_output_dir)
    print(f"GRPO adapter saved to: {final_grpo_adapter_dir}")


def main():
    args = parse_args()
    print(f"bf16 enabled: {should_use_bf16()}")
    sft_dataset, rm_dataset, grpo_dataset = build_demo_datasets()
    sft_adapter_path = train_sft_lora(args, sft_dataset)
    rm_adapter_path = train_rm_lora(args, rm_dataset)
    train_grpo_lora(args, grpo_dataset, sft_adapter_path, rm_adapter_path)
    print("\nPipeline done: SFT LoRA + RM LoRA + GRPO LoRA.")


if __name__ == "__main__":
    main()
