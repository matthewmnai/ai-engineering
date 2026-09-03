# -*- coding: utf-8 -*-
"""
Qwen2.5-7B SFT微调（Alpaca数据集）
课程：LLM模型蒸馏与微调实操
功能：使用Unsloth框架对Qwen2.5-7B进行Alpaca格式的监督微调
环境：AutoDL GPU实例，建议 A100/3090
"""

# ========================================
# Step 1: 模型加载与4bit量化
# ========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

# 确保项目根在 sys.path，使 paths.py 能被导入
import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from environment.paths import get_model_dir, get_dataset_dir

# 定义输出目录（当前文件目录../outputs）
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "sft_train")
__CHECKPOINT_DIR = os.path.join(_OUTPUT_DIR, "checkpoint")
__LORA_DIR = os.path.join(_OUTPUT_DIR, "lora_model")


from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # 最大序列长度
dtype = None  # 自动检测，Tesla T4用Float16，Ampere+用Bfloat16
load_in_4bit = True  # 4bit量化减少显存

# 加载预训练模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=get_model_dir("Qwen2.5-7B-Instruct", "pretrained"),
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ========================================
# Step 2: LoRA适配器配置
# ========================================

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 减少30%显存
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# ========================================
# Step 3: Alpaca数据集准备
# ========================================

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

from datasets import load_dataset
dataset = load_dataset(get_dataset_dir("AI-ModelScope/alpaca-cleaned", "public"), split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)


# ========================================
# Step 4: SFTTrainer训练
# ========================================

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=10,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=__CHECKPOINT_DIR,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# 显示GPU信息
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 开始训练
trainer_stats = trainer.train()

# 训练统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")


# ========================================
# Step 5: 模型推理验证
# ========================================

FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [alpaca_prompt.format(
        "Continue the fibonnaci sequence.",
        "1, 1, 2, 3, 5, 8",
        "",
    )],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.batch_decode(outputs))

# 流式推理
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

inputs = tokenizer(
    [alpaca_prompt.format(
        "What is a famous tall tower in Paris?",
        "",
        "",
    )],
    return_tensors="pt"
).to("cuda")

_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


# ========================================
# Step 6: 模型保存
# ========================================
# 保存LoRA参数
os.makedirs(__LORA_DIR, exist_ok=True)
model.save_pretrained(__LORA_DIR)
tokenizer.save_pretrained(__LORA_DIR)

# 加载并验证保存的模型（路径必须和保存时一致，否则 unsloth 会当成 HF 仓库 ID 查找）

if os.path.isdir(__LORA_DIR):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=__LORA_DIR,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
        [alpaca_prompt.format("What is a famous tall tower in Paris?", "", "")],
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
else:
    print(f"⚠️  LoRA 目录不存在，跳过加载验证: {_lora_dir}")


# ========================================
# 模型导出选项（按需取消注释）
# ========================================

# 保存为16bit浮点格式
# model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# 保存为4bit格式
# model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")

# 仅保存LoRA适配器
# model.save_pretrained_merged("model", tokenizer, save_method="lora")

# 保存为GGUF格式（用于llama.cpp / Ollama）
# model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
