# -*- coding: utf-8 -*-
"""
Qwen3.5-0.8B 中文医疗模型SFT微调 (GPU版)
课程: 高质量微调数据工程与评估
环境: 需要GPU，建议4GB以上显存
依赖: pip install unsloth transformers trl datasets pandas
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

# 修复 AutoDL 等环境下 OMP_NUM_THREADS 未正确设置的问题
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "4"

import json
import glob
import torch
import pandas as pd
from datasets import Dataset

# ========================================
# 配置
# ========================================

_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "Qwen", "Qwen3___5-0___8B")
if os.path.exists(_LOCAL_MODEL_DIR):
    MODEL_NAME = _LOCAL_MODEL_DIR
else:
    MODEL_NAME = "Qwen/Qwen3.5-0.8B"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "【数据集】中文医疗数据")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")

OUTPUT_DIR = "outputs_medical"
LORA_SAVE_DIR = "lora_model_medical"

MAX_SEQ_LENGTH = 2048
MAX_STEPS = 100
BATCH_SIZE = 2
GRAD_ACCUM = 4
LORA_R = 16
LR = 2e-4

MEDICAL_PROMPT = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""


# ========================================
# 数据准备
# ========================================

def read_csv_with_encoding(file_path):
    """尝试使用不同编码读取CSV，gb18030是GBK超集优先尝试"""
    import io
    for encoding in ['gb18030', 'utf-8', 'gbk', 'gb2312']:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(file_path, 'r', encoding='gb18030', errors='replace') as f:
        return pd.read_csv(io.StringIO(f.read()))


def load_from_jsonl(jsonl_path):
    """从JSONL文件加载数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  从JSONL加载 {len(data)} 条数据: {jsonl_path}")
    return data


def load_from_csv(data_dir, max_samples=2000):
    """从原始CSV文件加载数据"""
    data = []
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'Oncology_肿瘤科': '肿瘤科',
        'OAGD_妇产科': '妇产科',
        'Andriatria_男科': '男科'
    }

    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            continue

        print(f"  处理{dept_name}数据...")
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            try:
                df = read_csv_with_encoding(file_path)
                for _, row in df.iterrows():
                    question = str(row.get('question', row.get('ask', ''))).strip()
                    answer = str(row.get('answer', row.get('response', ''))).strip()

                    if not question or not answer or question == 'nan' or answer == 'nan':
                        continue
                    if len(question) < 5 or len(answer) < 10:
                        continue

                    data.append({
                        "instruction": "请回答以下医疗相关问题",
                        "input": question,
                        "output": answer,
                    })
                    if len(data) >= max_samples:
                        break
            except Exception as e:
                print(f"    处理文件出错: {e}")
                continue
            if len(data) >= max_samples:
                break
        if len(data) >= max_samples:
            break

    print(f"  从CSV加载 {len(data)} 条数据")
    return data


def prepare_training_data():
    """准备训练数据，优先使用processed_data下的JSONL，否则回退到CSV"""
    jsonl_pattern = os.path.join(PROCESSED_DATA_DIR, "*.jsonl")
    candidates = [
        os.path.join(PROCESSED_DATA_DIR, "medical_alpaca_train.jsonl"),
        os.path.join(PROCESSED_DATA_DIR, "medical_alpaca_sampled.jsonl"),
        os.path.join(PROCESSED_DATA_DIR, "medical_alpaca_full.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_from_jsonl(path)
    for path in sorted(glob.glob(jsonl_pattern)):
        return load_from_jsonl(path)
    print("  未找到processed_data下的JSONL，将从原始CSV加载")
    return load_from_csv(DATA_DIR, max_samples=2000)


def make_dataset(data_list, tokenizer):
    """将数据列表转换为训练Dataset，每条末尾添加EOS"""
    eos_token = tokenizer.eos_token or ""

    def formatting_func(examples):
        texts = []
        for inp, out in zip(examples["input"], examples["output"]):
            text = MEDICAL_PROMPT.format(inp, out) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(data_list)
    dataset = dataset.map(formatting_func, batched=True)
    return dataset


# ========================================
# 模型加载与LoRA
# ========================================

def load_model():
    """使用Unsloth加载模型，4bit量化"""
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def setup_lora(model):
    """配置LoRA适配器，使用unsloth梯度检查点"""
    from unsloth import FastLanguageModel
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model


# ========================================
# 训练
# ========================================

def train_sft(model, tokenizer, dataset):
    """使用SFTTrainer训练，SFTConfig优先，回退到TrainingArguments"""
    from unsloth import is_bfloat16_supported
    bf16_val = is_bfloat16_supported()
    fp16_val = not bf16_val

    train_kwargs = dict(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        fp16=fp16_val,
        bf16=bf16_val,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    )

    from trl import SFTTrainer
    try:
        from trl import SFTConfig
        sft_config = SFTConfig(
            **train_kwargs,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            dataset_num_proc=2,
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )
    except ImportError:
        from transformers import TrainingArguments
        training_args = TrainingArguments(**train_kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"  GPU: {gpu_stats.name}, 总显存: {round(gpu_stats.total_memory / 1024 / 1024 / 1024, 1)} GB")
    print(f"  训练前已占用显存: {start_mem} GB")

    trainer.train()

    peak_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"  训练后峰值显存: {peak_mem} GB")


# ========================================
# 推理
# ========================================

def generate_response(model, tokenizer, question, max_new_tokens=256):
    """推理前调用for_inference，张量置于cuda"""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    prompt = MEDICAL_PROMPT.format(question, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### 回答：" in response:
        answer = response.split("### 回答：")[-1].strip()
    elif "### 回答:" in response:
        answer = response.split("### 回答:")[-1].strip()
    else:
        answer = response
    return answer


# ========================================
# 主入口
# ========================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 未检测到GPU，本脚本仅支持GPU运行")
        exit(1)

    print(f"\n{'='*60}")
    print("  Qwen3.5-0.8B 中文医疗模型SFT微调 (GPU版)")
    print(f"  模型: {MODEL_NAME}")
    print(f"  训练步数: {MAX_STEPS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"{'='*60}\n")

    # 1. 加载模型
    print("--- 1/7 加载模型 ---")
    model, tokenizer = load_model()

    # 2. 配置LoRA
    print("\n--- 2/7 配置LoRA ---")
    model = setup_lora(model)

    # 3. 准备数据并格式化为带EOS的文本
    print("\n--- 3/7 准备训练数据 ---")
    data_list = prepare_training_data()
    dataset = make_dataset(data_list, tokenizer)
    print(f"  训练样本数: {len(dataset)}")

    # 4. SFT训练
    print("\n--- 4/7 开始SFT训练 ---")
    train_sft(model, tokenizer, dataset)

    # 5. 打印GPU显存统计
    print("\n--- 5/7 GPU显存统计 ---")
    if torch.cuda.is_available():
        allocated = round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 3)
        reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"  当前分配: {allocated} GB, 峰值保留: {reserved} GB")

    # 6. 测试推理
    print("\n--- 6/7 测试推理 ---")
    test_questions = [
        "我最近总是感觉头晕，应该怎么办？",
        "感冒发烧应该吃什么药？",
        "高血压患者需要注意什么？",
    ]
    for q in test_questions:
        ans = generate_response(model, tokenizer, q)
        print(f"  Q: {q}")
        print(f"  A: {ans[:300]}...")
        print()

    # 7. 保存LoRA模型
    print("--- 7/7 保存LoRA模型 ---")
    os.makedirs(LORA_SAVE_DIR, exist_ok=True)
    model.save_pretrained(LORA_SAVE_DIR)
    tokenizer.save_pretrained(LORA_SAVE_DIR)
    print(f"  已保存至: {LORA_SAVE_DIR}")

    print("\n微调完成")
