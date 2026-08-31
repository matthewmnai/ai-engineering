# -*- coding: utf-8 -*-
"""
清洗前后SFT微调快速对比实验 (GPU版)
课程: 高质量微调数据工程与评估
功能: 分别使用清洗前和清洗后的数据进行快速微调，对比Loss曲线和生成效果
环境: 需要GPU (Unsloth, 4bit量化)
依赖: pip install unsloth transformers trl datasets pandas
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"
import io
import re
import json
import hashlib
import pandas as pd
import torch
from datasets import Dataset

# ========================================
# 配置
# ========================================

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "Qwen", "Qwen3___5-0___8B")
if os.path.exists(_LOCAL_MODEL_DIR):
    MODEL_NAME = _LOCAL_MODEL_DIR

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "【数据集】中文医疗数据")

MAX_SEQ_LENGTH = 2048
MAX_STEPS = 30
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_SAMPLES = 300

OUTPUT_DIR_RAW = "outputs_raw_data"
OUTPUT_DIR_CLEAN = "outputs_clean_data"

MEDICAL_PROMPT = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

TEST_QUESTIONS = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？",
]


# ========================================
# 数据准备：两种数据版本
# ========================================

def read_csv_with_encoding(file_path):
    """尝试使用不同编码读取CSV，gb18030是GBK超集优先尝试"""
    for encoding in ['gb18030', 'utf-8', 'gbk', 'gb2312']:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(file_path, 'r', encoding='gb18030', errors='replace') as f:
        return pd.read_csv(io.StringIO(f.read()))


def load_raw_data(data_dir=None, max_samples=300):
    """版本A: 原始数据（未清洗），直接加载不做任何过滤"""
    if data_dir is None:
        data_dir = DATA_DIR
    data = []
    departments = {
        'IM_内科': '内科',
        'Andriatria_男科': '男科',
    }

    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            continue

        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            try:
                df = read_csv_with_encoding(file_path)
                for _, row in df.iterrows():
                    question = str(row.get('question', row.get('ask', '')))
                    answer = str(row.get('answer', row.get('response', '')))
                    data.append({
                        "instruction": "请回答以下医疗相关问题",
                        "input": question,
                        "output": answer,
                    })
                    if len(data) >= max_samples:
                        break
            except Exception:
                continue
            if len(data) >= max_samples:
                break
        if len(data) >= max_samples:
            break

    print(f"  [原始数据] 加载 {len(data)} 条（未清洗）")
    return data


def load_cleaned_data(data_dir=None, max_samples=300):
    """版本B: 清洗后的数据，过滤：去空值、去过短、过长、MD5去重"""
    if data_dir is None:
        data_dir = DATA_DIR
    data = []
    seen_hashes = set()

    departments = {
        'IM_内科': '内科',
        'Andriatria_男科': '男科',
    }

    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            continue

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
                    if len(question) > 500 or len(answer) > 2000:
                        continue

                    q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
                    if q_hash in seen_hashes:
                        continue
                    seen_hashes.add(q_hash)

                    question = re.sub(r'\s+', ' ', question)
                    answer = re.sub(r'\s+', ' ', answer)

                    data.append({
                        "instruction": "请回答以下医疗相关问题",
                        "input": question,
                        "output": answer,
                    })
                    if len(data) >= max_samples:
                        break
            except Exception:
                continue
            if len(data) >= max_samples:
                break
        if len(data) >= max_samples:
            break

    print(f"  [清洗数据] 加载 {len(data)} 条（已清洗）")
    return data


def prepare_dataset(data_list, tokenizer):
    """将数据列表转换为训练Dataset"""
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
# 模型训练 (Unsloth加速)
# ========================================

def train_model(dataset, output_dir, experiment_name="experiment"):
    """训练模型并返回统计信息，使用Unsloth + 4bit量化"""
    from unsloth import FastLanguageModel, is_bfloat16_supported

    print(f"\n{'='*60}")
    print(f"  开始训练: {experiment_name}")
    print(f"  数据量: {len(dataset)} 条, 模式: GPU (Unsloth)")
    print(f"  步数: {MAX_STEPS}")
    print(f"{'='*60}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    fp16_val = not is_bfloat16_supported()
    bf16_val = is_bfloat16_supported()

    train_kwargs = dict(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=2,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=fp16_val,
        bf16=bf16_val,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
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

    trainer_stats = trainer.train()

    stats = {
        "experiment": experiment_name,
        "data_size": len(dataset),
        "train_time_seconds": round(trainer_stats.metrics['train_runtime'], 2),
        "final_loss": round(trainer_stats.metrics.get('train_loss', 0), 4),
    }

    loss_history = []
    for log in trainer.state.log_history:
        if 'loss' in log:
            loss_history.append({"step": log.get('step', 0), "loss": round(log['loss'], 4)})
    stats["loss_history"] = loss_history

    # 推理测试
    print(f"\n--- {experiment_name}: 推理测试 ---")
    FastLanguageModel.for_inference(model)
    model.eval()

    generation_results = []
    for question in TEST_QUESTIONS:
        prompt = MEDICAL_PROMPT.format(question, "")
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256,
                temperature=0.7, top_p=0.9, use_cache=True, do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### 回答：" in response:
            answer = response.split("### 回答：")[-1].strip()
        elif "### 回答:" in response:
            answer = response.split("### 回答:")[-1].strip()
        else:
            answer = response

        generation_results.append({"question": question, "answer": answer[:500]})
        print(f"  Q: {question}")
        print(f"  A: {answer[:200]}")
        print()

    stats["generation_results"] = generation_results

    model.save_pretrained(os.path.join(output_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora_model"))

    del model, tokenizer, trainer
    torch.cuda.empty_cache()

    return stats


# ========================================
# 对比分析
# ========================================

def compare_results(raw_stats, clean_stats):
    """对比两次实验结果，打印格式化对比表"""
    print("\n" + "=" * 60)
    print("  对比结果: 清洗前 vs 清洗后")
    print("=" * 60)

    print(f"\n--- 训练统计 ---")
    print(f"  {'指标':<20} {'清洗前':<15} {'清洗后':<15}")
    print(f"  {'数据量':<20} {raw_stats['data_size']:<15} {clean_stats['data_size']:<15}")
    print(f"  {'训练时间(秒)':<20} {raw_stats['train_time_seconds']:<15} {clean_stats['train_time_seconds']:<15}")
    print(f"  {'最终Loss':<20} {raw_stats['final_loss']:<15} {clean_stats['final_loss']:<15}")

    print(f"\n--- Loss曲线对比 ---")
    raw_losses = raw_stats.get("loss_history", [])
    clean_losses = clean_stats.get("loss_history", [])

    max_len = max(len(raw_losses), len(clean_losses))
    step_interval = max(1, max_len // 8)
    for i in range(0, max_len, step_interval):
        raw_loss = raw_losses[i]["loss"] if i < len(raw_losses) else "-"
        clean_loss = clean_losses[i]["loss"] if i < len(clean_losses) else "-"
        step = raw_losses[i]["step"] if i < len(raw_losses) else clean_losses[i]["step"]
        print(f"  Step {step:3d}: 清洗前={raw_loss}  清洗后={clean_loss}")

    print(f"\n--- 生成质量对比 ---")
    raw_gens = raw_stats.get("generation_results", [])
    clean_gens = clean_stats.get("generation_results", [])

    for r, c in zip(raw_gens, clean_gens):
        print(f"\n  问题: {r['question']}")
        print(f"  清洗前: {r['answer'][:150]}")
        print(f"  清洗后: {c['answer'][:150]}")

    comparison = {
        "raw_data": raw_stats,
        "clean_data": clean_stats,
        "conclusion": {
            "loss_improvement": round(raw_stats['final_loss'] - clean_stats['final_loss'], 4),
            "time_diff_seconds": round(raw_stats['train_time_seconds'] - clean_stats['train_time_seconds'], 2),
        }
    }

    report_path = "comparison_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n  对比报告已保存: {report_path}")

    return comparison


# ========================================
# 主流程
# ========================================

def main():
    """完整对比实验: 清洗前 vs 清洗后"""
    if not torch.cuda.is_available():
        print("错误: 未检测到GPU，请使用 sft_quick_comparison_CPU.py")
        exit(1)

    data_dir = DATA_DIR

    print(f"\n{'='*60}")
    print(f"  清洗前后SFT微调对比实验 (GPU版)")
    print(f"  每组数据: {MAX_SAMPLES}条, 训练: {MAX_STEPS}步")
    print(f"{'='*60}")

    raw_data = load_raw_data(data_dir, max_samples=MAX_SAMPLES)
    clean_data = load_cleaned_data(data_dir, max_samples=MAX_SAMPLES)

    from unsloth import FastLanguageModel
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=True,
    )

    raw_dataset = prepare_dataset(raw_data, tokenizer)
    clean_dataset = prepare_dataset(clean_data, tokenizer)
    del tokenizer
    torch.cuda.empty_cache()

    raw_stats = train_model(raw_dataset, OUTPUT_DIR_RAW, "原始数据(未清洗)")
    clean_stats = train_model(clean_dataset, OUTPUT_DIR_CLEAN, "清洗后数据")
    compare_results(raw_stats, clean_stats)

    print(f"\n{'='*60}")
    print("  核心结论: 数据质量 > 数据数量")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
