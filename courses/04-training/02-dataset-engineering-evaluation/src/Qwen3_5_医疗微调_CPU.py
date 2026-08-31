# -*- coding: utf-8 -*-
"""
Qwen3.5-0.8B 中文医疗模型SFT微调 (CPU版)
课程: 高质量微调数据工程与评估
环境: 无需GPU，约需3.2GB内存，训练10步约需1-3分钟
依赖: pip install "transformers>=5.2.0" peft trl datasets pandas
"""

import os
import json
import glob
import torch
import pandas as pd
from datasets import Dataset

# ========================================
# 配置
# ========================================

MEDICAL_PROMPT = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "【数据集】中文医疗数据")
_PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "Qwen", "Qwen3___5-0___8B")
if os.path.exists(_LOCAL_MODEL_DIR):
    MODEL_NAME = _LOCAL_MODEL_DIR
else:
    MODEL_NAME = "Qwen/Qwen3.5-0.8B"


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
                    data.append({"input": question, "output": answer})
                    if len(data) >= max_samples:
                        break
            except Exception as e:
                continue
            if len(data) >= max_samples:
                break
        if len(data) >= max_samples:
            break
    return data


def prepare_training_data():
    """准备训练数据，优先检查processed_data/*.jsonl，否则回退到CSV"""
    candidates = [
        os.path.join(_PROCESSED_DIR, "medical_alpaca_train.jsonl"),
        os.path.join(_PROCESSED_DIR, "medical_alpaca_sampled.jsonl"),
        os.path.join(_PROCESSED_DIR, "medical_alpaca_full.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_from_jsonl(path)
    for path in sorted(glob.glob(os.path.join(_PROCESSED_DIR, "*.jsonl"))):
        return load_from_jsonl(path)
    return load_from_csv(_DATA_DIR, max_samples=2000)


if __name__ == "__main__":
    import transformers
    from packaging.version import Version
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    # Qwen3.5 的 model_type=qwen3_5 从 transformers 5.2.0 才注册
    if Version(transformers.__version__) < Version("5.2.0"):
        raise RuntimeError(
            f"Qwen3.5 需要 transformers>=5.2.0，当前为 {transformers.__version__}。"
            '请执行: python -m pip install -U "transformers>=5.2.0"'
        )

    # 1. 加载模型（文本专用 CausalLM，不加载视觉塔）
    print("CPU模式: float32加载 (约需3.2GB内存)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 配置LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 准备数据并格式化为带EOS的文本
    data_list = prepare_training_data()
    eos_token = tokenizer.eos_token or ""

    def formatting_func(examples):
        texts = []
        for inp, out in zip(examples["input"], examples["output"]):
            text = MEDICAL_PROMPT.format(inp, out) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(data_list)
    dataset = dataset.map(formatting_func, batched=True)
    print(f"训练数据: {len(dataset)} 条")

    # 4. SFT训练
    train_kwargs = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        use_cpu=True,
        output_dir="outputs_medical",
        report_to="none",
        logging_steps=1,
    )
    try:
        from trl import SFTConfig
        sft_config = SFTConfig(
            **train_kwargs,
            max_seq_length=512,
            dataset_text_field="text",
            dataset_num_proc=1,
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )
    except (ImportError, TypeError):
        from transformers import TrainingArguments
        training_args = TrainingArguments(**train_kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )

    trainer.train()

    # 5. 测试推理
    test_questions = [
        "我最近总是感觉头晕，应该怎么办？",
        "感冒发烧应该吃什么药？",
        "高血压患者需要注意什么？",
    ]
    model.eval()
    for q in test_questions:
        prompt = MEDICAL_PROMPT.format(q, "")
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### 回答：" in response:
            answer = response.split("### 回答：")[-1].strip()
        elif "### 回答:" in response:
            answer = response.split("### 回答:")[-1].strip()
        else:
            answer = response
        print(f"Q: {q}")
        print(f"A: {answer[:200]}")
        print()

    # 6. 保存LoRA模型
    save_dir = "lora_model_medical"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"LoRA模型已保存: {save_dir}")
