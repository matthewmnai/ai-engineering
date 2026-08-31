# -*- coding: utf-8 -*-
"""
Qwen2.5-7B 中文医疗模型SFT微调
课程：高质量微调数据工程与评估
功能：使用中文医疗数据对Qwen2.5-7B进行垂直领域微调（完整实操代码）
环境：AutoDL GPU实例
数据来源：本课程的医疗数据清洗Pipeline产出的数据
"""

# ========================================
# Step 1: 模型加载
# ========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# ========================================
# Step 2: LoRA配置
# ========================================

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
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


# ========================================
# Step 3: 医疗数据准备
# ========================================

import os
import pandas as pd
from datasets import Dataset

# 医疗对话提示模板
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

EOS_TOKEN = tokenizer.eos_token


def read_csv_with_encoding(file_path):
    """尝试使用不同编码读取CSV文件"""
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何编码读取文件: {file_path}")


def load_medical_data(data_dir):
    """加载医疗对话数据"""
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
            print(f"目录不存在: {dept_path}")
            continue

        print(f"\n处理{dept_name}数据...")

        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            print(f"正在处理文件: {csv_file}")

            try:
                df = read_csv_with_encoding(file_path)
                print(f"文件 {csv_file} 的列名: {df.columns.tolist()}")

                for _, row in df.iterrows():
                    try:
                        question = None
                        answer = None

                        if 'question' in row:
                            question = str(row['question']).strip()
                        elif 'ask' in row:
                            question = str(row['ask']).strip()

                        if 'answer' in row:
                            answer = str(row['answer']).strip()
                        elif 'response' in row:
                            answer = str(row['response']).strip()

                        if not question or not answer:
                            continue
                        if len(question) > 200 or len(answer) > 200:
                            continue

                        data.append({
                            "instruction": "请回答以下医疗相关问题",
                            "input": question,
                            "output": answer
                        })

                    except Exception as e:
                        print(f"处理数据行时出错: {e}")
                        continue

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

    if not data:
        raise ValueError("没有成功处理任何数据!")

    print(f"\n成功处理 {len(data)} 条数据")
    return Dataset.from_list(data)


def formatting_prompts_func(examples):
    """格式化提示"""
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        text = medical_prompt.format(input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


# 加载医疗数据
dataset = load_medical_data("Data_数据")
dataset = dataset.map(formatting_prompts_func, batched=True)


# ========================================
# Step 4: 模型训练
# ========================================

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=-1,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
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

# 显存信息
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
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ========================================
# Step 5: 模型推理
# ========================================

def generate_medical_response(question):
    """生成医疗回答"""
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        [medical_prompt.format(question, "")],
        return_tensors="pt"
    ).to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )


test_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？"
]

for question in test_questions:
    print("\n" + "=" * 50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(question)


# ========================================
# Step 6: 模型保存与加载
# ========================================

# 保存LoRA参数
model.save_pretrained("lora_model_medical")
tokenizer.save_pretrained("lora_model_medical")

# 加载微调后的模型
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model_medical",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question)
