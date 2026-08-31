# -*- coding: utf-8 -*-
"""
Qwen3.5-0.8B 中文医疗模型SFT微调
课程: 高质量微调数据工程与评估
功能: 使用清洗后的中文医疗数据对 Qwen3.5-0.8B 进行垂直领域微调
支持: GPU (Unsloth加速) / CPU (transformers + peft)
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"
import json
import torch
import pandas as pd
from datasets import Dataset

# ========================================
# 环境检测与配置
# ========================================

HAS_GPU = torch.cuda.is_available()

# 优先使用本地已下载的模型，否则从HuggingFace下载
_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "Qwen", "Qwen3___5-0___8B")
if os.path.exists(_LOCAL_MODEL_DIR):
    MODEL_NAME = _LOCAL_MODEL_DIR
else:
    MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "【数据集】中文医疗数据")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
USE_CLEANED_DATA = True

OUTPUT_DIR = "outputs_medical"
LORA_SAVE_DIR = "lora_model_medical"

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


def load_from_jsonl(jsonl_path):
    """从清洗后的JSONL文件加载数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  从JSONL加载 {len(data)} 条数据")
    return data


def prepare_training_data():
    """准备训练数据，优先使用清洗后的数据"""
    if USE_CLEANED_DATA:
        candidates = [
            os.path.join(PROCESSED_DATA_DIR, "medical_alpaca_train.jsonl"),
            os.path.join(PROCESSED_DATA_DIR, "medical_alpaca_sampled.jsonl"),
            os.path.join(PROCESSED_DATA_DIR, "medical_alpaca_full.jsonl"),
        ]
        for path in candidates:
            if os.path.exists(path):
                print(f"  使用清洗后的数据: {path}")
                return load_from_jsonl(path)
        print("  未找到清洗后的数据，将从原始CSV加载")

    return load_from_csv(DATA_DIR, max_samples=2000)


def make_dataset(data_list, tokenizer):
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
# 模型加载（GPU/CPU兼容）
# ========================================

def load_model(model_name=None, use_gpu=None):
    """
    加载模型和tokenizer
    GPU: 使用Unsloth + 4bit量化
    CPU: 使用transformers + float32
    """
    if model_name is None:
        model_name = MODEL_NAME
    if use_gpu is None:
        use_gpu = HAS_GPU

    if use_gpu:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  CPU模式: float32加载 (约需3.2GB内存)")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def setup_lora(model, use_gpu=None, lora_r=None):
    """
    配置LoRA适配器
    GPU: 使用Unsloth的优化LoRA
    CPU: 使用peft标准LoRA
    """
    if use_gpu is None:
        use_gpu = HAS_GPU
    if lora_r is None:
        lora_r = 16 if use_gpu else 8

    if use_gpu:
        from unsloth import FastLanguageModel
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
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
    else:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


# ========================================
# 训练
# ========================================

def train_sft(model, tokenizer, dataset, output_dir=None, max_steps=None, use_gpu=None):
    """
    执行SFT训练
    返回训练统计信息(dict)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if use_gpu is None:
        use_gpu = HAS_GPU
    if max_steps is None:
        max_steps = 100 if use_gpu else 10

    from trl import SFTTrainer

    if use_gpu:
        from unsloth import is_bfloat16_supported
        fp16_val = not is_bfloat16_supported()
        bf16_val = is_bfloat16_supported()
        optim = "adamw_8bit"
        batch_size = 2
        grad_accum = 4
        max_seq = 2048
    else:
        fp16_val = False
        bf16_val = False
        optim = "adamw_torch"
        batch_size = 1
        grad_accum = 2
        max_seq = 512

    # 兼容新版trl(>=0.12 SFTConfig) 和 旧版trl(TrainingArguments)
    train_kwargs = dict(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=2,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=fp16_val,
        bf16=bf16_val,
        logging_steps=1,
        optim=optim,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
    )
    if not use_gpu:
        train_kwargs["use_cpu"] = True

    try:
        from trl import SFTConfig
        sft_config = SFTConfig(
            **train_kwargs,
            max_seq_length=max_seq,
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
    except ImportError:
        from transformers import TrainingArguments
        training_args = TrainingArguments(**train_kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )

    if use_gpu:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"  GPU = {gpu_stats.name}, 显存 = {round(gpu_stats.total_memory / 1024 / 1024 / 1024, 1)} GB")
    else:
        print(f"  CPU训练, max_steps={max_steps}, batch={batch_size}x{grad_accum}")

    trainer_stats = trainer.train()

    train_time = round(trainer_stats.metrics['train_runtime'], 2)
    final_loss = round(trainer_stats.metrics.get('train_loss', 0), 4)

    if use_gpu:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"  训练完成: {train_time}秒, Loss={final_loss}, 峰值显存={used_memory}GB")
    else:
        print(f"  训练完成: {train_time}秒, Loss={final_loss}")

    loss_history = []
    for log in trainer.state.log_history:
        if 'loss' in log:
            loss_history.append({"step": log.get('step', 0), "loss": round(log['loss'], 4)})

    return {
        "train_time_seconds": train_time,
        "final_loss": final_loss,
        "loss_history": loss_history,
    }


# ========================================
# 推理
# ========================================

def generate_response(model, tokenizer, question, use_gpu=None, max_new_tokens=128):
    """
    用模型生成医疗回答
    返回: 回答文本(str)
    """
    if use_gpu is None:
        use_gpu = HAS_GPU

    prompt = MEDICAL_PROMPT.format(question, "")

    if use_gpu:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    else:
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt")

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
# 完整微调流程
# ========================================

def run_finetune(model_name=None, data_path=None, output_dir=None,
                 save_dir=None, max_steps=None, use_gpu=None,
                 test_questions=None):
    """
    完整的微调流程

    参数:
        model_name: 模型名称或路径
        data_path: JSONL训练数据路径 (None则自动查找)
        output_dir: 训练输出目录
        save_dir: LoRA模型保存目录
        max_steps: 训练步数 (GPU默认100, CPU默认10)
        use_gpu: 是否使用GPU (None则自动检测)
        test_questions: 测试问题列表

    返回:
        (model, tokenizer, stats_dict)
    """
    if model_name is None:
        model_name = MODEL_NAME
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if save_dir is None:
        save_dir = LORA_SAVE_DIR
    if use_gpu is None:
        use_gpu = HAS_GPU
    if max_steps is None:
        max_steps = 100 if use_gpu else 10
    if test_questions is None:
        test_questions = [
            "我最近总是感觉头晕，应该怎么办？",
            "感冒发烧应该吃什么药？",
            "高血压患者需要注意什么？",
        ]

    mode_str = "GPU (Unsloth)" if use_gpu else "CPU (transformers+peft)"
    print(f"\n{'='*60}")
    print(f"  Qwen3.5-0.8B 医疗微调")
    print(f"  运行模式: {mode_str}")
    print(f"  训练步数: {max_steps}")
    print(f"{'='*60}")

    # 1. 加载模型
    print(f"\n--- 1/6 加载模型: {model_name} ---")
    model, tokenizer = load_model(model_name, use_gpu=use_gpu)

    # 2. 配置LoRA
    print(f"\n--- 2/6 配置LoRA ---")
    model = setup_lora(model, use_gpu=use_gpu)

    # 3. 微调前推理（LoRA权重为零，等价于基座模型）
    print(f"\n--- 3/6 微调前推理 (基座模型效果) ---")
    before_answers = []
    for q in test_questions:
        answer = generate_response(model, tokenizer, q, use_gpu, max_new_tokens=128)
        before_answers.append(answer)
        print(f"  Q: {q}")
        print(f"  A: {answer[:200]}")
        print()

    # 4. 准备数据 & 训练
    print(f"--- 4/6 准备训练数据 ---")
    if data_path and os.path.exists(data_path):
        data_list = load_from_jsonl(data_path)
    else:
        data_list = prepare_training_data()

    dataset = make_dataset(data_list, tokenizer)
    print(f"  训练数据: {len(dataset)} 条")

    print(f"\n--- 5/6 开始SFT训练 ---")
    stats = train_sft(model, tokenizer, dataset, output_dir, max_steps, use_gpu)

    # 5. 微调后推理
    print(f"\n--- 6/6 微调后推理 ---")
    after_answers = []
    for q in test_questions:
        answer = generate_response(model, tokenizer, q, use_gpu, max_new_tokens=128)
        after_answers.append(answer)
        print(f"  Q: {q}")
        print(f"  A: {answer[:200]}")
        print()

    stats["before_answers"] = before_answers
    stats["after_answers"] = after_answers
    stats["test_questions"] = test_questions

    # 6. 保存模型和推理结果
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  LoRA模型已保存: {save_dir}")

    # 保存推理结果供Step 5 BLEU评估使用
    eval_results = {
        "test_questions": test_questions,
        "before_answers": before_answers,
        "after_answers": after_answers,
        "train_stats": {
            "train_time_seconds": stats["train_time_seconds"],
            "final_loss": stats["final_loss"],
            "max_steps": max_steps,
            "mode": mode_str,
        }
    }
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    eval_path = os.path.join(PROCESSED_DATA_DIR, "eval_results.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"  推理结果已保存: {eval_path}")

    return model, tokenizer, stats


# ========================================
# 主入口
# ========================================

if __name__ == "__main__":
    model, tokenizer, stats = run_finetune()
    print("\n微调完成!")
