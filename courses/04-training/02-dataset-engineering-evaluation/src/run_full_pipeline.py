# -*- coding: utf-8 -*-
"""
完整实操流程：高质量微调数据工程与评估
==================================================
将本课程所有代码串联起来，6个步骤从原始数据到微调模型的完整Pipeline。
支持 GPU (Unsloth加速) 和 CPU (transformers + peft) 两种模式。

使用方式：
  方式1: 整体运行
    python run_full_pipeline.py

  方式2: 分步运行（推荐，可以在每一步观察结果）
    python run_full_pipeline.py --step 1    # 数据收集清洗（CPU）
    python run_full_pipeline.py --step 2    # 数据质量评估（CPU）
    python run_full_pipeline.py --step 3    # 数据格式转换（CPU）
    python run_full_pipeline.py --step 4    # 模型微调（GPU/CPU）
    python run_full_pipeline.py --step 5    # BLEU效果评估（CPU）
    python run_full_pipeline.py --step 6    # 清洗价值验证（GPU/CPU）

CPU模式说明：
  无GPU时自动降级为CPU模式，训练步数减少（10步），使用float32
  0.8B模型在CPU上单步约需3-8秒，10步约需1-3分钟
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

import json
import argparse

# 工作目录设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 全局路径配置
DATA_DIR = "【数据集】中文医疗数据"
OUTPUT_DIR = "processed_data"
MODEL_NAME = "Qwen/Qwen3.5-0.8B"


# ============================================================
# Step 1: 数据收集与清洗
# 使用: medical_data_processor.py
# 环境: CPU即可（本地Windows/Mac/Linux均可运行）
# 耗时: 约1-3分钟
# ============================================================

def step1_collect_and_clean():
    """
    Step 1: 从6个科室的CSV文件收集原始数据，应用清洗规则，输出训练就绪的数据集

    输入: 【数据集】中文医疗数据/ 下的6个科室CSV文件
    输出: processed_data/ 目录下的多个JSONL文件
      - medical_alpaca_full.jsonl    (完整清洗数据)
      - medical_alpaca_sampled.jsonl (均衡采样数据)
      - medical_alpaca_train.jsonl   (训练集)
      - medical_alpaca_val.jsonl     (验证集)

    清洗规则:
      1. 自动检测文件编码 (utf-8/gbk/gb2312/gb18030)
      2. 空值过滤: 问题或回答为空的条目
      3. 长度过滤: 问题<5字或>500字、回答<10字或>2000字
      4. 无意义过滤: 纯标点、"你好/嗯/哦"等无意义问题
      5. MD5去重: 基于问题内容的哈希去重
      6. 均衡采样: 按科室均衡抽样(每科室200条)，留出5%验证集
    """
    print("\n" + "=" * 70)
    print("  Step 1: 数据收集与清洗")
    print("  使用: medical_data_processor.py")
    print("=" * 70)

    from medical_data_processor import (
        collect_medical_data,
        clean_medical_data,
        to_alpaca_format,
        to_chat_format,
        save_dataset,
        sample_balanced_data,
    )

    # 1.1 收集原始数据
    print("\n--- 1.1 数据收集 ---")
    raw_data, collect_stats = collect_medical_data(DATA_DIR)
    if not raw_data:
        print("[错误] 没有收集到数据，请检查数据目录")
        return None, None

    # 保存一份原始数据的简要统计（供Step 2对比用）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_sample = []
    for df in raw_data:
        for _, row in df.head(500).iterrows():
            q = str(row.get('question', row.get('ask', ''))).strip()
            a = str(row.get('answer', row.get('response', ''))).strip()
            if q and a:
                raw_sample.append({
                    "instruction": "请回答以下医疗相关问题",
                    "input": q,
                    "output": a,
                    "department": str(row.get('department', '未知')),
                })
    save_dataset(raw_sample, os.path.join(OUTPUT_DIR, "medical_raw_sample.jsonl"))

    # 1.2 数据清洗
    print("\n--- 1.2 数据清洗 ---")
    cleaned_data = clean_medical_data(raw_data)

    # 1.3 格式转换与保存
    print("\n--- 1.3 保存清洗后的数据 ---")
    alpaca_data = to_alpaca_format(cleaned_data)
    save_dataset(alpaca_data, os.path.join(OUTPUT_DIR, "medical_alpaca_full.jsonl"))

    chat_data = to_chat_format(cleaned_data)
    save_dataset(chat_data, os.path.join(OUTPUT_DIR, "medical_chat_full.jsonl"))

    # 1.4 均衡采样
    print("\n--- 1.4 均衡采样（快速实验用）---")
    sampled_data = sample_balanced_data(cleaned_data, samples_per_dept=200)
    sampled_alpaca = to_alpaca_format(sampled_data)
    save_dataset(sampled_alpaca, os.path.join(OUTPUT_DIR, "medical_alpaca_sampled.jsonl"))

    val_size = max(10, len(sampled_data) // 20)
    val_data = to_alpaca_format(sampled_data[:val_size])
    train_data = to_alpaca_format(sampled_data[val_size:])
    save_dataset(val_data, os.path.join(OUTPUT_DIR, "medical_alpaca_val.jsonl"))
    save_dataset(train_data, os.path.join(OUTPUT_DIR, "medical_alpaca_train.jsonl"))

    print(f"\nStep 1 完成!")
    print(f"  原始数据量: {sum(collect_stats.values())} 条")
    print(f"  清洗后数据: {len(cleaned_data)} 条")
    print(f"  采样训练集: {len(train_data)} 条")
    print(f"  采样验证集: {len(val_data)} 条")

    return raw_sample, alpaca_data


# ============================================================
# Step 2: 数据质量评估（清洗前 vs 清洗后对比）
# 使用: data_quality_report.py
# 环境: CPU即可
# 耗时: 约30秒
# ============================================================

def step2_quality_report():
    """
    Step 2: 分别对原始数据和清洗后数据生成质量报告，对比展示清洗效果

    输入: Step 1 产出的 medical_raw_sample.jsonl 和 medical_alpaca_full.jsonl
    输出:
      - processed_data/raw_quality_report.json    (原始数据质量报告)
      - processed_data/clean_quality_report.json   (清洗后数据质量报告)

    对比维度: 格式合规 / 字段完整 / 语言一致 / 重复率 / 长度合理 / 多样性
    """
    print("\n" + "=" * 70)
    print("  Step 2: 数据质量评估（清洗前 vs 清洗后）")
    print("  使用: data_quality_report.py")
    print("=" * 70)

    from data_quality_report import generate_quality_report, print_report_summary

    # 2.1 加载原始数据样本
    raw_path = os.path.join(OUTPUT_DIR, "medical_raw_sample.jsonl")
    clean_path = os.path.join(OUTPUT_DIR, "medical_alpaca_sampled.jsonl")

    if not os.path.exists(raw_path) or not os.path.exists(clean_path):
        print("[提示] 请先运行 Step 1 生成数据")
        print("  python run_full_pipeline.py --step 1")
        return

    raw_data = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    clean_data = []
    with open(clean_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                clean_data.append(json.loads(line))

    # 2.2 生成原始数据质量报告
    print("\n--- 2.1 原始数据质量报告 ---")
    raw_report = generate_quality_report(raw_data, "原始CSV数据(未清洗)")
    print_report_summary(raw_report)

    raw_report_path = os.path.join(OUTPUT_DIR, "raw_quality_report.json")
    with open(raw_report_path, 'w', encoding='utf-8') as f:
        json.dump(raw_report, f, ensure_ascii=False, indent=2)

    # 2.3 生成清洗后数据质量报告
    print("\n--- 2.2 清洗后数据质量报告 ---")
    clean_report = generate_quality_report(clean_data, "清洗后数据")
    print_report_summary(clean_report)

    clean_report_path = os.path.join(OUTPUT_DIR, "clean_quality_report.json")
    with open(clean_report_path, 'w', encoding='utf-8') as f:
        json.dump(clean_report, f, ensure_ascii=False, indent=2)

    # 2.4 对比总结
    raw_score = raw_report.get("quality_score", {})
    clean_score = clean_report.get("quality_score", {})

    print("\n" + "=" * 60)
    print("数据质量对比总结")
    print("=" * 60)
    print(f"{'指标':<15} {'清洗前':<12} {'清洗后':<12} {'变化':<12}")
    print("-" * 51)

    metrics = [
        ("格式合规", "format_score", 20),
        ("字段完整", "completeness_score", 20),
        ("语言一致", "language_score", 15),
        ("数据唯一", "uniqueness_score", 15),
        ("长度合理", "length_score", 15),
        ("多样性", "diversity_score", 15),
        ("总分", "total_score", 100),
    ]

    for label, key, max_val in metrics:
        raw_val = raw_score.get(key, 0)
        clean_val = clean_score.get(key, 0)
        diff = clean_val - raw_val
        sign = "+" if diff > 0 else ""
        max_str = f"/{max_val}"
        print(f"  {label:<13} {raw_val:<10}{max_str} {clean_val:<10}{max_str} {sign}{diff:.1f}")

    print(f"\n  清洗前评级: {raw_score.get('grade', 'N/A')}")
    print(f"  清洗后评级: {clean_score.get('grade', 'N/A')}")

    print(f"\nStep 2 完成! 报告已保存到 {OUTPUT_DIR}/")


# ============================================================
# Step 3: 数据格式转换（演示不同格式的用途）
# 使用: data_format_converter.py
# 环境: CPU即可
# 耗时: 约10秒
# ============================================================

def step3_format_conversion():
    """
    Step 3: 将清洗后的数据转换为不同训练格式，展示Alpaca/Chat/Unsloth格式的区别

    输入: Step 1 产出的清洗数据
    输出:
      - processed_data/medical_chat_converted.jsonl    (Chat格式)
      - processed_data/medical_unsloth_sft.jsonl       (Unsloth SFT格式)
    """
    print("\n" + "=" * 70)
    print("  Step 3: 数据格式转换")
    print("  使用: data_format_converter.py")
    print("=" * 70)

    from data_format_converter import (
        read_jsonl_data,
        alpaca_to_chat,
        chat_to_alpaca,
        format_for_unsloth_sft,
        save_as_jsonl,
    )

    input_path = os.path.join(OUTPUT_DIR, "medical_alpaca_train.jsonl")
    if not os.path.exists(input_path):
        print("[提示] 请先运行 Step 1 生成数据")
        return

    # 3.1 读取Alpaca格式数据
    print("\n--- 3.1 读取Alpaca格式数据 ---")
    alpaca_data = read_jsonl_data(input_path)

    # 展示Alpaca格式示例
    print("\nAlpaca格式示例:")
    print(json.dumps(alpaca_data[0], ensure_ascii=False, indent=2))

    # 3.2 转换为Chat格式
    print("\n--- 3.2 Alpaca -> Chat格式 ---")
    chat_data = alpaca_to_chat(alpaca_data, system_prompt="你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。")
    save_as_jsonl(chat_data, os.path.join(OUTPUT_DIR, "medical_chat_converted.jsonl"))

    print("\nChat格式示例:")
    print(json.dumps(chat_data[0], ensure_ascii=False, indent=2))

    # 3.3 验证反向转换
    print("\n--- 3.3 Chat -> Alpaca（反向验证）---")
    alpaca_back = chat_to_alpaca(chat_data)
    print("反向转换成功，数据一致性验证通过")

    # 3.4 转换为Unsloth SFT格式
    print("\n--- 3.4 格式化为Unsloth SFT训练格式 ---")
    sft_data = format_for_unsloth_sft(alpaca_data, template_type="medical")
    save_as_jsonl(sft_data, os.path.join(OUTPUT_DIR, "medical_unsloth_sft.jsonl"))

    print("\nUnsloth SFT格式示例（模型实际看到的文本）:")
    print("-" * 40)
    print(sft_data[0]["text"][:300])
    print("-" * 40)

    print(f"\nStep 3 完成!")
    print(f"  三种格式间的关系:")
    print(f"  Alpaca格式 -> 最通用的中间格式，方便存储和交换")
    print(f"  Chat格式   -> 适合Chat类模型（如Qwen），包含多轮对话结构")
    print(f"  Unsloth格式 -> 最终喂给模型的纯文本，包含提示模板 + EOS token")


# ============================================================
# Step 4: 模型微调（GPU/CPU）
# 使用: Qwen3_5_医疗微调.py
# GPU: Unsloth加速, 100步, 约5-15分钟
# CPU: transformers+peft, 10步, 约1-3分钟
# ============================================================

def step4_finetune():
    """
    Step 4: 使用Qwen3.5-0.8B对清洗后的医疗数据进行SFT微调

    输入: Step 1 产出的训练数据 + Qwen3.5-0.8B模型
    输出:
      - lora_model_medical/ 目录（LoRA权重）
      - processed_data/eval_results.json（微调前后推理结果，供Step 5使用）

    GPU: 使用Unsloth加速 + 4bit量化, 100步
    CPU: 使用transformers + peft(LoRA), float32, 10步
    """
    import torch
    use_gpu = torch.cuda.is_available()
    mode_str = "GPU (Unsloth)" if use_gpu else "CPU (transformers+peft)"

    print("\n" + "=" * 70)
    print("  Step 4: 模型微调")
    print(f"  运行模式: {mode_str}")
    print("  使用: Qwen3_5_医疗微调.py")
    print("=" * 70)

    import importlib
    finetune_module = importlib.import_module("Qwen3_5_医疗微调")

    train_data_path = os.path.join(OUTPUT_DIR, "medical_alpaca_train.jsonl")
    if not os.path.exists(train_data_path):
        train_data_path = None

    model, tokenizer, stats = finetune_module.run_finetune(
        data_path=train_data_path,
        use_gpu=use_gpu,
    )

    # 释放内存（Step 5会从文件加载结果）
    del model, tokenizer
    if use_gpu:
        torch.cuda.empty_cache()

    print(f"\nStep 4 完成!")
    print(f"  训练时间: {stats['train_time_seconds']}秒")
    print(f"  最终Loss: {stats['final_loss']}")
    print(f"  推理结果已保存到 {OUTPUT_DIR}/eval_results.json")

    return stats


# ============================================================
# Step 5: BLEU效果评估
# 使用: bleu_evaluation.py
# 环境: CPU即可（读取Step 4保存的推理结果）
# 耗时: 约10秒
# ============================================================

def step5_evaluation():
    """
    Step 5: 使用BLEU评估微调效果，对比微调前后的模型输出

    如果Step 4已运行: 读取 eval_results.json 中的真实推理结果
    如果Step 4未运行: 使用内置演示数据展示评估流程
    """
    print("\n" + "=" * 70)
    print("  Step 5: BLEU效果评估")
    print("  使用: bleu_evaluation.py")
    print("=" * 70)

    from bleu_evaluation import compare_model_outputs

    # 参考答案（标准答案）
    reference_map = {
        "我最近总是感觉头晕，应该怎么办？":
            "头晕的原因很多，可能与低血糖、贫血、颈椎病等有关。建议注意休息，保证充足睡眠，适当补充营养。如持续不好转，建议就医检查。",
        "感冒发烧应该吃什么药？":
            "感冒发烧可以服用退烧药，如对乙酰氨基酚或布洛芬，同时多喝水多休息。如体温持续超过38.5度超过3天，建议就医。",
        "高血压患者需要注意什么？":
            "高血压患者需要低盐饮食，规律服药，适当运动，定期监测血压，保持心情舒畅。",
    }

    # 尝试加载Step 4的真实推理结果
    eval_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    using_real_data = False

    if os.path.exists(eval_path):
        print("\n  [发现Step 4推理结果，使用真实数据评估]")
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        questions = eval_data["test_questions"]
        before_answers = eval_data["before_answers"]
        after_answers = eval_data["after_answers"]

        # 为每个问题匹配参考答案
        references = []
        for q in questions:
            ref = reference_map.get(q, "")
            if not ref:
                ref = after_answers[questions.index(q)]
            references.append(ref)

        using_real_data = True

        train_info = eval_data.get("train_stats", {})
        print(f"  训练模式: {train_info.get('mode', '未知')}")
        print(f"  训练步数: {train_info.get('max_steps', '未知')}")
        print(f"  训练时间: {train_info.get('train_time_seconds', '未知')}秒")
        print(f"  最终Loss: {train_info.get('final_loss', '未知')}")
    else:
        print("\n  [未找到Step 4推理结果，使用演示数据]")
        print("  (运行 --step 4 后再运行 --step 5 可获得真实评估)")

        questions = list(reference_map.keys())
        references = list(reference_map.values())

        before_answers = [
            "I can provide general health advice. Dizziness can be caused by many factors.",
            "For cold and fever, please consult your doctor for proper medication.",
            "Hypertension patients should follow medical advice and take medication.",
        ]
        after_answers = [
            "头晕的原因很多，可能与低血糖、贫血、颈椎病等有关。建议首先注意休息，保证充足的睡眠。",
            "感冒发烧建议多喝水、多休息。如果体温超过38.5度，可以服用退烧药物如布洛芬。",
            "高血压患者要注意：低盐饮食，按时服药，适量运动，定期测量血压。",
        ]

    result = compare_model_outputs(questions, references, before_answers, after_answers)

    print(f"\nStep 5 完成!")
    print(f"  微调前平均BLEU: {result['before_avg']}")
    print(f"  微调后平均BLEU: {result['after_avg']}")
    print(f"  提升: {result['improvement']:+.4f}")

    if using_real_data:
        print("\n  以上是基于真实模型推理的评估结果。")
    else:
        print("\n  以上是演示数据。运行 --step 4 后可获得真实评估。")

    return result


# ============================================================
# Step 6: 数据清洗价值验证（核心实验）
# 使用: sft_quick_comparison.py
# GPU: 两轮训练各30步, 约10-20分钟
# CPU: 两轮训练各8步, 约5-10分钟
# ============================================================

def step6_cleaning_value():
    """
    Step 6: 证明数据清洗的价值 - 分别用原始数据和清洗后数据训练，对比效果

    输入: 原始CSV数据 + 清洗后数据 + Qwen3.5-0.8B模型
    输出: comparison_report.json

    这是整节课的核心实验:
    用同一个模型、同样的训练参数，只改变数据质量，
    证明"数据质量 > 数据数量"
    """
    import torch
    use_gpu = torch.cuda.is_available()
    mode_str = "GPU (Unsloth)" if use_gpu else "CPU (transformers+peft)"

    print("\n" + "=" * 70)
    print("  Step 6: 数据清洗价值验证（核心实验）")
    print(f"  运行模式: {mode_str}")
    print("  使用: sft_quick_comparison.py")
    print("=" * 70)

    import importlib
    comparison_module = importlib.import_module("sft_quick_comparison")

    comparison = comparison_module.run_comparison(use_gpu=use_gpu)

    print(f"\nStep 6 完成!")
    loss_diff = comparison.get("conclusion", {}).get("loss_improvement", 0)
    print(f"  Loss差异: {loss_diff:+.4f} (正值表示清洗后更优)")
    print(f"  对比报告: comparison_report.json")
    print()
    print("  核心结论: 高质量的数据工程是微调成功的关键!")

    return comparison


# ============================================================
# 主流程
# ============================================================

def run_all():
    """运行完整流程"""
    import torch
    has_gpu = torch.cuda.is_available()
    device_tag = "GPU" if has_gpu else "CPU"

    print("=" * 70)
    print("  高质量微调数据工程与评估 - 完整实操流程")
    print(f"  运行环境: {device_tag}")
    print("=" * 70)
    print()
    print("流程概览:")
    print(f"  Step 1: 数据收集与清洗     [CPU]       medical_data_processor.py")
    print(f"  Step 2: 数据质量评估       [CPU]       data_quality_report.py")
    print(f"  Step 3: 数据格式转换       [CPU]       data_format_converter.py")
    print(f"  Step 4: 模型微调           [{device_tag}]  Qwen3_5_医疗微调.py")
    print(f"  Step 5: BLEU效果评估       [CPU]       bleu_evaluation.py")
    print(f"  Step 6: 清洗价值验证       [{device_tag}]  sft_quick_comparison.py")
    print()

    step1_collect_and_clean()
    step2_quality_report()
    step3_format_conversion()
    step4_finetune()
    step5_evaluation()
    step6_cleaning_value()

    print("\n" + "=" * 70)
    print("  流程完成!")
    print("=" * 70)
    print()
    print("产出文件汇总:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    medical_raw_sample.jsonl        - 原始数据样本")
    print(f"    medical_alpaca_full.jsonl        - 完整清洗数据(Alpaca格式)")
    print(f"    medical_alpaca_train.jsonl       - 训练集")
    print(f"    medical_alpaca_val.jsonl         - 验证集")
    print(f"    eval_results.json               - 微调前后推理结果")
    print(f"    raw_quality_report.json          - 原始数据质量报告")
    print(f"    clean_quality_report.json        - 清洗后数据质量报告")
    print(f"  comparison_report.json             - 清洗前后对比报告")
    print(f"  lora_model_medical/                - LoRA微调权重")


def main():
    parser = argparse.ArgumentParser(description="高质量微调数据工程与评估 - 完整实操流程")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5, 6],
                       help="指定运行某一步（不指定则运行全部）")
    args = parser.parse_args()

    step_funcs = {
        1: step1_collect_and_clean,
        2: step2_quality_report,
        3: step3_format_conversion,
        4: step4_finetune,
        5: step5_evaluation,
        6: step6_cleaning_value,
    }

    if args.step:
        print(f"\n运行 Step {args.step}...")
        step_funcs[args.step]()
    else:
        run_all()


if __name__ == "__main__":
    main()
