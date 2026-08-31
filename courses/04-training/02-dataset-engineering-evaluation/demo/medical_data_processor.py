# -*- coding: utf-8 -*-
"""
医疗数据收集、清洗、处理Pipeline
课程：高质量微调数据工程与评估
功能：从6个科室的CSV文件中收集、清洗、合并医疗对话数据，输出训练就绪的数据集

## 流程总览

Step 1 数据收集  ·  多编码策略读取 CSV，按目录名注入 department / source_file 元数据列
Step 2 数据清洗  ·  三层过滤（表格级问答列识别 → 行级清洗+校验 → 全局按 question MD5 去重）
Step 3 格式化    ·  转为 Alpaca / ChatML 格式，保存全量 JSONL 供正式训练
Step 4 采样切分  ·  按科室均衡采样 + 显式随机切分 train/val，产出快速实验用小数据集
"""

import os
import re
import json
import random
import hashlib
import pandas as pd
from collections import Counter, defaultdict
from datasets import Dataset


# ========================================
# 第一步：数据收集 - 多源异构数据整合
# ========================================

# 科室映射表
DEPARTMENTS = {
    'IM_内科': '内科',
    'Surgical_外科': '外科',
    'Pediatric_儿科': '儿科',
    'Oncology_肿瘤科': '肿瘤科',
    'OAGD_妇产科': '妇产科',
    'Andriatria_男科': '男科'
}


def read_csv_with_encoding(file_path):
    """多编码策略读取 CSV。

    优先严格模式：gb18030 → utf-8 → gbk → gb2312
    严格模式全部失败时回退 gb18030 容错模式（errors='replace'）。

    Args:
        file_path: CSV 文件路径

    Returns:
        pd.DataFrame: 读取成功的 DataFrame
    """
    import io

    encodings = ['gb18030', 'utf-8', 'gbk', 'gb2312']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"  编码: {encoding}, 行数: {len(df)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    print(f"  使用gb18030容错模式读取")
    with open(file_path, 'r', encoding='gb18030', errors='replace') as f:
        content = f.read()
    df = pd.read_csv(io.StringIO(content))
    print(f"  容错模式读取成功, 行数: {len(df)}")
    return df


def collect_medical_data(data_dir):
    """从 6 个科室目录批量收集原始医疗 CSV 数据。

    按 DEPARTMENTS 映射遍历 data_dir 下每个科室子目录，
    读取其中所有 CSV，附加 department / source_file 列后汇总。

    Args:
        data_dir: 数据根目录，下级应包含 DEPARTMENTS 中定义的 6 个子目录

    Returns:
        raw_data: List[pd.DataFrame]，每个科室的每个 CSV 对应一个 DataFrame
        stats: defaultdict(int)，按科室统计条数
    """
    raw_data = []
    stats = defaultdict(int)

    for dept_dir, dept_name in DEPARTMENTS.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"[警告] 目录不存在: {dept_path}")
            continue

        print(f"\n--- 处理{dept_name}数据 ---")
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            print(f"正在读取: {csv_file}")

            try:
                df = read_csv_with_encoding(file_path)
                print(f"  列名: {df.columns.tolist()}")
                print(f"  行数: {len(df)}")

                df['department'] = dept_name
                df['source_file'] = csv_file
                raw_data.append(df)
                stats[dept_name] += len(df)
            except Exception as e:
                print(f"  [错误] 处理文件失败: {e}")
                continue

    print(f"\n========== 数据收集统计 ==========")
    for dept, count in stats.items():
        print(f"  {dept}: {count} 条")
    print(f"  总计: {len(raw_data)}个表格，{sum(stats.values())} 条数据")

    return raw_data, stats, len(raw_data)

# ========================================
# 第二步：数据清洗
# ========================================

def extract_qa_fields(df):
    """【表格级】从 DataFrame 中识别问题列和回答列。

    列名忽略大小写和首尾空格后匹配候选列表：
    - 问题列候选：question / ask / 问题 / query / input
    - 回答列候选：answer / response / 回答 / output / reply

    只要有一个没匹配到，返回 (None, None)，调用方应整张表跳过。

    Args:
        df: 待解析的 DataFrame

    Returns:
        (question_col, answer_col): 匹配到的原始列名，未匹配则为 None
    """
    question_col = None
    answer_col = None

    question_candidates = ['question', 'ask', '问题', 'query', 'input']
    answer_candidates = ['answer', 'response', '回答', 'output', 'reply']

    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in question_candidates and question_col is None:
            question_col = col
        if col_lower in answer_candidates and answer_col is None:
            answer_col = col

    return question_col, answer_col


def clean_text(text):
    """【行级-清洗】文本标准化，不做过滤。

    依次执行：strip() → 连续空白压缩 → 移除不可见控制字符。
    非 str 类型直接返回空串。
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text


def is_valid_qa(question, answer, min_q_len=5, max_q_len=500, min_a_len=10, max_a_len=2000):
    """【行级-校验】判断问答对是否有效。

    按顺序检查，命中任何一条规则就返回 (False, reason)：
      1. 空值 → "空值"
      2. len(question) < min_q_len → "问题过短"
      3. len(question) > max_q_len → "问题过长"
      4. len(answer) < min_a_len → "回答过短"
      5. len(answer) > max_a_len → "回答过长"
      6. 问题仅含标点 → "问题仅含标点"
      7. 无意义问题（hello/纯数字等）→ "无意义问题"

    全部通过返回 (True, "有效")。
    """
    if not question or not answer:
        return False, "空值"

    if len(question) < min_q_len:
        return False, "问题过短"
    if len(question) > max_q_len:
        return False, "问题过长"
    if len(answer) < min_a_len:
        return False, "回答过短"
    if len(answer) > max_a_len:
        return False, "回答过长"

    if re.match(r'^[，。！？、；："‘’“”（）\s]+$', question):
        return False, "问题仅含标点"

    meaningless_patterns = [
        r'^(你好|hello|hi|嗯|哦|好的|谢谢)$',
        r'^[\d\s]+$',
    ]
    for pattern in meaningless_patterns:
        if re.match(pattern, question, re.IGNORECASE):
            return False, "无意义问题"

    return True, "有效"


def deduplicate_by_question(data_list):
    """【全局】按 question 内容 MD5 去重。

    相同问题只保留首次出现的那条，打印去重统计。

    Args:
        data_list: List[dict]，每条含 question 字段

    Returns:
        去重后的 List[dict]
    """
    seen_hashes = set()
    unique_data = []
    dup_count = 0

    for item in data_list:
        q_hash = hashlib.md5(item['question'].encode('utf-8')).hexdigest()
        if q_hash not in seen_hashes:
            seen_hashes.add(q_hash)
            unique_data.append(item)
        else:
            dup_count += 1

    print(f"  去重: 移除 {dup_count} 条重复数据")
    return unique_data


def clean_medical_data(raw_data_list):
    """Step 2 主流程：数据清洗 Pipeline。

    对应文件头流程图的四步编排：
      1. 逐个 DataFrame → extract_qa_fields 识别问答列，失败则整张表跳过
      2. 逐行 → clean_text 标准化 → is_valid_qa 过滤，过滤原因记入 filter_stats
      3. 所有表处理完 → deduplicate_by_question 全局按 question MD5 去重

    Args:
        raw_data_list: List[pd.DataFrame]，由 collect_medical_data 产出

    Returns:
        List[dict]，每条含 question / answer / department
    """
    print(f"\n========== 开始数据清洗 ==========")

    all_data = []
    filter_stats = Counter()

    for idx, df in enumerate(raw_data_list):
        print(f"正在处理表格 {idx+1}/{len(raw_data_list)}")

        q_col, a_col = extract_qa_fields(df)
        if q_col is None or a_col is None:
            print(f"  [警告] 无法识别问答列，跳过")
            continue

        for _, row in df.iterrows():
            question = clean_text(str(row[q_col]) if pd.notna(row[q_col]) else "")
            answer = clean_text(str(row[a_col]) if pd.notna(row[a_col]) else "")
            department = row.get('department', '未知')

            is_valid, reason = is_valid_qa(question, answer)
            if not is_valid:
                filter_stats[reason] += 1
                continue

            all_data.append({
                'question': question,
                'answer': answer,
                'department': department,
            })
            filter_stats["有效"] += 1

    print(f"\n--- 过滤统计 ---")
    for reason, count in filter_stats.most_common():
        print(f"  {reason}: {count} 条")

    all_data = deduplicate_by_question(all_data)

    print(f"\n清洗完成: {len(all_data)} 条有效数据")
    return all_data


# ========================================
# 第三步：数据格式化（转为训练格式）
# ========================================

def to_alpaca_format(data_list, system_prompt="你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。"):
    """
    转换为Alpaca格式，保留department字段用于质量评估中的多样性分析
    """
    alpaca_data = []
    for item in data_list:
        entry = {
            "instruction": system_prompt,
            "input": item['question'],
            "output": item['answer'],
        }
        if 'department' in item:
            entry["department"] = item['department']
        alpaca_data.append(entry)
    return alpaca_data


def to_chat_format(data_list, system_prompt="你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。"):
    """
    转换为Chat对话格式（适用于Qwen等Chat模型）
    """
    chat_data = []
    for item in data_list:
        chat_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['question']},
                {"role": "assistant", "content": item['answer']},
            ]
        })
    return chat_data


def save_dataset(data, output_path, format_type="jsonl"):
    """保存数据集"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    if format_type == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif format_type == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已保存: {output_path} ({len(data)} 条)")

# ========================================
# 第四步：数据采样（用于快速实验）
# ========================================

def sample_balanced_data(data_list, samples_per_dept=100, seed=42):
    """
    按科室均衡采样，用于快速实验
    """
    import random
    random.seed(seed)
    
    dept_data = defaultdict(list)
    for item in data_list:
        dept_data[item['department']].append(item)
    print(f"dept_data keys {dept_data.keys()}")
    sampled = []
    for dept, items in dept_data.items():
        n = min(samples_per_dept, len(items))
        sampled.extend(random.sample(items, n))
        print(f"  {dept}: 采样 {n} 条")
    
    random.shuffle(sampled)
    print(f"均衡采样完成: {len(sampled)} 条")
    return sampled

# ========================================
# 主流程
# ========================================



def main():
    # 配置数据目录（根据实际路径修改）
    # 工作目录设置
    # 设置工作目录 ai-engineering/courses/04-training/02-dataset-engineering-evaluation
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(SCRIPT_DIR)
    PROJET_DIR = os.getcwd()
    data_dir = "data/Data_中文医疗数据"
    output_dir = "outputs/clean_data"
    print(f"数据目录: {os.path.abspath(data_dir)}")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    
    # Step 1: 数据收集
    print("=" * 60)
    print("Step 1: 数据收集")
    print("=" * 60)
    raw_data, collect_stats, table_size = collect_medical_data(data_dir)

    # Step 2: 数据清洗
    print("\n" + "=" * 60)
    print("Step 2: 数据清洗")
    print("=" * 60)
    cleaned_data = clean_medical_data(raw_data)
    
    # Step 3: 格式转换与保存
    print("\n" + "=" * 60)
    print("Step 3: 格式转换与保存")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存Alpaca格式（完整）
    alpaca_data = to_alpaca_format(cleaned_data)
    save_dataset(alpaca_data, os.path.join(output_dir, "medical_alpaca_full.jsonl"))
    
    # 保存Chat格式（完整）
    chat_data = to_chat_format(cleaned_data)
    save_dataset(chat_data, os.path.join(output_dir, "medical_chat_full.jsonl"))
    
    # Step 4: 均衡采样（用于快速实验）
    print("\n" + "=" * 60)
    print("Step 4: 均衡采样（快速实验用）")
    print("=" * 60)
    sampled_data = sample_balanced_data(cleaned_data, samples_per_dept=200)
    sampled_alpaca = to_alpaca_format(sampled_data)
    save_dataset(sampled_alpaca, os.path.join(output_dir, "medical_alpaca_sampled.jsonl"))
    
    # 留出验证集（5%）—— 显式随机切分，不依赖上游顺序
    val_size = max(10, len(sampled_data) // 20)
    rng = random.Random(42)
    indices = list(range(len(sampled_data)))
    rng.shuffle(indices)

    val_indices   = indices[:val_size]
    train_indices = indices[val_size:]
    val_data   = to_alpaca_format([sampled_data[i] for i in val_indices])
    train_data = to_alpaca_format([sampled_data[i] for i in train_indices])

    print(f"\n数据划分: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")
    save_dataset(val_data,   os.path.join(output_dir, "medical_alpaca_val.jsonl"))
    save_dataset(train_data, os.path.join(output_dir, "medical_alpaca_train.jsonl"))
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"完整数据集: {len(alpaca_data)} 条")
    print(f"采样数据集: {len(sampled_alpaca)} 条")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")


if __name__ == "__main__":
    main()