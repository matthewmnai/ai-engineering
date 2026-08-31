# -*- coding: utf-8 -*-
"""
数据格式转换工具
课程：高质量微调数据工程与评估
功能：在不同训练数据格式之间进行转换
支持格式：CSV, JSON, JSONL, Alpaca, Chat(Messages)
"""

import os
import json
import pandas as pd


# ========================================
# 格式定义与示例
# ========================================

# Alpaca格式示例
ALPACA_EXAMPLE = {
    "instruction": "请回答以下医疗相关问题",
    "input": "我最近总是感觉头晕，应该怎么办？",
    "output": "头晕的原因很多，建议您首先注意休息..."
}

# Chat消息格式示例（适用于Qwen等Chat模型）
CHAT_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "你是一个专业的医疗助手。"},
        {"role": "user", "content": "我最近总是感觉头晕，应该怎么办？"},
        {"role": "assistant", "content": "头晕的原因很多，建议您首先注意休息..."}
    ]
}


# ========================================
# 读取函数
# ========================================

def read_csv_data(file_path, question_col='question', answer_col='answer'):
    """读取CSV格式数据，gb18030是GBK超集优先尝试"""
    import io
    encodings = ['gb18030', 'utf-8', 'gbk', 'gb2312']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"成功读取CSV，编码: {enc}，行数: {len(df)}")
            print(f"列名: {list(df.columns)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 容错模式
    with open(file_path, 'r', encoding='gb18030', errors='replace') as f:
        df = pd.read_csv(io.StringIO(f.read()))
    print(f"成功读取CSV(容错模式)，行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    return df


def read_jsonl_data(file_path):
    """读取JSONL格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"成功读取JSONL: {len(data)} 条")
    return data


def read_json_data(file_path):
    """读取JSON格式数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"成功读取JSON: {len(data)} 条")
        return data
    else:
        raise ValueError("JSON文件顶层必须是数组")


# ========================================
# 转换函数
# ========================================

def csv_to_alpaca(df, instruction="请回答以下问题", 
                  question_col='question', answer_col='answer'):
    """CSV -> Alpaca格式"""
    q_col = None
    a_col = None
    
    col_map_q = ['question', 'ask', '问题', 'query', 'input']
    col_map_a = ['answer', 'response', '回答', 'output', 'reply']
    
    for col in df.columns:
        if col.strip().lower() in col_map_q:
            q_col = col
        if col.strip().lower() in col_map_a:
            a_col = col
    
    if q_col is None or a_col is None:
        print(f"  可用列名: {list(df.columns)}")
        raise ValueError(f"无法自动匹配问答列，请指定 question_col 和 answer_col")
    
    result = []
    for _, row in df.iterrows():
        q = str(row[q_col]).strip() if pd.notna(row[q_col]) else ""
        a = str(row[a_col]).strip() if pd.notna(row[a_col]) else ""
        if q and a:
            result.append({
                "instruction": instruction,
                "input": q,
                "output": a,
            })
    
    print(f"CSV -> Alpaca: {len(result)} 条")
    return result


def alpaca_to_chat(alpaca_data, system_prompt=None):
    """Alpaca格式 -> Chat消息格式"""
    chat_data = []
    for item in alpaca_data:
        messages = []
        
        sys_content = system_prompt or item.get("instruction", "")
        if sys_content:
            messages.append({"role": "system", "content": sys_content})
        
        user_content = item.get("input", "")
        if not user_content:
            user_content = item.get("instruction", "")
        messages.append({"role": "user", "content": user_content})
        
        messages.append({"role": "assistant", "content": item.get("output", "")})
        
        chat_data.append({"messages": messages})
    
    print(f"Alpaca -> Chat: {len(chat_data)} 条")
    return chat_data


def chat_to_alpaca(chat_data):
    """Chat消息格式 -> Alpaca格式"""
    alpaca_data = []
    for item in chat_data:
        messages = item.get("messages", [])
        
        instruction = ""
        user_input = ""
        output = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                instruction = content
            elif role == "user":
                user_input = content
            elif role == "assistant":
                output = content
        
        if user_input and output:
            alpaca_data.append({
                "instruction": instruction or "请回答以下问题",
                "input": user_input,
                "output": output,
            })
    
    print(f"Chat -> Alpaca: {len(alpaca_data)} 条")
    return alpaca_data


# ========================================
# Unsloth训练数据格式化
# ========================================

def create_unsloth_prompt_template(format_type="medical"):
    """
    创建Unsloth训练用的提示模板
    """
    templates = {
        "medical": """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}""",
        "alpaca": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""",
        "general": """请根据以下问题提供详细的回答。

### 问题：
{}

### 回答：
{}""",
    }
    
    return templates.get(format_type, templates["general"])


def format_for_unsloth_sft(data, template_type="medical", eos_token="<|endoftext|>"):
    """
    将数据格式化为Unsloth SFTTrainer可用的格式
    """
    template = create_unsloth_prompt_template(template_type)
    
    formatted = []
    for item in data:
        if template_type == "alpaca":
            text = template.format(
                item.get("instruction", ""),
                item.get("input", ""),
                item.get("output", "")
            ) + eos_token
        else:
            text = template.format(
                item.get("input", item.get("question", "")),
                item.get("output", item.get("answer", ""))
            ) + eos_token
        
        formatted.append({"text": text})
    
    print(f"格式化为Unsloth SFT格式: {len(formatted)} 条")
    return formatted


# ========================================
# 保存函数
# ========================================

def save_as_jsonl(data, output_path):
    """保存为JSONL格式"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存: {output_path} ({len(data)} 条)")


def save_as_json(data, output_path):
    """保存为JSON格式"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path} ({len(data)} 条)")


# ========================================
# 主流程 - 演示各种转换
# ========================================

def main():
    print("=" * 60)
    print("数据格式转换工具 - 演示")
    print("=" * 60)
    
    # 演示1：创建示例数据
    print("\n--- 示例：Alpaca格式 ---")
    print(json.dumps(ALPACA_EXAMPLE, ensure_ascii=False, indent=2))
    
    print("\n--- 示例：Chat格式 ---")
    print(json.dumps(CHAT_EXAMPLE, ensure_ascii=False, indent=2))
    
    # 演示2：格式互转
    sample_alpaca = [
        {"instruction": "请回答医疗问题", "input": "感冒发烧怎么办？", "output": "建议多休息、多喝水，如体温超过38.5度可服用退烧药。"},
        {"instruction": "请回答医疗问题", "input": "高血压要注意什么？", "output": "控制饮食、减少盐分摄入、适当运动、规律服药。"},
    ]
    
    print("\n--- 演示：Alpaca -> Chat ---")
    chat_result = alpaca_to_chat(sample_alpaca, system_prompt="你是一个专业的医疗助手")
    print(json.dumps(chat_result[0], ensure_ascii=False, indent=2))
    
    print("\n--- 演示：Chat -> Alpaca ---")
    alpaca_result = chat_to_alpaca(chat_result)
    print(json.dumps(alpaca_result[0], ensure_ascii=False, indent=2))
    
    print("\n--- 演示：格式化为Unsloth SFT ---")
    sft_result = format_for_unsloth_sft(sample_alpaca, template_type="medical")
    print(sft_result[0]["text"])
    
    print("\n格式转换演示完成!")


if __name__ == "__main__":
    main()
