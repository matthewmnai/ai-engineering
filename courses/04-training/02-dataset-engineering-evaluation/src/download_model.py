# -*- coding: utf-8 -*-
"""
模型下载脚本
课程：高质量微调数据工程与评估
功能：下载 Qwen3.5-0.8B 模型到本地目录
"""

import os
import sys

# ========================================
# 配置区域（根据环境修改）
# ========================================

MODEL_ID = "Qwen/Qwen3.5-0.8B"

# AutoDL环境
AUTODL_CACHE_DIR = "/root/autodl-tmp/models"

# 本地Windows环境
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def detect_environment():
    """检测当前运行环境"""
    if sys.platform == "win32":
        return "windows", LOCAL_CACHE_DIR
    elif os.path.exists("/root/autodl-tmp"):
        return "autodl", AUTODL_CACHE_DIR
    else:
        return "linux", LOCAL_CACHE_DIR


def download_with_modelscope(model_id, cache_dir):
    """使用ModelScope下载模型（国内推荐）"""
    from modelscope import snapshot_download

    print(f"使用ModelScope下载模型: {model_id}")
    print(f"下载目录: {cache_dir}")

    os.makedirs(cache_dir, exist_ok=True)
    model_dir = snapshot_download(model_id, cache_dir=cache_dir)

    print(f"模型下载完成: {model_dir}")
    return model_dir


def download_with_huggingface(model_id, cache_dir):
    """使用HuggingFace下载模型"""
    from huggingface_hub import snapshot_download

    print(f"使用HuggingFace下载模型: {model_id}")
    print(f"下载目录: {cache_dir}")

    os.makedirs(cache_dir, exist_ok=True)
    model_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        local_dir=os.path.join(cache_dir, model_id.split("/")[-1]),
    )

    print(f"模型下载完成: {model_dir}")
    return model_dir


def main():
    env_name, cache_dir = detect_environment()
    print(f"检测到环境: {env_name}")
    print(f"模型: {MODEL_ID}")
    print(f"下载目录: {cache_dir}")
    print()

    # 优先使用ModelScope（国内速度更快）
    try:
        model_dir = download_with_modelscope(MODEL_ID, cache_dir)
    except ImportError:
        print("ModelScope未安装，尝试使用HuggingFace...")
        print("提示: pip install modelscope")
        try:
            model_dir = download_with_huggingface(MODEL_ID, cache_dir)
        except ImportError:
            print("HuggingFace Hub也未安装")
            print("请先安装: pip install modelscope 或 pip install huggingface_hub")
            return

    print()
    print("=" * 60)
    print("下载完成!")
    print(f"模型路径: {model_dir}")
    print()
    print("在微调脚本中使用此路径:")
    print(f'  model_name = "{model_dir}"')
    print("=" * 60)


if __name__ == "__main__":
    main()
