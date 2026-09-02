"""项目路径定义 —— 单一真实源（Single Source of Truth）。

所有课程脚本和工具脚本都应从此模块导入路径，避免硬编码。

设计原则：
  1. 从 env_config.yaml 读取路径配置（单一真实源）
  2. 自动检测当前平台（mac / autodl），解析对应路径
  3. autodl 用绝对路径（数据盘），mac 用相对 ROOT 的相对路径
  4. 额外提供课程脚本常用的便捷 helper

用法：
    from environment.paths import PATHS, ROOT, MODELS_PRETRAINED

    # 方式 1：直接用常量
    print(MODELS_PRETRAINED)

    # 方式 2：dict 形式，遍历创建
    import os
    os.makedirs(PATHS["models_pretrained"], exist_ok=True)

    # 方式 3：便捷 helper
    from environment.paths import get_model_dir, get_dataset_dir, ensure_dirs
    ensure_dirs()                                # 一次性创建所有目录
    model_dir = get_model_dir("Qwen2.5-7B-Instruct", "pretrained")
    dataset_dir = get_dataset_dir("alpaca-cleaned", "public")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# ============ 项目根目录 ============
# 本文件位于 environment/，向上一级即项目根
ROOT: Path = Path(__file__).resolve().parents[1]


# ============ 平台检测 ============
def detect_platform() -> str:
    """检测当前运行平台：'mac' 或 'autodl'"""
    if os.uname().sysname == "Darwin":
        return "mac"
    return "autodl"


PLATFORM: str = detect_platform()


# ============ 加载 env_config.yaml 中的路径配置 ============
def _load_yaml_paths() -> dict:
    """从 env_config.yaml 读取 paths 节"""
    config_path = ROOT / "environment" / "env_config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("paths", {})
    except Exception:
        return {}


def resolve_paths(platform: Optional[str] = None) -> dict:
    """解析指定平台的所有路径，相对路径自动拼 ROOT。

    这是公开函数，download 脚本等需要 --target 覆盖平台时调用。

    Args:
        platform: 'mac' 或 'autodl'，默认取 detect_platform()

    Returns:
        路径字典，key 如 'models_pretrained'，value 为绝对路径字符串
    """
    target = platform or detect_platform()
    yaml_paths = _load_yaml_paths()

    # 取目标平台的路径配置
    platform_cfg = yaml_paths.get(target, {})

    # 兜底默认值（与 env_config.yaml autodl 节保持一致）
    defaults = {
        "models_pretrained": str(ROOT / "models" / "pretrained"),
        "models_finetuned": str(ROOT / "models" / "finetuned"),
        "datasets_public": str(ROOT / "datasets" / "public"),
        "datasets_course": str(ROOT / "datasets" / "course"),
        # 按课程细分的 course 数据集（方便课程脚本直接引用）
        "course_04_02": str(ROOT / "datasets" / "course" / "04-training" / "02-dataset-engineering-evaluation"),
        "checkpoints": str(ROOT / "checkpoints"),
        "outputs": str(ROOT / "outputs"),
        "logs": str(ROOT / "logs"),
    }

    # 合并：yaml 里有的用 yaml，没有的用 defaults
    merged = {**defaults, **platform_cfg}

    # 解析相对路径
    result = {}
    for k, v in merged.items():
        if os.path.isabs(v):
            result[k] = v
        else:
            result[k] = str(ROOT / v)

    return result


# ============ 路径字典（全局唯一实例，基于自动检测的平台） ============
PATHS: dict = resolve_paths()


# ============ 方便单独导入的常量 ============
MODELS_PRETRAINED: str = PATHS["models_pretrained"]
MODELS_FINETUNED: str = PATHS["models_finetuned"]
DATASETS_PUBLIC: str = PATHS["datasets_public"]
DATASETS_COURSE: str = PATHS["datasets_course"]
COURSE_04_02: str = PATHS["course_04_02"]   # 模块4 课程02 医疗数据
CHECKPOINTS: str = PATHS["checkpoints"]
OUTPUTS: str = PATHS["outputs"]
LOGS: str = PATHS["logs"]


# ============ 扩展目录（env_config.yaml 未定义但课程脚本常用） ============
CONFIGS: str = str(ROOT / "configs")
SCRIPTS: str = str(ROOT / "scripts")
TMP: str = str(ROOT / "tmp")


# ============ 便捷 Helper ============
def ensure_dirs(*keys: str) -> list[str]:
    """创建指定的路径目录。不传参则创建所有 PATHS 中的目录。

    Returns:
        已创建/确认存在的绝对路径列表
    """
    targets = keys if keys else tuple(PATHS.keys())
    created = []
    for key in targets:
        path = PATHS.get(key)
        if path:
            os.makedirs(path, exist_ok=True)
            created.append(path)
    return created


def get_model_dir(repo_id: str, model_type: str = "pretrained") -> str:
    """根据 repo_id 和类型，返回模型应该存放的绝对路径。

    Args:
        repo_id: 模型仓库名，如 'Qwen/Qwen2.5-7B-Instruct' 或 'Qwen2.5-7B-Instruct'
        model_type: 'pretrained'(基座) 或 'finetuned'(微调产出)

    Returns:
        绝对路径字符串
    """
    base_key = f"models_{model_type}"  # models_pretrained / models_finetuned
    base_dir = PATHS.get(base_key, PATHS["models_pretrained"])
    # 取子目录名：repo_id 最后一段
    subdir = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    return os.path.join(base_dir, subdir)


def get_dataset_dir(name: str, ds_type: str = "public") -> str:
    """根据名称和类型，返回数据集应该存放的绝对路径。

    Args:
        name: 数据集名称，如 'AI-ModelScope/alpaca-cleaned'
        ds_type: 'public'(公开下载) 或 'course'(课程专属)

    Returns:
        绝对路径字符串
    """
    base_key = f"datasets_{ds_type}"  # datasets_public / datasets_course
    base_dir = PATHS.get(base_key, PATHS["datasets_public"])
    # 子目录名：把 / 替换成 _（和 download_datasets.py 保持一致）
    subdir = name.replace("/", "_")
    return os.path.join(base_dir, subdir)


def get_checkpoint_path(run_name: str, filename: str = "") -> str:
    """获取某次训练 run 的 checkpoint 目录（或具体文件路径）。"""
    run_dir = os.path.join(CHECKPOINTS, run_name)
    if filename:
        return os.path.join(run_dir, filename)
    return run_dir


def get_output_path(run_name: str, filename: str = "") -> str:
    """获取某次 run 的 outputs 目录（或具体文件路径）。"""
    run_dir = os.path.join(OUTPUTS, run_name)
    if filename:
        return os.path.join(run_dir, filename)
    return run_dir


def get_log_path(run_name: str, filename: str = "") -> str:
    """获取某次 run 的 logs 目录（或具体文件路径）。"""
    run_dir = os.path.join(LOGS, run_name)
    if filename:
        return os.path.join(run_dir, filename)
    return run_dir


# ============ 打印当前路径配置（调试用） ============
def print_paths() -> None:
    """打印所有已解析的路径，便于快速确认。"""
    print(f"Platform: {PLATFORM}")
    print(f"ROOT:     {ROOT}")
    print("-" * 60)
    for k, v in PATHS.items():
        print(f"  {k:<22} {v}")
    print("-" * 60)


# ============ 模块入口 ============
if __name__ == "__main__":
    print_paths()
