"""批量下载 HuggingFace 模型权重到本地缓存。

读取 environment/env_config.yaml，按模块分组下载，支持只下某组。
依赖 huggingface_hub（已在 requirements-base.txt 中）。

用法:
    # 全量下载
    python environment/download_models.py

    # 只下载某个模块的模型（模块名即 YAML 中 models 下的 key）
    python environment/download_models.py module4_training

    # 下载多个模块
    python environment/download_models.py module1_llm_basics module5_rag
"""
import os
import sys
from pathlib import Path

# 允许直接 `python environment/download_models.py` 运行，无需改成 module
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "environment" / "env_config.yaml"


def load_config():
    """安全读取 env_config.yaml（不引入 yaml 依赖时的兜底逻辑）"""
    try:
        import yaml  # type: ignore
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # 极简兜底：直接 exec YAML 里需要的字段 —— 仅用于首次 bootstrap
        print("⚠️  yaml 未安装，请先运行 setup.sh 或 pip install pyyaml")
        sys.exit(1)


def download_one(name: str, backend: str, cache_dir: str):
    """下载单个模型到 cache_dir"""
    if backend == "huggingface":
        from huggingface_hub import snapshot_download
        print(f"  ⬇️  [{backend}] {name}")
        snapshot_download(
            repo_id=name,
            cache_dir=cache_dir,
            resume_download=True,
        )
    elif backend == "modelscope":
        try:
            from modelscope import snapshot_download as ms_download
        except ImportError:
            print(f"  ❌  backend=modelscope 但未安装 modelscope")
            return False
        print(f"  ⬇️  [{backend}] {name}")
        ms_download(model_id=name, cache_dir=cache_dir)
    else:
        print(f"  ⚠️  未知 backend={backend}，跳过 {name}")
        return False
    return True


def main():
    cfg = load_config()

    # HF 镜像：setup.sh 已持久化 HF_ENDPOINT，这里兼容手动运行场景
    if not os.environ.get("HF_ENDPOINT"):
        mirror = cfg.get("env", {}).get("hf_mirror", "")
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror
            print(f"🔄  设置 HF 镜像: {mirror}")

    cache_dir = cfg.get("paths", {}).get("model_cache")
    if not cache_dir:
        print("❌  env_config.yaml: paths.model_cache 未配置")
        sys.exit(1)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📦  缓存目录: {cache_dir}")

    # 过滤目标分组
    targets = sys.argv[1:]
    groups = cfg.get("models", {})
    if not groups:
        print("⚠️  env_config.yaml: models 未配置任何项")
        return

    print(f"\n🗺️  待下载分组: {list(groups.keys())}")
    if targets:
        print(f"🎯  仅下载: {targets}")

    total_ok, total_fail = 0, 0
    for group_key, items in groups.items():
        if targets and group_key not in targets:
            continue
        print(f"\n==== 模块分组: {group_key} ({len(items)} 个) ====")
        for item in items:
            ok = download_one(
                name=item["name"],
                backend=item.get("backend", "huggingface"),
                cache_dir=cache_dir,
            )
            if ok:
                total_ok += 1
            else:
                total_fail += 1
    print(f"\n✅ 成功 {total_ok}  ❌ 失败 {total_fail}")


if __name__ == "__main__":
    main()
