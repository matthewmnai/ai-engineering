"""批量下载 HuggingFace 数据集到本地 data_root。

读取 environment/env_config.yaml，按模块分组下载。

用法同 download_models.py：
    python environment/download_datasets.py            # 全量
    python environment/download_datasets.py module4_training  # 指定分组
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "environment" / "env_config.yaml"


def load_config():
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_one(item: dict, data_root: str) -> bool:
    """下载单个数据集；本地样例则跳过远程拉取"""
    if "path" in item:
        # 本地小样例：只确保目录存在，不触发远程下载
        p = item["path"]
        target = ROOT / p if not os.path.isabs(p) else Path(p)
        os.makedirs(target, exist_ok=True)
        print(f"  📂  本地样例目录已就绪: {target}")
        return True

    name = item["name"]
    backend = item.get("backend", "huggingface")
    target = os.path.join(data_root, name.replace("/", "_"))

    os.makedirs(target, exist_ok=True)
    print(f"  ⬇️  [{backend}] {name} → {target}")

    try:
        if backend == "huggingface":
            from datasets import load_dataset
            ds = load_dataset(name)
            ds.save_to_disk(target)
        else:
            print(f"  ⚠️  暂不支持 backend={backend}")
            return False
    except Exception as e:
        print(f"  ❌  {name} 下载失败: {e}")
        return False
    return True


def main():
    cfg = load_config()

    # HF 镜像
    if not os.environ.get("HF_ENDPOINT"):
        mirror = cfg.get("env", {}).get("hf_mirror", "")
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror

    data_root = cfg.get("paths", {}).get("data_root")
    if not data_root:
        print("❌  env_config.yaml: paths.data_root 未配置")
        sys.exit(1)
    os.makedirs(data_root, exist_ok=True)
    print(f"📦  数据集根: {data_root}")

    targets = sys.argv[1:]
    groups = cfg.get("datasets", {})

    total_ok, total_fail = 0, 0
    for group_key, items in groups.items():
        if targets and group_key not in targets:
            continue
        print(f"\n==== 数据集分组: {group_key} ({len(items)} 个) ====")
        for item in items:
            if download_one(item, data_root):
                total_ok += 1
            else:
                total_fail += 1

    print(f"\n✅ 成功 {total_ok}  ❌ 失败 {total_fail}")


if __name__ == "__main__":
    main()
