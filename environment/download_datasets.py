"""批量下载数据集到本地 datasets/ 目录。

核心特性：
  1. backend="auto" → 优先 ModelScope，失败回退 HuggingFace
  2. type="public" → 存 datasets/public/，type="course" → 存 datasets/course/
  3. --target mac|autodl|all 按 runtime 字段过滤

用法:
    python environment/download_datasets.py --target autodl
    python environment/download_datasets.py --target mac 05_rag
    python environment/download_datasets.py --target all
"""
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "environment" / "env_config.yaml"


def load_config():
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_platform():
    if os.uname().sysname == "Darwin":
        return "mac"
    return "autodl"


def get_paths(cfg: dict, target: str) -> dict:
    paths_section = cfg.get("paths", {})
    if target in paths_section and isinstance(paths_section[target], dict):
        p = paths_section[target]
    else:
        p = {
            "datasets_public": "datasets/public",
            "datasets_course": "datasets/course",
        }
    result = {}
    for k, v in p.items():
        result[k] = v if os.path.isabs(v) else str(ROOT / v)
    return result


def should_download(item: dict, target: str) -> bool:
    if target == "all":
        return True
    runtime = item.get("runtime", "both")
    return runtime == "both" or runtime == target


def resolve_backend(item_backend: str, preferred: str) -> list:
    """解析 backend 字段，返回尝试顺序列表"""
    if item_backend == "auto":
        if preferred == "modelscope":
            return ["modelscope", "huggingface"]
        elif preferred == "huggingface":
            return ["huggingface", "modelscope"]
        elif preferred == "modelscope_only":
            return ["modelscope"]
        elif preferred == "huggingface_only":
            return ["huggingface"]
        else:
            return ["modelscope", "huggingface"]
    return [item_backend]


def get_subdir(item: dict, name: str) -> str:
    if "subdir" in item:
        return item["subdir"]
    return name.replace("/", "_")


def download_huggingface_ds(name: str, target_dir: str) -> bool:
    try:
        from datasets import load_dataset
        ds = load_dataset(name)
        ds.save_to_disk(target_dir)
        return True
    except Exception as e:
        print(f"    ↳ HuggingFace 失败: {e}")
        return False


def download_modelscope_ds(name: str, target_dir: str) -> bool:
    """用 ModelScope 下载数据集"""
    try:
        from modelscope.msdatasets import MsDataset
        # ModelScope 的数据集下载
        ds = MsDataset.load(name)
        # MsDataset 返回的是兼容 HF datasets 格式的对象
        if hasattr(ds, 'save_to_disk'):
            ds.save_to_disk(target_dir)
        else:
            # 转成 HF datasets 格式再保存
            import datasets
            if isinstance(ds, datasets.DatasetDict):
                ds.save_to_disk(target_dir)
            else:
                datasets.Dataset.from_dict(ds[:]).save_to_disk(target_dir)
        return True
    except Exception as e:
        print(f"    ↳ ModelScope 失败: {e}")
        return False


def download_one(item: dict, paths: dict, preferred_backend: str) -> bool:
    name = item.get("name", "")
    ds_type = item.get("type", "public")  # public / course
    backend = item.get("backend", "auto")

    # course 类型：课程专属数据，落 datasets/course/<subdir>/（迁移打包）
    if ds_type == "course":
        base_dir = paths.get("datasets_course", "datasets/course")
        target = Path(base_dir) / get_subdir(item, name)
        os.makedirs(target, exist_ok=True)
        print(f"  📂  课程数据目录: {target}")
        return True

    # public 类型：远程下载
    base_key = f"datasets_{ds_type}"  # datasets_public / datasets_course
    base_dir = paths.get(base_key, paths.get("datasets_public", "datasets/public"))
    subdir = get_subdir(item, name)
    target_dir = os.path.join(base_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)

    # 已下载则跳过
    if os.listdir(target_dir):
        print(f"  ✅  {subdir}/ 已存在，跳过")
        return True

    backends = resolve_backend(backend, preferred_backend)
    print(f"  ⬇️  [{ds_type}] {name} → datasets/{ds_type}/{subdir}/  (尝试: {' → '.join(backends)})")

    for b in backends:
        if b == "modelscope":
            if download_modelscope_ds(name, target_dir):
                print(f"    ✅ ModelScope 下载完成")
                return True
        elif b == "huggingface":
            if download_huggingface_ds(name, target_dir):
                print(f"    ✅ HuggingFace 下载完成")
                return True

    print(f"    ❌ 所有后端均失败")
    return False


def main():
    parser = argparse.ArgumentParser(description="批量下载数据集")
    parser.add_argument("--target", choices=["mac", "autodl", "all"],
                        help="目标平台（默认自动检测）")
    parser.add_argument("modules", nargs="*", help="可选：只下载指定模块")
    args = parser.parse_args()

    cfg = load_config()
    target = args.target or detect_platform()

    if not os.environ.get("HF_ENDPOINT"):
        mirror = cfg.get("env", {}).get("hf_mirror", "")
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror

    preferred = cfg.get("env", {}).get("preferred_backend", "modelscope")
    paths = get_paths(cfg, target)

    # 创建目录
    for key in ["datasets_public", "datasets_course"]:
        os.makedirs(paths.get(key, "datasets/public"), exist_ok=True)

    print(f"🤖  目标平台: {target}")
    print(f"📦  优先后端: {preferred}")
    print(f"📂  公开数据集: {paths['datasets_public']}")
    print(f"📂  课程数据集: {paths['datasets_course']}")

    groups = cfg.get("datasets", {})
    if args.modules:
        groups = {k: v for k, v in groups.items() if k in args.modules}

    total_ok, total_fail, total_skip = 0, 0, 0
    for group_key, items in groups.items():
        print(f"\n==== 数据集分组: {group_key} ====")
        for item in items:
            if not should_download(item, target):
                print(f"  ⏭️  [{item.get('runtime','both')}] {item.get('name', item.get('path'))} 跳过")
                total_skip += 1; continue
            if download_one(item, paths, preferred):
                total_ok += 1
            else:
                total_fail += 1

    print(f"\n{'='*50}")
    print(f"✅ 成功 {total_ok}  ⏭️ 跳过 {total_skip}  ❌ 失败 {total_fail}")
    print(f"\n💡 迁移到新 AutoDL 机器时:")
    print(f"   🔴 必带: {paths['datasets_course']}")
    print(f"   🟡 建议带: {paths['datasets_public']}（可重新下载但省时间）")


if __name__ == "__main__":
    main()
