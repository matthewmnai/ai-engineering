"""批量下载模型权重到本地 models/ 目录。

核心特性：
  1. backend="auto" → 优先 ModelScope，失败回退 HuggingFace（国内更快）
  2. type="pretrained" → 存 models/pretrained/，type="finetuned" → 存 models/finetuned/
  3. --target mac|autodl|all 按 runtime 字段过滤
  4. 每个 repo 存到独立子目录（不混放），方便迁移

用法:
    python environment/download_models.py                              # 自动检测平台
    python environment/download_models.py --target mac                 # 只下载 Mac 用的
    python environment/download_models.py --target autodl 04_training  # 指定模块
    python environment/download_models.py --target all                 # 全量
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
    """取当前平台对应的路径配置，相对路径自动拼 ROOT"""
    paths_section = cfg.get("paths", {})
    if target in paths_section and isinstance(paths_section[target], dict):
        p = paths_section[target]
    else:
        # 兜底旧结构
        p = {
            "models_pretrained": "models/pretrained",
            "models_finetuned": "models/finetuned",
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
    """取子目录名：优先用 item.subdir，否则取 name 最后一段"""
    if "subdir" in item:
        return item["subdir"]
    return name.split("/")[-1]


def download_modelscope(name: str, target_dir: str) -> bool:
    """用 ModelScope 下载，存到 target_dir"""
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError:
        print(f"    ↳ modelscope 未安装，回退")
        return False
    try:
        # ModelScope 的 cache_dir 会创建 models--{org}--{name} 结构
        # 我们用 local_dir 直接指定到目标目录
        ms_download(model_id=name, local_dir=target_dir)
        return True
    except Exception as e:
        print(f"    ↳ ModelScope 失败: {e}")
        return False


def download_huggingface(name: str, target_dir: str) -> bool:
    """用 HuggingFace 下载，存到 target_dir"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(f"    ↳ huggingface_hub 未安装")
        return False
    try:
        snapshot_download(
            repo_id=name,
            local_dir=target_dir,
            resume_download=True,
        )
        return True
    except Exception as e:
        print(f"    ↳ HuggingFace 失败: {e}")
        return False


def download_one(item: dict, paths: dict, preferred_backend: str) -> bool:
    """下载单个模型，返回是否成功"""
    name = item["name"]
    model_type = item.get("type", "pretrained")  # pretrained / finetuned
    backend = item.get("backend", "auto")

    # 确定目标目录
    base_key = f"models_{model_type}"  # models_pretrained / models_finetuned
    base_dir = paths.get(base_key, paths.get("models_pretrained", "models/pretrained"))
    subdir = get_subdir(item, name)
    target_dir = os.path.join(base_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)

    # 已下载则跳过（检查目录非空）
    if os.listdir(target_dir):
        print(f"  ✅  {subdir}/ 已存在，跳过")
        return True

    # 解析后端尝试顺序
    backends = resolve_backend(backend, preferred_backend)
    print(f"  ⬇️  [{model_type}] {name} → models/{model_type}/{subdir}/  (尝试: {' → '.join(backends)})")

    for b in backends:
        if b == "modelscope":
            if download_modelscope(name, target_dir):
                print(f"    ✅ ModelScope 下载完成")
                return True
        elif b == "huggingface":
            if download_huggingface(name, target_dir):
                print(f"    ✅ HuggingFace 下载完成")
                return True

    print(f"    ❌ 所有后端均失败")
    return False


def main():
    parser = argparse.ArgumentParser(description="批量下载模型")
    parser.add_argument("--target", choices=["mac", "autodl", "all"],
                        help="目标平台（默认自动检测）")
    parser.add_argument("modules", nargs="*",
                        help="可选：只下载指定模块（如 04_training）")
    args = parser.parse_args()

    cfg = load_config()
    target = args.target or detect_platform()

    # HF 镜像
    if not os.environ.get("HF_ENDPOINT"):
        mirror = cfg.get("env", {}).get("hf_mirror", "")
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror

    preferred = cfg.get("env", {}).get("preferred_backend", "modelscope")
    paths = get_paths(cfg, target)

    # 创建目录
    for key in ["models_pretrained", "models_finetuned"]:
        os.makedirs(paths.get(key, "models/pretrained"), exist_ok=True)

    print(f"🤖  目标平台: {target}")
    print(f"📦  优先后端: {preferred}")
    print(f"📂  基座模型: {paths['models_pretrained']}")
    print(f"📂  微调产出: {paths['models_finetuned']}")

    groups = cfg.get("models", {})
    if args.modules:
        groups = {k: v for k, v in groups.items() if k in args.modules}

    total_ok, total_fail, total_skip = 0, 0, 0
    for group_key, items in groups.items():
        print(f"\n==== 模块分组: {group_key} ====")
        for item in items:
            if not should_download(item, target):
                print(f"  ⏭️  [{item.get('runtime','both')}] {item['name']} 跳过（非 {target} 目标）")
                total_skip += 1; continue
            if download_one(item, paths, preferred):
                total_ok += 1
            else:
                total_fail += 1

    print(f"\n{'='*50}")
    print(f"✅ 成功 {total_ok}  ⏭️ 跳过 {total_skip}  ❌ 失败 {total_fail}")
    print(f"\n💡 迁移到新 AutoDL 机器时:")
    print(f"   🔴 必带: {paths['models_finetuned']}")
    print(f"   🟢 可不带: {paths['models_pretrained']}（本脚本可重新下载）")


if __name__ == "__main__":
    main()
