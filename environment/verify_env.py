"""环境自检：检查关键依赖、镜像配置、CUDA。
setup.sh 末尾会自动运行此脚本。
"""
import importlib
import os
import sys

# 按功能分组，每组给出核心包清单
REQUIRED = {
    "基础生态": ["torch", "transformers", "huggingface_hub", "datasets", "peft"],
    "框架应用": ["langchain", "langchain_community", "openai"],
    "工具": ["yaml", "numpy", "PIL"],
}

OPTIONAL = {
    "微调加速": ["unsloth"],
    "TensorFlow": ["tensorflow"],
    "模型源": ["modelscope"],
}


def group_check(packages: dict) -> tuple[int, int]:
    ok_cnt, fail_cnt = 0, 0
    for group, pkgs in packages.items():
        print(f"\n[{group}]")
        for p in pkgs:
            try:
                m = importlib.import_module(p)
                ver = getattr(m, "__version__", "unknown")
                print(f"  ✅ {p:<22} {ver}")
                ok_cnt += 1
            except Exception:
                print(f"  ❌ {p:<22} 未安装")
                fail_cnt += 1
    return ok_cnt, fail_cnt


def check_cuda():
    try:
        import torch
        available = torch.cuda.is_available()
        print(f"\n[CUDA]")
        print(f"  可用: {available}")
        if available:
            print(f"  设备: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("\n[CUDA] torch 未安装，跳过检测")


def main():
    print("=" * 50)
    print("  AI Engineering —— 环境自检")
    print("=" * 50)

    ok, fail = group_check(REQUIRED)
    print(f"\n── 可选依赖 ──")
    opt_ok, opt_fail = group_check(OPTIONAL)

    print(f"\n[镜像]")
    mirror = os.environ.get("HF_ENDPOINT", "")
    if mirror:
        print(f"  ✅ HF_ENDPOINT = {mirror}")
    else:
        print(f"  ⚠️  HF_ENDPOINT 未设置（国内网络建议设为 https://hf-mirror.com）")

    check_cuda()

    print("\n" + "=" * 50)
    print(f"  必选依赖: ✅ {ok}  ❌ {fail}")
    print(f"  可选依赖: ✅ {opt_ok}  ❌ {opt_fail}")
    if fail == 0:
        print("  ✅ 核心依赖全部就绪，可以开始学习！")
    else:
        print("  ⚠️  还有必选依赖缺失，请先运行 setup.sh")
    print("=" * 50)

    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
