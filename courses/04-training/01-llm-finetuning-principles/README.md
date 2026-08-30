# 01 · LLM 微调底层原理

> 模块 4 · 模型训练与微调 ｜ 日期：08-17

## 环境标签

🖥️ **仅 AutoDL GPU**（必须有 CUDA 显卡）

## 学习目标

- 理解大语言模型微调的底层原理与数学基础
- 掌握全量微调与参数高效微调（PEFT）的本质区别
- 搞懂 LoRA / QLoRA 的低秩分解原理与显存优势
- 理解前向传播、反向传播、梯度更新在微调中的作用

## 知识点

| # | 知识点 | 说明 |
|:-:|---|---|
| 1 | 全量微调（Full Fine-tuning） | 更新全部参数，效果最好但显存开销巨大 |
| 2 | 冻结与解冻策略 | 选择性更新部分层，平衡效果与成本 |
| 3 | LoRA 原理 | 低秩矩阵分解 `W + ΔW = W + BA`，只训练 B、A |
| 4 | QLoRA 原理 | 4-bit 量化基座 + LoRA，显存极致压缩 |
| 5 | 反向传播与梯度 | 链式法则、梯度下降、学习率调度 |
| 6 | 显存优化技巧 | 梯度累积、混合精度、梯度检查点 |

## 运行说明

```bash
# 在 AutoDL 上，先确保环境就绪
bash setup.sh                       # 路由到 setup-autodl.sh
python environment/verify_env.py    # 自检 CUDA / GPU 依赖

# 拉取本模块所需基座模型（按 runtime 过滤，只拉 AutoDL 的）
python environment/download_models.py --target autodl 04_training
```

- 训练脚本位于 `src/`，运行产出写入 `outputs/`（已被 `.gitignore` 忽略）
- 基座模型存放路径：`/root/autodl-tmp/ai-engineering/models/pretrained/`
