# 03 · LLM 模型蒸馏 + 微调实操落地

> 模块 4 · 模型训练与微调 ｜ 日期：08-24

## 环境标签

🖥️ **仅 AutoDL GPU**（必须有 CUDA 显卡）

## 学习目标

- 理解知识蒸馏的核心思想：大模型（Teacher）→ 小模型（Student）
- 掌握白盒蒸馏与黑盒蒸馏的适用场景
- 完成一个完整的微调实操落地流程（数据→训练→保存→推理）
- 学会保存 / 加载 LoRA 适配器并合并权重

## 知识点

| # | 知识点 | 说明 |
|:-:|---|---|
| 1 | 知识蒸馏原理 | 用 Teacher 的软标签训 Student，逼近大模型能力 |
| 2 | 白盒蒸馏 | 能拿到 Teacher 权重，用 logits / 中间层对齐 |
| 3 | 黑盒蒸馏 | 只能调用 Teacher API，靠生成数据蒸馏 |
| 4 | 微调实操全流程 | 数据准备 → LoRA 训练 → 保存适配器 → 合并 → 推理验证 |
| 5 | 适配器保存与合并 | `save_pretrained` + `merge_and_unload` |
| 6 | 训练监控 | loss 曲线、过拟合识别、早停策略 |

## 运行说明

```bash
bash setup.sh
python environment/verify_env.py

# 拉取 Teacher / Student 基座模型
python environment/download_models.py --target autodl 04_training
```

- 微调产出保存在 `/root/autodl-tmp/ai-engineering/models/finetuned/`（不可再生，迁移必带）
- 训练脚本、蒸馏脚本位于 `src/`
