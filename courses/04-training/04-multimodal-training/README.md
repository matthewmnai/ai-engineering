# 04 · 视觉 / 多模态模型训练

> 模块 4 · 模型训练与微调 ｜ 日期：08-27

## 环境标签

🖥️ **仅 AutoDL GPU**（必须有 CUDA 显卡，显存建议 ≥ 24G）

## 学习目标

- 理解多模态大模型的架构：视觉编码器 + 投影层 + LLM
- 掌握图文对齐训练与多模态指令微调
- 了解 CLIP、LLaVA 等主流多模态架构的设计思想
- 完成一个多模态微调实操

## 知识点

| # | 知识点 | 说明 |
|:-:|---|---|
| 1 | 多模态架构 | 视觉编码器（ViT）+ 模态投影层 + 语言模型 |
| 2 | CLIP 原理 | 对比学习，图文共享嵌入空间 |
| 3 | LLaVA 架构 | ViT + MLP projector + LLM，图文指令微调 |
| 4 | 图文对齐训练 | 预训练对齐阶段 + 指令微调阶段 |
| 5 | 多模态指令数据 | `(image, instruction, response)` 格式 |
| 6 | 多模态评测 | VQA、图像描述、图文匹配准确率 |

## 运行说明

```bash
bash setup.sh
python environment/verify_env.py

# 拉取多模态基座模型（视觉编码器 + LLM）
python environment/download_models.py --target autodl 04_training
```

- 图像样例放在 `datasets/course/04_training/`（大图不入库，迁移自动打包）
- 小样例可直接放本课程目录下（随 git 同步）
- 多模态微调脚本位于 `src/`
