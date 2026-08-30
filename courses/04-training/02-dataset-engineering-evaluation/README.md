# 02 · 高质量微调数据集工程 + 模型效果评估

> 模块 4 · 模型训练与微调 ｜ 日期：08-20

## 环境标签

🖥️ **仅 AutoDL GPU**（必须有 CUDA 显卡）

## 学习目标

- 掌握高质量微调数据集的构建、清洗与格式化全流程
- 理解 SFT、DPO、RLHF 不同阶段的数据格式差异
- 学会用客观指标 + 主观评测综合评估微调效果
- 建立数据质量驱动的微调迭代闭环

## 知识点

| # | 知识点 | 说明 |
|:-:|---|---|
| 1 | SFT 数据格式 | `instruction / input / output` 三元组，指令监督 |
| 2 | DPO 数据格式 | `(prompt, chosen, rejected)` 偏好对齐 |
| 3 | 数据清洗工程 | 去重、去噪、长度过滤、敏感词过滤 |
| 4 | 数据配比与课程学习 | 多源数据配比、由易到难的训练策略 |
| 5 | 客观评估指标 | Perplexity、BLEU、ROUGE、准确率 |
| 6 | 主观评估 | 人工评分、模型互评（LLM-as-a-Judge） |

## 运行说明

```bash
bash setup.sh
python environment/verify_env.py

# 拉取本模块所需数据集（课程专属 + 公开数据集）
python environment/download_datasets.py --target autodl 04_training
```

- 样例数据放在 `data/`，大文件不入库
- 数据清洗脚本位于 `src/`，清洗产出写入 `outputs/`
