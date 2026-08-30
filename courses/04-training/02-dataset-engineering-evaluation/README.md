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

本目录下的 `setup.sh` 会自动完成两步初始化：

```bash
bash setup.sh
```

脚本内部流程：

| 步骤 | 动作 | 说明 |
|:-:|---|---|
| step1 | 调用项目根 `setup.sh` | 安装依赖（common + gpu）、配置 HF 镜像、创建资源目录、运行 `verify_env.py` 自检 |
| step2 | 下载本模块模型 | `python environment/download_models.py --target autodl 04_training`（已下载的模型会自动跳过） |

如需本模块数据集，单独执行（`setup.sh` 未包含）：

```bash
python environment/download_datasets.py --target autodl 04_training
```

### 目录结构

```
datasets/
├── public/           # 公开数据集（脚本下载，🟡 建议迁移）
└── course/
    └── 04_training/  # 课程大数据（🔴 必带迁移，pack_migration.sh 自动打包）

src/                  # 数据清洗脚本
outputs/              # 清洗产出（git 忽略）
```

- 课程数据放在 `datasets/course/04_training/`（跨机器迁移由 `pack_migration.sh` 自动打包）
- 小样例 json/txt 可直接放本课程目录下（随 git 同步）
- 数据清洗脚本位于 `src/`，清洗产出写入 `outputs/`
