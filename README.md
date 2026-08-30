# AI Engineering 学习工程

> 一个系统化学习 AI 工程化能力的实践仓库，覆盖大模型基础应用、训练微调、RAG 知识库、Agent 开发与工业级部署全链路。
> 按课程七大模块顺序组织，配套**自动化开发环境**与**资源下载工具**。

---

## 📌 核心特点

| 特点 | 说明 |
|---|---|
| 🗂️ **模块驱动** | 目录即课程顺序，每个章节目录含 README + 练习代码 + 数据样例 |
| 🤖 **环境自动化** | `bash setup.sh` 一键装依赖、配镜像、拉模型、下数据集 |
| 📦 **配置驱动** | 模型 / 数据集清单集中在 `environment/env_config.yaml`，增删改只动 YAML |
| 🔒 **大文件隔离** | `.gitignore` 已覆盖权重 / 数据集 / 密钥 / 产出，不污染仓库 |
| 🧩 **公共能力复用** | 统一 LLM 客户端、配置加载、日志工具，各章节 import 即可用（规划中） |

---

## 🚀 快速开始

```bash
# 1. 克隆
git clone <your-repo-url> ai-engineering && cd ai-engineering

# 2. 一键部署环境（conda + pip + HF 镜像 + 自检）
bash setup.sh

# 3. 填入密钥
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 等（已在 .gitignore 中，不会入库）

# 4. 拉模型 / 数据集（支持按模块分组，避免一次拉全量）
python environment/download_models.py          # 全量
python environment/download_models.py module5_rag   # 只拉 RAG 模块所需

# 5. 开始学习：进入对应 moduleX/0X-章节/，按其 README 运行
```

### 环境依赖

| 项目 | 值 |
|---|---|
| Python | 3.10（conda 环境名 `ai-engineering`） |
| pip 源 | 清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple` |
| HF 镜像 | `https://hf-mirror.com`（国内加速，setup.sh 自动持久化到 `~/.bashrc`） |
| 资源盘 | `/root/autodl-tmp/ai-engineering/`（AutoDL 数据盘，避免系统盘写满） |

### 依赖分层

```
requirements-base.txt          # 基础依赖，setup.sh 默认安装
requirements-modules/*.txt     # 可选重包（tensorflow / unsloth / sglang …），按需装
```

---

## 📂 项目架构

```
ai-engineering/
├── .gitignore                          # ✅ 大文件 / 密钥 / Notebook 忽略规则
├── .env.example                        # ✅ 密钥模板（复制为 .env 后填入真实值）
├── setup.sh                            # ✅ 一键部署软链接 → environment/setup.sh
│
├── environment/                        # ✅ 已落地 —— 开发环境自动化
│   ├── setup.sh                        #   conda + pip + HF 镜像 + 自检
│   ├── verify_env.py                   #   环境自检（必选依赖 / 可选依赖 / CUDA）
│   ├── download_models.py              #   配置驱动批量拉模型（HF / ModelScope）
│   ├── download_datasets.py            #   配置驱动批量拉数据集
│   ├── env_config.yaml                 #   模型 / 数据集 / 路径 统一配置入口
│   ├── requirements-base.txt           #   基础依赖（PyTorch / HF / LangChain …）
│   └── requirements-modules/            #   可选重包（按需安装）
│       ├── module3-tensorflow.txt
│       └── module4-finetune.txt
│
├── common/                             # 🛠️ 规划中 —— 公共工具库（各模块 import 复用）
│   ├── llm_client.py                   #   统一 LLM 调用（兼容 OpenAI / DeepSeek / Qwen …）
│   ├── config.py                       #   配置加载（env_config.yaml + .env）
│   ├── logger.py                       #   日志工具
│   └── utils.py                        #   通用函数
│
├── module1-llm-basics/                 # 📚 规划中 —— 模块1：初步接触
├── module2-llm-foundation/             # 📚 规划中 —— 模块2：基础入门
├── module3-frameworks/                 # 📚 规划中 —— 模块3：开发框架
├── module4-training/                   # 📚 规划中 —— 模块4：训练与微调 ⭐核心重难点
├── module5-rag/                        # 📚 规划中 —— 模块5：知识库与RAG体系 ⭐企业落地
├── module6-agent/                      # 📚 规划中 —— 模块6：AI Coding + Agent 深度实战
├── module7-deployment/                 # 📚 规划中 —— 模块7：模型部署与高并发
│
├── projects/                           # 🏭 规划中 —— 工业级综合项目
│   ├── visual-quality-inspection/      #   AI 视觉质检工业项目
│   └── rag-champion/                   #   RAG 大赛冠军项目拆解
│
├── assets/                             # 🗂️ 规划中 —— 入库的小样例数据
├── docs/                               # 📖 规划中 —— 文档与学习路线图
└── scripts/                            # 🔧 规划中 —— 辅助脚本
```

### 章节内部统一结构（模板）

每个 `moduleX/0X-chapter-name/` 内部保持一致，便于检索和自动化：

```
01-prompt-engineering/
├── README.md          # 本章学习目标 + 知识点 + 运行说明
├── src/               # 练习代码（可独立运行）
├── data/              # 本章小样例数据（大文件不入库）
└── outputs/           # 运行产出（git 忽略）
```

---

## 📚 完整课程目录（七大模块）

> 按课程学习顺序排列。模块 4 为**核心重难点**，模块 5 为**企业落地核心技术**，模块 6 为**高阶应用**，模块 7 为**工业上线**。

### 模块 1 · 初步接触（大模型基础应用入门）

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | 开班典礼 | 07-09 |
| 2 | 提示工程 + RAG：大模型知识交互底层能力 | 07-10 |
| 3 | Agent 基础：可控逻辑、自主反思机制 | 07-13 |
| 4 | 多模态前沿：Agent 拓展 + 视频 AIGC | 07-16 |

### 模块 2 · 基础入门（LLM 基础 + AI 编程）

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | AI 大模型原理 + API 实战调用 | 07-21 |
| 2 | AI 编程全流程实操 | 07-23 |

### 模块 3 · 开发框架（主流 LLM 工程框架）

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | LangChain：多任务应用开发 | 07-27 |
| 2 | AI 框架选型与架构设计 | 07-30 |
| 3 | HuggingFace 生态：模型调用 + 高效微调实战 | 08-03 |
| 4 | 神经网络基础 + TensorFlow 实战 | 08-06 |
| 5 | Pytorch + 视觉检测项目 | 08-10 |
| 6 | 框架方向简历、面试专项辅导 | 08-13 |

### 模块 4 · 模型训练与微调（核心重难点，匹配预训练 + 微调范式）

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | LLM 微调底层原理 | 08-17 |
| 2 | 高质量微调数据集工程 + 模型效果评估 | 08-20 |
| 3 | LLM 模型蒸馏 + 微调实操落地 | 08-24 |
| 4 | 视觉 / 多模态模型训练 | 08-27 |
| 5 | AI 视觉质检工业项目 | 08-31 |
| 6 | 训练微调方向简历面试辅导 | 09-03 |

### 模块 5 · 知识库与 RAG 体系（企业落地核心技术）

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | Embedding 向量 + 向量数据库原理 | 09-07 |
| 2 | RAG 完整技术链路与业务落地 | 09-11 |
| 3 | 多模态 RAG 图文音视频处理 | 09-14 |
| 4 | RAG 全链路调优方案 | 09-17 |
| 5 | 企业级知识库（RAG 大赛冠军项目拆解） | 09-21 |
| 6 | 可替代 RAG 的技术方案对比 | 09-24 |
| 7 | LLM 知识库 Wiki 搭建 | 09-28 |
| 8 | RAG 岗位简历面试辅导 | 10-01 |

### 模块 6 · AI Coding + Agent 深度实战（高阶应用）

#### 6.1 AI Coding 单元

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | 大厂工程师 AI 编程工作流 | 10-05 |
| 2 | 大型软件 AI 开发、代码重构方案 | 10-08 |
| 3 | AI 时代团队分工、全新协作模式 | 10-12 |
| 4 | 昇腾硬件部署 DeepSeek V4，对接本地 Claude Code | 10-15 |

#### 6.2 Agent 高阶开发单元

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | Function Calling、MCP 协议原理 | 10-19 |
| 2 | Agent 自主规划、自定义工具开发 | 10-22 |
| 3 | Agent 性能优化、量化评估体系 | 10-26 |
| 4 | OpenManus 智能体完整实战 | 10-29 |
| 5 | Harness 工程化体系 | 11-02 |
| 6 | Hermes Agent 长期记忆、自进化能力搭建 | 11-05 |
| 7 | Hermes 多 Agent 协作、主 Agent 调度系统 | 11-09 |
| 8 | Agent 岗位简历面试辅导 | 11-12 |

### 模块 7 · 模型部署与高并发（工业上线）

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | 企业级 AI 部署：硬件选型、推理框架对比 | 11-16 |
| 2 | AI 服务高并发底层原理、性能监控调优 | 11-19 |
| 3 | SGLang 深度优化：Radix 缓存、高吞吐复杂任务 | 11-23 |
| 4 | 视频 AIGC（短剧换脸）显卡资源分配、任务排队系统 | 11-26 |

---

## 🗺️ 学习路线拓扑

```
模块1 初步接触  ──→  模块2 基础入门  ──→  模块3 开发框架
                                              │
                                              ▼
模块7 部署上线  ←──  模块6 Agent高阶  ←──  模块4 训练微调
                                              │
                                              ▼
                                        模块5 RAG体系
```

---

## 📁 environment/ 目录详解

| 文件 | 职责 | 用法 |
|---|---|---|
| [setup.sh](environment/setup.sh) | 一键部署：conda 环境 + pip 依赖 + HF 镜像 + 自检 | `bash setup.sh` |
| [verify_env.py](environment/verify_env.py) | 环境自检：必选依赖 / 可选依赖 / 镜像 / CUDA | `python environment/verify_env.py` |
| [download_models.py](environment/download_models.py) | 配置驱动批量拉模型，支持按模块分组 | `python environment/download_models.py [模块key...]` |
| [download_datasets.py](environment/download_datasets.py) | 配置驱动批量拉数据集，本地样例自动跳过 | `python environment/download_datasets.py [模块key...]` |
| [env_config.yaml](environment/env_config.yaml) | 模型 / 数据集 / 资源路径 统一配置入口 | 编辑此文件增删资源 |
| [requirements-base.txt](environment/requirements-base.txt) | 基础依赖清单（PyTorch / HF / LangChain 等） | setup.sh 默认安装 |
| [requirements-modules/*.txt](environment/requirements-modules/) | 可选重包（TensorFlow / unsloth / sglang 等） | 按需 `pip install -r` |

### env_config.yaml 示例（部分）

```yaml
models:
  module4_training:
    - name: "Qwen/Qwen2.5-1.5B-Instruct"
      backend: "huggingface"
      usage: "LoRA/QLoRA微调练习(小卡可跑)"
  module5_rag:
    - name: "BAAI/bge-large-zh-v1.5"
      backend: "huggingface"
      usage: "Embedding向量化"

paths:
  model_cache: "/root/autodl-tmp/ai-engineering/hf_cache"
  data_root:   "/root/autodl-tmp/ai-engineering/data"
```

增删模型 / 数据集 → 改 YAML → 重新跑 download 脚本即可。

---

## 📌 工程约定

- **大文件不进 Git**：权重（`*.safetensors / *.bin / *.gguf`）、数据集（`*.parquet`）、密钥（`.env`）、产出（`outputs/`）均在 `.gitignore` 中。
- **配置驱动**：`environment/env_config.yaml` 是资源清单单一真源。
- **编号排序**：模块 / 章节目录统一 `0X-` 前缀，文件管理器天然按学习顺序排列。
- **重包隔离**：体积大且非每章必用的依赖（tensorflow / unsloth / sglang）放 `requirements-modules/`，按需安装。
- **资源落数据盘**：AutoDL 系统盘有限，缓存 / 数据集 / 日志全部落 `/root/autodl-tmp/`。

---

## 📄 License

本项目用于个人学习，代码可自由使用。所涉及的第三方模型、数据集请遵循各自 License。
