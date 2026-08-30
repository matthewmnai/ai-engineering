# AI Engineering 学习工程

> 一个系统化学习 AI 工程化能力的实践仓库，覆盖大模型基础应用、训练微调、RAG 知识库、Agent 开发与工业级部署全链路。
> 按课程七大模块顺序组织，配套**双平台自动化开发环境**与**配置驱动的资源下载工具**。

---

## 📌 核心特点

| 特点 | 说明 |
|---|---|
| 🗂️ **模块驱动** | 目录即课程顺序，每个章节目录含 README + 练习代码 + 数据样例 |
| 🤖 **环境自动化** | `bash setup.sh` 自动检测 Mac/AutoDL 并路由，装对的依赖 |
| 🍎🐧 **双平台适配** | Mac 本地跑 API + 小 embedding；AutoDL GPU 跑训练 / 微调 / 大模型推理 |
| 📦 **配置驱动** | 模型 / 数据集清单在 `env_config.yaml` 标 `runtime` + `type`，download 脚本自动过滤 |
| 🔄 **ModelScope 优先** | `backend: "auto"` 优先从 ModelScope 下载，失败回退 HuggingFace（国内更快） |
| 🚚 **一键迁移** | `pack_migration.sh` 按优先级打包微调产出 + 课程数据，跨 AutoDL 机器无缝迁移 |
| 🔒 **大文件隔离** | `.gitignore` 覆盖权重 / 数据集 / 密钥 / 产出，不污染仓库 |
| 🧩 **公共能力复用** | 统一 LLM 客户端、配置加载、日志工具，各章节 import 即可用（规划中） |

---

## 🚀 快速开始

### Mac 本地（API 调用 + 本地小 embedding）

```bash
git clone <your-repo-url> ai-engineering && cd ai-engineering

# 一键部署 —— 自动检测 Mac，只装 common + mac 依赖（CPU 版 torch）
bash setup.sh          # 等价于 bash environment/setup-mac.sh

cp .env.example .env   # 填入 API Key

# （可选）只拉 Mac 需要的小 embedding 模型
python environment/download_models.py --target mac
```

### AutoDL / Linux GPU（全栈，含训练 / 微调）

```bash
# setup.sh 在 Linux 上自动路由到 setup-autodl.sh
bash setup.sh          # conda + common 依赖 + CUDA 版 torch + HF 镜像

cp .env.example .env

# 拉模型 / 数据集（默认按平台过滤，只拉 AutoDL 需要的）
python environment/download_models.py                              # 自动检测
python environment/download_models.py --target autodl 04_training   # 指定模块

# 全量下载（忽略 runtime 过滤）
python environment/download_models.py --target all
```

### 两种写法

| 命令 | 行为 |
|---|---|
| `bash setup.sh` | 自动检测平台，路由到对应 setup-*.sh |
| `bash setup.sh --target mac` | 强制 Mac 依赖 |
| `bash setup.sh --target autodl` | 强制 AutoDL 依赖 |
| `bash environment/setup-mac.sh` | 直接调用 Mac 版本 |
| `bash environment/setup-autodl.sh` | 直接调用 AutoDL 版本 |

---

## 🖥️ 课程模块 × 运行环境

| 模块 | 核心任务 | Mac | AutoDL GPU | 说明 |
|---|---|:-:|:-:|---|
| **模块 1** 初步接触 | Prompt / RAG / Agent 基础演示 | ✅ | ✅ | 全走 API |
| **模块 2** 基础入门 | API 调用 / AI Coding | ✅ | ✅ | 全走 API |
| **模块 3** 开发框架 | LangChain / HF / TensorFlow / PyTorch | ⚠️ | ✅ | TF/PyTorch 本地推理 Mac M 系列可勉强跑 |
| **模块 4** 训练微调 | LoRA / QLoRA / 蒸馏 / 多模态训练 | ❌ | ✅ | **必须 AutoDL GPU** |
| **模块 5** RAG 体系 | Embedding / 向量库 / RAG 调优 | ✅ | ✅ | Mac 本地跑小 embedding 模型 |
| **模块 6** Agent 高阶 | MCP / OpenManus / Hermes | ✅ | ✅ | 全走 API |
| **模块 7** 部署与高并发 | SGLang / 视频 AIGC | ❌ | ✅ | **必须 AutoDL GPU** |

> ✅ 推荐 · ⚠️ 可以但可能慢 · ❌ 不支持

---

## � 项目架构

```
ai-engineering/
├── .gitignore                          # ✅ 大文件 / 密钥 / Notebook 忽略规则
├── .env.example                        # ✅ 密钥模板（复制为 .env 后填入）
├── setup.sh                            # ✅ 根目录路由脚本 —— 自动检测平台
│
├── environment/                        # ✅ 已落地 —— 开发环境自动化
│   ├── setup-autodl.sh                 #   AutoDL/Linux GPU 环境（common + gpu）
│   ├── setup-mac.sh                    #   Mac 本地环境（common + mac）
│   ├── verify_env.py                   #   环境自检（必选依赖 / 可选依赖 / 镜像 / CUDA）
│   ├── download_models.py              #   批量拉模型（ModelScope优先 + --target 过滤）
│   ├── download_datasets.py            #   批量拉数据集（ModelScope优先 + --target 过滤）
│   ├── pack_migration.sh               #   跨 AutoDL 机器迁移打包脚本
│   ├── env_config.yaml                 #   统一配置（backend/runtime/type + 分平台路径）
│   ├── requirements-common.txt         #   两边通用轻量依赖（openai / langchain / pyyaml …）
│   ├── requirements-mac.txt            #   只 Mac：CPU torch + sentence-transformers
│   ├── requirements-gpu.txt            #   只 AutoDL：CUDA torch + peft + bitsandbytes …
│   └── requirements-modules/            #   可选重包（按需装，对应 courses/ 模块）
│       ├── 03-tensorflow.txt
│       └── 04-finetune.txt
│
├── models/                             # ✅ 资源目录（setup.sh 自动创建，不入 Git）
│   ├── pretrained/                     #   基座模型（大，可重新下载 🟢）
│   └── finetuned/                      #   微调产出（不可再生，迁移必带 🔴）
├── datasets/                           # ✅ 数据集目录（不入 Git，pack_migration.sh 打包）
│   ├── public/                         #   公开数据集（可重新下载，建议带 🟡）
│   └── course/                         #   课程专属数据（迁移必带 🔴）
│       └── 04_training/                #     模块4 课程数据
├── checkpoints/                        # ✅ 训练中间状态（断点续训，可选 🟢）
├── outputs/                           # ✅ 训练日志 / 评估结果（不迁移）
├── logs/                              # ✅ 运行日志
│
├── common/                             # 🛠️ 规划中 —— 公共工具库
│   ├── llm_client.py                   #   统一 LLM 调用客户端
│   ├── config.py                       #   配置加载（env_config.yaml + .env）
│   ├── logger.py
│   └── utils.py
│
├── courses/                            # 📚 课程模块（七大模块，按编号排序）
│   ├── 01-llm-basics/                  #   规划中
│   ├── 02-llm-foundation/              #   规划中
│   ├── 03-frameworks/                  #   规划中
│   ├── 04-training/                    #   规划中 ⭐ 必须 AutoDL
│   ├── 05-rag/                         #   规划中
│   ├── 06-agent/                       #   规划中
│   └── 07-deployment/                  #   规划中 ⭐ 必须 AutoDL
│
├── projects/                           # 🏭 规划中 —— 工业级综合项目
└── docs/                               # 📄 规划中
```

### 依赖分层

```
requirements-common.txt     # openai / langchain / huggingface_hub / 工具库 —— 两边都装
requirements-mac.txt         # CPU torch / transformers / sentence-transformers —— 只 Mac
requirements-gpu.txt         # CUDA torch / peft / trl / bitsandbytes —— 只 AutoDL
requirements-modules/*.txt   # tensorflow / unsloth / sglang —— 按需、只 AutoDL
```

### 章节内部统一结构

```
courses/NN-name/0X-chapter-name/
├── README.md          # 学习目标 + 知识点 + 运行说明 + 环境标签
├── src/               # 练习代码
└── outputs/           # 运行产出（git 忽略）

课程数据统一存放在 datasets/course/<module>/（脚本下载/手动放入，pack_migration.sh 自动打包）
小数据需入 Git 时直接放对应 courses/ 课程目录下
```

---

## 📚 完整课程目录（七大模块）

### 模块 1 · 初步接触 🖥️ Mac / AutoDL

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | 开班典礼 | 07-09 |
| 2 | 提示工程 + RAG：大模型知识交互底层能力 | 07-10 |
| 3 | Agent 基础：可控逻辑、自主反思机制 | 07-13 |
| 4 | 多模态前沿：Agent 拓展 + 视频 AIGC | 07-16 |

### 模块 2 · 基础入门 🖥️ Mac / AutoDL

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | AI 大模型原理 + API 实战调用 | 07-21 |
| 2 | AI 编程全流程实操 | 07-23 |

### 模块 3 · 开发框架 🖥️ AutoDL（Mac 可跑前 3 节）

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | LangChain：多任务应用开发 | 07-27 |
| 2 | AI 框架选型与架构设计 | 07-30 |
| 3 | HuggingFace 生态：模型调用 + 高效微调实战 | 08-03 |
| 4 | 神经网络基础 + TensorFlow 实战 | 08-06 |
| 5 | Pytorch + 视觉检测项目 | 08-10 |
| 6 | 框架方向简历、面试专项辅导 | 08-13 |

### 模块 4 · 模型训练与微调 🖥️ **仅 AutoDL**

| # | 课程 | 日期 |
|:-:|---|:-:|
| 1 | LLM 微调底层原理 | 08-17 |
| 2 | 高质量微调数据集工程 + 模型效果评估 | 08-20 |
| 3 | LLM 模型蒸馏 + 微调实操落地 | 08-24 |
| 4 | 视觉 / 多模态模型训练 | 08-27 |
| 5 | AI 视觉质检工业项目 | 08-31 |
| 6 | 训练微调方向简历面试辅导 | 09-03 |

### 模块 5 · 知识库与 RAG 体系 🖥️ Mac / AutoDL

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

### 模块 6 · AI Coding + Agent 深度实战 🖥️ Mac / AutoDL

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

### 模块 7 · 模型部署与高并发 🖥️ **仅 AutoDL**

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
模块7 部署上线  ←──  模块6 Agent高阶  ←──  模块4 训练微调  (🖥️ 仅 AutoDL)
                                              │
                                              ▼
                                        模块5 RAG体系
```

---

## 📁 environment/ 目录详解

| 文件 | 职责 | 用法 |
|---|---|---|
| [setup.sh](setup.sh) | 根目录路由脚本，自动检测平台 | `bash setup.sh` 或 `bash setup.sh --target mac` |
| [setup-autodl.sh](environment/setup-autodl.sh) | AutoDL GPU：common + gpu 依赖 + HF 镜像 | `bash environment/setup-autodl.sh` |
| [setup-mac.sh](environment/setup-mac.sh) | Mac：common + mac 依赖（CPU torch） | `bash environment/setup-mac.sh` |
| [verify_env.py](environment/verify_env.py) | 环境自检：必选/可选依赖、镜像、CUDA | `python environment/verify_env.py` |
| [download_models.py](environment/download_models.py) | ModelScope 优先批量拉模型，按 type 分流到 pretrained/finetuned | `--target mac\|autodl\|all` |
| [download_datasets.py](environment/download_datasets.py) | ModelScope 优先批量拉数据集，按 type 分流到 public/course | 同上 |
| [pack_migration.sh](environment/pack_migration.sh) | 跨 AutoDL 机器迁移打包 | `--level must\|should\|all` |
| [env_config.yaml](environment/env_config.yaml) | 统一配置：backend/runtime/type + 分平台路径 | 编辑此文件 |
| requirements-*.txt | common / mac / gpu 三层依赖 | 分别被对应 setup 脚本安装 |

### env_config.yaml 示例

```yaml
# 后端策略：优先 ModelScope，失败回退 HuggingFace
env:
  preferred_backend: "modelscope"

models:
  05_rag:
    - name: "BAAI/bge-large-zh-v1.5"
      backend: "auto"          # 用 env.preferred_backend 策略
      runtime: "both"          # Mac 本地 + AutoDL 训练都拉
      type: "pretrained"       # 存到 models/pretrained/
    - name: "Qwen/Qwen2.5-7B-Instruct"
      backend: "auto"
      runtime: "autodl"        # 只有 AutoDL 拉（大模型）
      type: "pretrained"

datasets:
  05_rag:
    - name: "course_rag_sample"
      type: "course"            # 课程专属数据，存到 datasets/course/
      runtime: "both"

# 路径按平台 + 用途分
paths:
  autodl:
    models_pretrained: "/root/autodl-tmp/ai-engineering/models/pretrained"
    models_finetuned: "/root/autodl-tmp/ai-engineering/models/finetuned"
    datasets_public: "/root/autodl-tmp/ai-engineering/datasets/public"
    datasets_course: "/root/autodl-tmp/ai-engineering/datasets/course"
  mac:
    models_pretrained: "models/pretrained"
    models_finetuned: "models/finetuned"
    datasets_public: "datasets/public"
    datasets_course: "datasets/course"
```

---

## � 资源迁移（跨 AutoDL 机器）

AutoDL 机器按时计费，换机器时需要迁移微调产出和课程数据：

```bash
# ---------- 旧机器上打包 ----------
bash environment/pack_migration.sh                    # 默认: must + should
bash environment/pack_migration.sh --level must       # 只打必带（微调产出 + 课程数据）
bash environment/pack_migration.sh --level all        # 全量（含基座模型，很慢）

# ---------- 新机器上解包 ----------
cd /root/autodl-tmp/ai-engineering
tar xzf ai-migration-*.tar.gz
bash setup.sh                                          # 补环境
python environment/download_models.py --target autodl  # 补缺失的基座模型
```

迁移优先级：

| 优先级 | 目录 | 说明 |
|---|---|---|
| 🔴 必带 | `models/finetuned/` | 微调产出，不可重新下载 |
| 🔴 必带 | `datasets/course/` | 课程专属数据，可能无法重新下载 |
| 🟡 建议带 | `datasets/public/` | 可重新下载但省时间 |
| 🟢 可不带 | `models/pretrained/` | 大，`download_models.py` 可重新拉 |
| 🟢 可不带 | `checkpoints/` | 除非要断点续训 |

---

## �� 工程约定

- **大文件不进 Git**：权重、数据集、密钥、产出均在 `.gitignore` 中
- **配置驱动**：`env_config.yaml` 是资源清单单一真源，增删模型改 YAML 重跑 download 脚本
- **ModelScope 优先**：`backend: "auto"` 优先 ModelScope 下载，失败回退 HuggingFace
- **双平台自动过滤**：download 脚本按 `runtime` 字段只拉当前平台需要的资源
- **按用途分流**：模型分 pretrained/finetuned，数据集分 public/course，迁移优先级清晰
- **编号排序**：模块 / 章节目录统一 `0X-` 前缀，文件管理器天然按学习顺序排列
- **重包隔离**：`bitsandbytes` / `trl` 等只 AutoDL 装，Mac 绝不会误装 CUDA torch
- **资源落数据盘**：AutoDL 用绝对路径 `/root/autodl-tmp/`，Mac 用项目内相对路径

---

## 📄 License

本项目用于个人学习，代码可自由使用。所涉及的第三方模型、数据集请遵循各自 License。
