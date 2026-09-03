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

## Demo 脚本速览

> 三个 demo 均基于 [Unsloth](https://github.com/unslothai/unsloth) 框架，需要 AutoDL GPU 环境。
> 运行前确保基座模型已通过 `download_models.py` 下载到本地。

| 脚本 | 任务类型 | 基座模型 | 数据集 | LoRA Rank | 输出目录 |
|---|---|---|---|---|---|
| `demo/Qwen2_5_(7B)_Alpaca.py` | SFT 监督微调 | Qwen2.5-7B-Instruct | alpaca-cleaned | 16 | `outputs/sft_train/` |
| `demo/Qwen2_5_(7B)_R1_GRPO.py` | GRPO 强化学习 | Qwen2.5-7B-Instruct | gsm8k | 32 | `outputs/grpo_train/` |
| `demo/qwen-vl1/qwen_vl_car_insurance_train.py` | 视觉模型 SFT | Qwen2.5-VL-3B-Instruct | `qwen-vl-train.xlsx` | 16 | `outputs/vlm_train/` |

### 1. Alpaca SFT 微调

```bash
cd demo
python Qwen2_5_\(7B\)_Alpaca.py
```

- **核心功能**：对 Qwen2.5-7B-Instruct 做 LoRA 监督微调，使用 Alpaca-Cleaned 指令跟随数据
- **训练方式**：Unsloth + `trl.SFTTrainer`，4bit 量化 + gradient checkpointing
- **奖励/评估**：训练前后各做一次推理验证（Fibonacci 数列、巴黎地标）
- **产出**：LoRA 适配器 + tokenizer 保存到 `outputs/sft_train/lora_model/`

### 2. GRPO 强化学习（R1 风格）

```bash
cd demo
python Qwen2_5_\(7B\)_R1_GRPO.py
```

- **核心功能**：用 GRPO（Group Relative Policy Optimization）训练模型的数学推理能力，输出 XML 格式的 `<reasoning>` / `<answer>`
- **奖励函数组合**（5 个）：
  - `correctness_reward_func` — 答案正确性（权重最高，2.0）
  - `xmlcount_reward_func` — XML 标签完整性
  - `soft_format_reward_func` / `strict_format_reward_func` — 格式合规
  - `int_reward_func` — 答案是否为整数
- **训练方式**：Unsloth + `trl.GRPOTrainer`，vLLM 快速推理 (`fast_inference=True`)
- **产出**：LoRA 适配器保存到 `outputs/grpo_train/lora_model/`

### 3. 视觉模型（Qwen2.5-VL-3B）汽车保险承保微调

```bash
cd demo/qwen-vl1
python qwen_vl_car_insurance_train.py
```

- **核心功能**：训练多模态模型识别车辆里程表图片，用于汽车保险承保场景
- **训练范围**：视觉层 + 语言层 + 注意力 + MLP **全部**参与 LoRA 微调（`finetune_*=True`）
- **数据格式**：Excel 文件（`qwen-vl-train.xlsx`），列包含 `image` / `prompt` / `response`，图片需本地可读
- **训练方式**：Unsloth `FastVisionModel` + `trl.SFTTrainer` + `UnslothVisionDataCollator`
- **注意**：脚本内置 `os.chdir(os.path.dirname(__file__))`，需在文件所在目录运行
- **产出**：LoRA 适配器 + tokenizer 保存到 `outputs/vlm_train/lora_model/`

### 路径修复说明

早期版本的 demo 脚本在计算项目根路径时写的是 `parents[4]`，但从 `demo/qwen-vl1/` 到项目根实际需要 `parents[5]`。最新版本已修正为 `parents[5]`（见 `qwen_vl_car_insurance_train.py` 第 12 行）。`demo/` 下直接放置的脚本（Alpaca / R1_GRPO）目录层级少一层，保持 `parents[4]` 即可。
