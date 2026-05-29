# couple-Qwen-0.8b

一个面向课程/实验展示的中文对联生成仓库，保留了三条可对照的路线：

- `legacy baseline`：手写 `GRU / Attention / Transformer` 小模型
- `Qwen zero-shot`：直接用 `Qwen3-0.6B-Base` 做 prompt baseline
- `Qwen LoRA`：在过滤后的对联子集上做 LoRA 微调

这个仓库当前最适合展示的是：

1. 我把本地可跑的对联生成链路搭起来了
2. 预训练小模型 + LoRA 这条路线相对已有 baseline 有明确改进
3. 结果已经足够做“路线成立”的演示，但还不能包装成稳定高质量成品

## 效果展示

### 1. 真实 CLI 输出

以下结果于 `2026-05-29` 通过仓库内脚本实际生成：

```powershell
.venv\Scripts\python.exe src\hf\generate_hf_lora.py --device cpu --text 风弦未拨心先乱
# 月笛犹弹梦已空

.venv\Scripts\python.exe src\hf\generate_hf_lora.py --device cpu --text 花梦粘于春袖口
# 诗情融在柳眉梢

.venv\Scripts\python.exe src\hf\generate_hf_lora.py --device cpu --text 春风送暖
# 瑞雪迎春
```

### 2. 固定样例对照

来自 [`baseline_results/qwen3_lora_filtered_e4_vs_baselines.md`](baseline_results/qwen3_lora_filtered_e4_vs_baselines.md) 的 10 条固定 7 字样例对照中，`LoRA E4(best@epoch2)` 的整体表现比 `zero-shot Prompt C` 和当前 `attention baseline` 更贴近任务。下面摘两条最有代表性的样例：

| 上联 | Zero-shot Prompt C | LoRA E4(best@epoch2) | Attention baseline |
| --- | --- | --- | --- |
| 风弦未拨心先乱 | 雨露无声草自荣 | 月笛犹弹梦已空 | 竹影犹留客难眠 |
| 花梦粘于春袖口 | 月影藏于夜幕中 | 诗魂凝在柳眉梢 | 秋风入画图中天 |

更完整的展示见 [docs/showcase.md](docs/showcase.md)。

## 当前结论

- `Qwen + LoRA` 明显比 `zero-shot` 更少复读上联，更愿意输出“像下联”的句子。
- `Qwen + LoRA` 相比当前手写 `attention baseline`，更少出现完全无关的模板句。
- 现阶段仍然存在语义发散、对仗不够工整、局部模板化等问题。
- 因此这个项目当前最适合公开呈现为“路线已验证”的实验仓库，而不是“已经完成的高质量生成器”。

## 仓库结构

```text
src/
  legacy/
    couplet_dataset.py
    model.py
    train.py
    generate.py
  hf/
    hf_couplet_dataset.py
    train_hf_lora.py
    generate_hf_clm.py
    generate_hf_lora.py
  data_tools/
    filter_couplets.py

data/
  couplet/
    couplet/
      train/
      test/
  filtered_couplets/
    selection_summary.json
    top_selected_in.txt
    top_selected_out.txt

baseline_results/
  qwen3_0.6b_base_zero_shot_*.md
  qwen3_lora_filtered_e*_vs_baselines.md
  qwen3_lora_filtered_e4_showcase_notes.md

checkpoints/
  qwen3_0_6b_lora_filtered_e4/
    best_adapter/
    run_config.json

docs/
  showcase.md
```

说明：

- 公开仓库只保留了展示用的 `LoRA best adapter` 与必要实验材料。
- 其他本地训练中间产物、缓存、日志和旧 checkpoint 默认不提交。
- 第一次运行 `LoRA` 推理时仍需要从 Hugging Face 拉取底模 `Qwen/Qwen3-0.6B-Base`。

## 环境准备

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- `torch`
- `transformers`
- `accelerate`
- `peft`
- `pypinyin`

## 快速开始

### 1. 运行 LoRA 展示入口

默认使用当前表现最好的 adapter：

```powershell
.venv\Scripts\python.exe src\hf\generate_hf_lora.py --text 风弦未拨心先乱
```

进入交互模式：

```powershell
.venv\Scripts\python.exe src\hf\generate_hf_lora.py
```

### 2. 运行 zero-shot baseline

```powershell
.venv\Scripts\python.exe src\hf\generate_hf_clm.py --text 风弦未拨心先乱
```

### 3. 运行 legacy baseline

如果你想继续对照手写小模型路线：

```powershell
.venv\Scripts\python.exe src\legacy\generate.py --checkpoint checkpoints\couplet_gru.pth --text 春风送暖
```

## 数据过滤

`src/data_tools/filter_couplets.py` 会对原始训练集做规则打分与筛选，当前保留下来的结果摘要见 `data/filtered_couplets/selection_summary.json`。

当前这轮筛选的核心事实：

- 原始打分条目数：`770491`
- 最终选中：`10000`
- 长度配额：
  - `5` 字联：`3000`
  - `7` 字联：`5000`
  - `9/11` 字联：`2000`

重新生成过滤子集：

```powershell
.venv\Scripts\python.exe src\data_tools\filter_couplets.py --top-k 10000
```

## 训练说明

### LoRA smoke run

```powershell
.venv\Scripts\python.exe src\hf\train_hf_lora.py --limit 16 --epochs 1 --batch-size 1 --grad-accum 2 --max-length 96 --output-dir checkpoints\qwen3_0_6b_lora_smoke --val-ratio 0.125
```

### 当前展示用 adapter

- 目录：`checkpoints/qwen3_0_6b_lora_filtered_e4/best_adapter`
- 底模：`Qwen/Qwen3-0.6B-Base`
- 训练集：过滤后的 `10000` 条样本
- 最佳 checkpoint：`epoch 2`

## 建议阅读顺序

如果你想快速了解这个项目，建议按下面顺序看：

1. `README.md`
2. `docs/showcase.md`
3. `baseline_results/qwen3_lora_filtered_e4_showcase_notes.md`
4. `baseline_results/qwen3_lora_filtered_e4_vs_baselines.md`
5. `src/hf/generate_hf_lora.py`

## 后续可以继续做什么

- 提升过滤后的训练数据质量，而不是只继续加 epoch
- 增加更明确的对仗/平仄约束
- 把当前 CLI 展示进一步封装成更友好的小界面
- 对 `legacy` 路线和 `Qwen` 路线做更系统的人工评测
