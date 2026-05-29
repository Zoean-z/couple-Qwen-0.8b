# Project Status

## Current Goal
- 把当前 `LoRA` 展示版本整理成适合公开仓库展示和远程提交的形态。

## Current State
- 已确认当前最适合展示的入口是 `src/hf/generate_hf_lora.py`。
- 已确认当前推荐展示用 adapter：
  - `checkpoints/qwen3_0_6b_lora_filtered_e4/best_adapter`
- 已补齐公开展示文档：
  - `README.md`
  - `docs/showcase.md`
  - `baseline_results/qwen3_lora_filtered_e4_showcase_notes.md`
- 已明确公共仓库边界：
  - 保留源码、必要数据、baseline 结果、展示用 adapter
  - 不提交训练日志、缓存、无关中间 checkpoint、大体积筛选中间表

## Current Judgement
- 当前仓库已经适合公开展示“路线成立”。
- 当前仓库仍不适合宣传为“稳定高质量的最终对联生成系统”。

## Planned Next Step
- 初始化 Git，完成首个公开提交并推送到远程 `main`。
