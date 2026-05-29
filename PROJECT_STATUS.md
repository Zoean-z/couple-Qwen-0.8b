# Project Status

## Current Goal
- 维持当前已经公开推送的展示版本，并在此基础上继续做质量或展示层迭代。

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
- 已完成 Git 初始化、首个 commit 与远程推送：
  - `origin/main`

## Current Judgement
- 当前仓库已经适合公开展示“路线成立”。
- 当前仓库仍不适合宣传为“稳定高质量的最终对联生成系统”。

## Planned Next Step
- 如果继续提升项目完成度，优先补更系统的人工评测与更友好的展示入口。
