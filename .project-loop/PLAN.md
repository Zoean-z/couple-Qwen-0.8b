# Working Plan

## Current Goal
- 重写公开仓库 README，补齐效果展示，整理公开提交边界，并推送到 `Zoean-z/couple-Qwen-0.8b`。

## Active Checklist
- [x] 读取 `AGENTS.md`、`README.md`、`PROJECT_STATUS.md`、`progress.md` 和当前目录结构
- [x] 确认远程仓库存在且当前账号具备 push 权限
- [x] 核对源码、结果文件、数据筛选产物和 checkpoint 体积
- [x] 重写 `README.md` 并新增 `docs/showcase.md`
- [x] 新增 `.gitignore`，收紧公开仓库边界
- [x] 更新 `PROJECT_STATUS.md` 与 `progress.md`
- [x] 初始化 Git、提交并推送到 `main`

## Decisions
- 公共仓库只保留源码、必要数据、baseline 结果和展示用最佳 `LoRA adapter`，不提交全部本地 checkpoint。
- README 必须如实呈现“路线有效但未到最终高质量成品”的状态。
- 最终效果展示以真实脚本输出和固定对照样例为主，不伪造前端 demo。

## Blockers
- 无。

## Next Step
- 等用户确认 README 呈现方式；若继续迭代，优先补更系统的人工评测与展示界面。
