# Progress

## 2026-05-25 - Writing Angle Planning

- Updated `.project-loop/PLAN.md` for the current writing discussion.
- No code, model, data, interface, or training files were changed.

### decision_audit

1. 本阶段改了哪些文件
   - `.project-loop/PLAN.md`
   - `progress.md`
2. 每个文件改动对应规划中的哪一条
   - `.project-loop/PLAN.md`: 对应 AGENTS.md 的 Planning memory 规则，给出文章结构规划时同步持久化短计划。
   - `progress.md`: 对应 AGENTS.md 的每阶段结束后强制输出规则，追加本阶段审计记录。
3. 哪些地方有实现权衡
   - 本次讨论不属于原模型训练任务，因此只记录写作任务的最小计划，不修改训练相关文件。
4. 是否存在潜在偏离
   - 无代码偏离；存在任务焦点从模型训练转向写作讨论的变化，已在计划中显式记录。
5. 下一阶段准备做什么
   - 如果用户确认角度，继续扩展为公众号长文草稿。

## 2026-05-25 - Article Draft Created

- Created `.project-loop/openai_unit_distance_ai_speed_article.md`.
- Updated `.project-loop/PLAN.md` to mark the article structure and draft complete.
- Verified the article file can be read as UTF-8 and contains about 5708 characters.

### decision_audit

1. 本阶段改了哪些文件
   - `.project-loop/openai_unit_distance_ai_speed_article.md`
   - `.project-loop/PLAN.md`
   - `progress.md`
2. 每个文件改动对应规划中的哪一条
   - `.project-loop/openai_unit_distance_ai_speed_article.md`: 对应用户要求，在指定 `.project-loop` 路径下用写作 skill 产出文章草稿。
   - `.project-loop/PLAN.md`: 对应 AGENTS.md 的 Planning memory 规则，更新写作任务状态和下一步。
   - `progress.md`: 对应 AGENTS.md 的每阶段结束后强制输出规则，追加本阶段审计记录。
3. 哪些地方有实现权衡
   - 没有博士生视频原文，因此文章只概括“有数学博士生表达怀疑”，没有编造具体台词。
   - OpenAI 结果表述保留 caveat，避免写成已完成长期同行评审的定论。
4. 是否存在潜在偏离
   - 无代码偏离；文章文件按用户指定路径创建。
5. 下一阶段准备做什么
   - 根据用户反馈补视频细节、调整标题、压缩或加强公众号发布感。

## 2026-05-25 - Article Ending Refocused

- Revised the latter part of `.project-loop/openai_unit_distance_ai_speed_article.md`.
- Updated `.project-loop/PLAN.md` to record the new thematic decision.
- Verified the revised article reads back as UTF-8 and is about 7032 characters.

### decision_audit

1. 本阶段改了哪些文件
   - `.project-loop/openai_unit_distance_ai_speed_article.md`
   - `.project-loop/PLAN.md`
   - `progress.md`
2. 每个文件改动对应规划中的哪一条
   - `.project-loop/openai_unit_distance_ai_speed_article.md`: 对应用户反馈，重写后半段情绪和主题落点。
   - `.project-loop/PLAN.md`: 对应 AGENTS.md 的 Planning memory 规则，记录主题从“重新学习”调整为“技能可补，特质更重要”。
   - `progress.md`: 对应 AGENTS.md 的每阶段结束后强制输出规则，追加本阶段审计记录。
3. 哪些地方有实现权衡
   - 保留前半段数学事件和普遍焦虑铺垫，只局部改后半段，避免整篇结构被打散。
   - 没有把“特质”写成纯鸡汤，而是和技能、培训、AI 补齐能力形成对照。
4. 是否存在潜在偏离
   - 无代码偏离；写作文件仍在用户指定目录。
5. 下一阶段准备做什么
   - 如果继续修改，优先加强开头和中段，让“技能/特质”的伏笔更早出现。

## 2026-05-25 - Article Middle Compressed

- Compressed the section from “真正吓人的是时间” to “所以我觉得那个数学博士生的视频很值得被认真对待” in `.project-loop/openai_unit_distance_ai_speed_article.md`.
- Updated `.project-loop/PLAN.md` to record this editing step.
- Verified the revised article reads back as UTF-8 and is about 4694 characters.

### decision_audit

1. 本阶段改了哪些文件
   - `.project-loop/openai_unit_distance_ai_speed_article.md`
   - `.project-loop/PLAN.md`
   - `progress.md`
2. 每个文件改动对应规划中的哪一条
   - `.project-loop/openai_unit_distance_ai_speed_article.md`: 对应用户反馈，压缩指定中段，减少重复铺陈。
   - `.project-loop/PLAN.md`: 对应 AGENTS.md 的 Planning memory 规则，记录中段压缩完成。
   - `progress.md`: 对应 AGENTS.md 的每阶段结束后强制输出规则，追加本阶段审计记录。
3. 哪些地方有实现权衡
   - 删除了大量职业类比和学习论述，只保留程序员、设计师、学生三个代表，避免过长。
   - 提前保留“技能可补，特质更重要”的转折，保证后文主题能接上。
4. 是否存在潜在偏离
   - 无代码偏离；写作文件仍在用户指定目录。
5. 下一阶段准备做什么
   - 若继续精修，优先检查后半段是否因中段压缩后出现观点重复。

## 2026-05-25 - Article Depth Revision

- Rewrote `.project-loop/openai_unit_distance_ai_speed_article.md` according to four editorial notes.
- Updated `.project-loop/PLAN.md` to record the revised ending and deeper OpenAI-event framing.
- Verified the revised article reads back as UTF-8 and is about 3865 characters.

### decision_audit

1. 本阶段改了哪些文件
   - `.project-loop/openai_unit_distance_ai_speed_article.md`
   - `.project-loop/PLAN.md`
   - `progress.md`
2. 每个文件改动对应规划中的哪一条
   - `.project-loop/openai_unit_distance_ai_speed_article.md`: 对应用户四条修改意见，重写 OpenAI 事件消化段、中段扩散和结尾。
   - `.project-loop/PLAN.md`: 对应 AGENTS.md 的 Planning memory 规则，记录新的结尾决策和本轮深修完成。
   - `progress.md`: 对应 AGENTS.md 的每阶段结束后强制输出规则，追加本阶段审计记录。
3. 哪些地方有实现权衡
   - 保留 OpenAI 事件 caveat，同时新增“工具延伸”和“发现动作”之间的差异，避免把论证建在未消化新闻上。
   - 职业扩散只保留程序员一个稍深入的例子，避免走马观花。
   - 合并“技能可补”的重复表达，把结尾篇幅让给数学博士生的具体处境。
4. 是否存在潜在偏离
   - 无代码偏离；写作文件仍在用户指定目录。
5. 下一阶段准备做什么
   - 根据用户反馈继续调整标题、开头力度或发布版语气。

## 2026-05-29 - README Rewrite And Public Repo Packaging

- Rewrote `README.md` around the current `legacy baseline / zero-shot / LoRA` project reality instead of the older rough notes.
- Added `docs/showcase.md` so the final effect is visible as a standalone artifact with real CLI outputs and fixed-sample comparisons.
- Added `.gitignore` to keep the public repo focused on source code, necessary data, baseline result notes, and the showcase LoRA adapter.
- Updated `.project-loop/PLAN.md` and `PROJECT_STATUS.md` to switch the repo focus back from the unrelated writing task to public packaging and push prep.
- Verified actual demo outputs by running `src/hf/generate_hf_lora.py` and `src/legacy/generate.py` locally.

### decision_audit

1. 本阶段改了哪些文件
   - `README.md`
   - `docs/showcase.md`
   - `.gitignore`
   - `.project-loop/PLAN.md`
   - `PROJECT_STATUS.md`
   - `progress.md`
2. 每个文件改动对应规划中的哪一条
   - `README.md`：对应用户“编写合适的 readme，记得把最终效果展示出来”。
   - `docs/showcase.md`：对应用户“把最终效果展示出来”，把真实脚本输出和固定对照样例单独落盘。
   - `.gitignore`：对应用户“整理项目结构提交到远程仓库”，收紧公开仓库边界，避免把缓存、日志和全部 checkpoint 一起提交。
   - `.project-loop/PLAN.md`：对应 AGENTS.md 的 planning memory 规则，把当前任务锚定为 README/整理/提交。
   - `PROJECT_STATUS.md`：对应 AGENTS.md 中“任务焦点显著变化时刷新状态”。
   - `progress.md`：对应 AGENTS.md 的每阶段结束后强制输出规则。
3. 哪些地方有实现权衡
   - 没有移动或删除大量本地实验文件，而是通过 `.gitignore` 只公开最适合展示的那一部分，避免越权整理。
   - README 没把当前结果写成“高质量成品”，而是明确限制在“路线有效”的展示口径内。
   - 效果展示采用真实 CLI 输出和已有固定对照样例，没有额外伪造前端页面或图片素材。
4. 是否存在潜在偏离
   - 无 `unmapped_change`。
   - 潜在风险只剩 GitHub 认证或远程推送阶段的大文件规则，但这属于提交阶段风险，不是实现方向偏离。
5. 下一阶段准备做什么
   - 初始化 Git 索引、检查待提交文件范围、提交并推送到远程 `main`。

## 2026-05-29 - Git Init And Remote Push

- Initialized the local Git repository on `main`.
- Added remote `origin` -> `https://github.com/Zoean-z/couple-Qwen-0.8b.git`.
- Created the first public commit and pushed it to `origin/main`.

### decision_audit

1. 本阶段改了哪些文件
   - `.project-loop/PLAN.md`
   - `PROJECT_STATUS.md`
   - `progress.md`
2. 每个文件改动对应规划中的哪一条
   - `.project-loop/PLAN.md`：对应本轮计划收尾，把 Git 初始化与远程推送标记为已完成。
   - `PROJECT_STATUS.md`：对应 AGENTS.md 的状态刷新规则，把仓库当前状态更新为“已公开推送”。
   - `progress.md`：对应 AGENTS.md 的每阶段结束后强制输出规则，记录提交与推送完成。
3. 哪些地方有实现权衡
   - 没有再扩展代码或数据范围，只补状态同步，避免为了“看起来更完整”而顺手增加无关改动。
4. 是否存在潜在偏离
   - 无 `unmapped_change`。
5. 下一阶段准备做什么
   - 如用户继续迭代，优先补更系统的人工评测和更友好的展示层。
