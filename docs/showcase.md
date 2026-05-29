# Showcase

这个文件只做一件事：把当前仓库最值得展示的效果单独放出来，便于老师或面试官快速浏览。

## 真实运行输出

运行命令：

```powershell
.venv\Scripts\python.exe src\hf\generate_hf_lora.py --device cpu --text 风弦未拨心先乱
.venv\Scripts\python.exe src\hf\generate_hf_lora.py --device cpu --text 花梦粘于春袖口
.venv\Scripts\python.exe src\hf\generate_hf_lora.py --device cpu --text 春风送暖
```

实际输出：

| 上联 | 下联 |
| --- | --- |
| 风弦未拨心先乱 | 月笛犹弹梦已空 |
| 花梦粘于春袖口 | 诗情融在柳眉梢 |
| 春风送暖 | 瑞雪迎春 |

## 固定对照样例

下面两条来自 `baseline_results/qwen3_lora_filtered_e4_vs_baselines.md`，用来说明这条路线为什么值得继续。

### 样例 1

| 项目 | 内容 |
| --- | --- |
| 上联 | 风弦未拨心先乱 |
| 参考 | 夜幕已沉梦更闲 |
| Zero-shot Prompt C | 雨露无声草自荣 |
| LoRA E1 | 月笛已吹梦已空 |
| LoRA E4(best@epoch2) | 月笛犹弹梦已空 |
| Attention baseline | 竹影犹留客难眠 |

### 样例 2

| 项目 | 内容 |
| --- | --- |
| 上联 | 花梦粘于春袖口 |
| 参考 | 莺声溅落柳枝头 |
| Zero-shot Prompt C | 月影藏于夜幕中 |
| LoRA E1 | 莺歌绕在柳枝头 |
| LoRA E4(best@epoch2) | 诗魂凝在柳眉梢 |
| Attention baseline | 秋风入画图中天 |

## 这轮结果最适合怎么讲

最稳妥的说法不是“已经做成了一个高质量对联生成器”，而是：

> 我已经验证了 `Qwen3-0.6B-Base + LoRA` 这条路线在这个任务上比 zero-shot 和当前手写 baseline 更有效，尤其是在减少复读和提升任务贴合度上；但如果想继续把质量往上推，下一阶段的重点应该是数据质量和约束设计，而不是机械继续加训练轮数。
