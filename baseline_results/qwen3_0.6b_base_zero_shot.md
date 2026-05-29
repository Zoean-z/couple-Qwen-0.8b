# Qwen3-0.6B-Base Zero-Shot Baseline

时间：2026-05-18  
模型：`Qwen/Qwen3-0.6B-Base`  
方式：零微调，本地 prompt 直接生成  
入口：`src/hf/generate_hf_clm.py`

## 命令

```powershell
.venv\Scripts\python.exe src\hf\generate_hf_clm.py
```

## Prompt 模板

```text
请对出下联，只输出下联，不要解释。
上联：{up}
下联：
```

## 固定样例输出

### 春风送暖

1. 秋雨润心
2. 夏雨润心

### 山高水长

1. 海阔天宽
2. 海阔天高

### 国泰民安

1. 家和万事兴
2. 民富国强

## 备注

- 当前脚本使用 `beam search` 零微调生成。
- 这份结果用于后续 LoRA / SFT 微调前后的直接对照。
- 首次运行会把模型下载到本机 Hugging Face 缓存目录。
