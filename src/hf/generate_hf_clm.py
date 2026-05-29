import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_BEAM_SIZE = 5
DEFAULT_NUM_CANDIDATES = 3
DEFAULT_PROMPT_TEMPLATE = "请对出下联，只输出下联，不要解释。\n上联：{up}\n下联："
DEFAULT_SAMPLE_TEXTS = ["春风送暖", "山高水长", "国泰民安"]


def setup_utf8_stdio():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="加载 Hugging Face causal LM 做零微调对联生成。")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face 模型 ID。")
    parser.add_argument("--text", type=str, default=None, help="单次输入上联。")
    parser.add_argument("--sample-text", action="append", default=None, help="批量测试的固定上联，可重复传入。")
    parser.add_argument("--device", default=None, help="手动指定设备，例如 cpu 或 cuda。")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE, help="beam search 宽度。")
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES, help="返回候选条数。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大新生成 token 数；默认按上联长度估算。")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE, help="prompt 模板，使用 {up} 代入上联。")
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if device == "cuda":
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def clean_output(text):
    text = text.replace("\r", "")
    text = text.split("\n", 1)[0]
    return text.strip()


def generate_candidates(model, tokenizer, up, device, beam_size, num_candidates, max_new_tokens, prompt_template):
    prompt = prompt_template.format(up=up)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    if max_new_tokens is None:
        max_new_tokens = max(8, len(up) + 6)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=max(beam_size, num_candidates),
        num_return_sequences=num_candidates,
        do_sample=False,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    candidates = []
    seen = set()
    for sequence in outputs.sequences:
        completion = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
        text = clean_output(completion)
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append(text)

    return candidates


def print_result(up, candidates):
    print(f"上联：{up}")
    if not candidates:
        print("未生成出有效候选。")
        return
    for idx, candidate in enumerate(candidates, start=1):
        print(f"{idx}. {candidate}")


def main():
    setup_utf8_stdio()
    args = parse_args()
    device = pick_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)

    if args.text:
        candidates = generate_candidates(
            model,
            tokenizer,
            args.text,
            device,
            args.beam_size,
            args.num_candidates,
            args.max_new_tokens,
            args.prompt_template,
        )
        print_result(args.text, candidates)
        return

    sample_texts = args.sample_text or DEFAULT_SAMPLE_TEXTS
    for idx, up in enumerate(sample_texts):
        if idx > 0:
            print()
        candidates = generate_candidates(
            model,
            tokenizer,
            up,
            device,
            args.beam_size,
            args.num_candidates,
            args.max_new_tokens,
            args.prompt_template,
        )
        print_result(up, candidates)


if __name__ == "__main__":
    main()
