import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_ADAPTER_DIR = Path("checkpoints") / "qwen3_0_6b_lora_filtered_e4" / "best_adapter"
DEFAULT_PROMPT_TEMPLATE = "上联：{up}\n下联："
DEFAULT_NUM_BEAMS = 5


def setup_utf8_stdio():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="加载 Qwen LoRA adapter 做对联生成。")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="底模 ID。")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR, help="LoRA adapter 目录。")
    parser.add_argument("--text", default=None, help="单次输入上联；不传则进入交互模式。")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE, help="生成 prompt 模板。")
    parser.add_argument("--device", default=None, help="手动指定设备，例如 cpu 或 cuda。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成 token 数。")
    parser.add_argument("--num-beams", type=int, default=DEFAULT_NUM_BEAMS, help="beam search 宽度。")
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def clean_output(text):
    text = text.replace("\r", "")
    text = text.split("\n", 1)[0]
    return text.strip()


def load_lora_model(model_id, adapter_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if device == "cuda":
        model_kwargs["dtype"] = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_once(model, tokenizer, up, device, prompt_template, max_new_tokens=None, num_beams=5):
    prompt = prompt_template.format(up=up)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    max_new_tokens = max_new_tokens or max(8, len(up) + 6)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return clean_output(completion)


def run_interactive(model, tokenizer, device, prompt_template, max_new_tokens, num_beams):
    print("进入交互模式。直接输入上联，回车生成下联；输入 exit 退出。")
    while True:
        try:
            up = input("上联> ").strip()
        except EOFError:
            print()
            break

        if not up:
            continue
        if up.lower() in {"exit", "quit", "q"}:
            break

        down = generate_once(
            model,
            tokenizer,
            up,
            device,
            prompt_template,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        print(f"下联> {down}\n")


def main():
    setup_utf8_stdio()
    args = parse_args()
    device = pick_device(args.device)
    model, tokenizer = load_lora_model(args.model_id, args.adapter_dir, device)

    if args.text:
        print(
            generate_once(
                model,
                tokenizer,
                args.text,
                device,
                args.prompt_template,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        )
        return

    run_interactive(
        model,
        tokenizer,
        device,
        args.prompt_template,
        args.max_new_tokens,
        args.num_beams,
    )


if __name__ == "__main__":
    main()
