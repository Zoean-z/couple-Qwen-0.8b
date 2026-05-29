import argparse
import json
import math
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .hf_couplet_dataset import DEFAULT_PROMPT_TEMPLATE, HFCollator, HFCoupletDataset, load_pairs_from_two_files
except ImportError:
    from hf_couplet_dataset import DEFAULT_PROMPT_TEMPLATE, HFCollator, HFCoupletDataset, load_pairs_from_two_files


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IN_PATH = BASE_DIR / "data" / "couplet" / "couplet" / "train" / "in.txt"
DEFAULT_OUT_PATH = BASE_DIR / "data" / "couplet" / "couplet" / "train" / "out.txt"
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_OUTPUT_DIR = BASE_DIR / "checkpoints" / "qwen3_0_6b_lora_smoke"
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-4
DEFAULT_LIMIT = 64
DEFAULT_VAL_RATIO = 0.1
DEFAULT_MAX_LENGTH = 128
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05


def setup_utf8_stdio():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B-Base LoRA 最小训练入口。")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--in-path", type=Path, default=DEFAULT_IN_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_dataset(dataset, val_ratio, seed):
    if val_ratio <= 0 or len(dataset) < 2:
        return dataset, None

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size = len(dataset) - 1
        val_size = 1

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / max(1, steps)


def save_run_config(args, output_dir, train_size, val_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": args.model_id,
        "in_path": str(args.in_path),
        "out_path": str(args.out_path),
        "prompt_template": args.prompt_template,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "limit": args.limit,
        "val_ratio": args.val_ratio,
        "max_length": args.max_length,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "train_size": train_size,
        "val_size": val_size,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    setup_utf8_stdio()
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = load_pairs_from_two_files(args.in_path, args.out_path, limit=args.limit)
    dataset = HFCoupletDataset(
        pairs,
        tokenizer=tokenizer,
        prompt_template=args.prompt_template,
        max_length=args.max_length,
    )
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)

    collator = HFCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model_kwargs = {}
    if device == "cuda":
        model_kwargs["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    train_size = len(train_dataset)
    val_size = 0 if val_dataset is None else len(val_dataset)
    save_run_config(args, args.output_dir, train_size, val_size)

    print(f"使用设备：{device}")
    print(f"训练样本：{train_size}")
    print(f"验证样本：{val_size}")
    print(f"输出目录：{args.output_dir}")

    best_val_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += outputs.loss.item()
            step_count += 1

        train_loss = running_loss / max(1, step_count)
        print(f"epoch {epoch} train_loss={train_loss:.4f}")

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"epoch {epoch} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = args.output_dir / "best_adapter"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"saved best adapter -> {best_dir}")

    final_dir = args.output_dir / "final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved final adapter -> {final_dir}")


if __name__ == "__main__":
    main()
