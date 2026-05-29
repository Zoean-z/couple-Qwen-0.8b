import argparse
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .couplet_dataset import BOS, EOS, PAD, UNK, CoupletDataset, build_vocab, collate_fn, load_pairs_from_two_files
    from .model import AttentionSeq2Seq, Seq2SeqGRU, Seq2SeqTransformer
except ImportError:
    from couplet_dataset import BOS, EOS, PAD, UNK, CoupletDataset, build_vocab, collate_fn, load_pairs_from_two_files
    from model import AttentionSeq2Seq, Seq2SeqGRU, Seq2SeqTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IN_PATH = BASE_DIR / "data" / "couplet" / "couplet" / "train" / "in.txt"
DEFAULT_OUT_PATH = BASE_DIR / "data" / "couplet" / "couplet" / "train" / "out.txt"
DEFAULT_EMBED_DIM = 128
DEFAULT_HIDDEN_DIM = 256
DEFAULT_D_MODEL = 256
DEFAULT_NHEAD = 4
DEFAULT_ENCODER_LAYERS = 4
DEFAULT_DECODER_LAYERS = 4
DEFAULT_FF_DIM = 512
DEFAULT_DROPOUT = 0.1
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 30
DEFAULT_LR = 0.001
DEFAULT_LIMIT = 5000
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_SAMPLE_TEXTS = ["春风送暖", "山高水长", "国泰民安"]
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_GRAD_CLIP = 0.0


def setup_utf8_stdio():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="训练中文对联生成模型。")
    parser.add_argument("--model-type", choices=["gru", "attention", "transformer"], default="gru", help="选择训练的模型类型。")
    parser.add_argument("--in-path", type=Path, default=DEFAULT_IN_PATH, help="上联训练数据路径。")
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH, help="下联训练数据路径。")
    parser.add_argument("--save-path", type=Path, default=None, help="latest checkpoint 保存路径。")
    parser.add_argument("--best-save-path", type=Path, default=None, help="best checkpoint 保存路径。")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM, help="GRU/Attention 使用的字向量维度。")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="GRU/Attention 使用的隐层维度。")
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL, help="Transformer 的模型维度。")
    parser.add_argument("--nhead", type=int, default=DEFAULT_NHEAD, help="Transformer 的注意力头数。")
    parser.add_argument("--num-encoder-layers", type=int, default=DEFAULT_ENCODER_LAYERS, help="Transformer encoder 层数。")
    parser.add_argument("--num-decoder-layers", type=int, default=DEFAULT_DECODER_LAYERS, help="Transformer decoder 层数。")
    parser.add_argument("--ff-dim", type=int, default=DEFAULT_FF_DIM, help="Transformer 前馈层维度。")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Transformer dropout。")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="训练 batch 大小。")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数。")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="学习率。")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="读取样本条数上限，传 0 表示全量。")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="验证集比例，传 0 表示不切验证集。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="训练/验证切分随机种子。")
    parser.add_argument("--sample-text", action="append", default=None, help="训练时每轮额外生成的固定样例，可重复传入。")
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING, help="交叉熵标签平滑系数。")
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP, help="梯度裁剪阈值，传 0 表示关闭。")
    parser.add_argument("--device", default=None, help="手动指定设备，例如 cpu 或 cuda。")
    return parser.parse_args()


def resolve_save_path(args):
    if args.save_path is not None:
        return args.save_path

    if args.model_type == "transformer":
        filename = "couplet_transformer.pth"
    elif args.model_type == "attention":
        filename = "couplet_attention.pth"
    else:
        filename = "couplet_gru.pth"

    return BASE_DIR / "checkpoints" / filename


def resolve_best_save_path(args, save_path):
    if args.best_save_path is not None:
        return args.best_save_path
    return save_path.with_name(f"{save_path.stem}_best{save_path.suffix}")


def build_model(args, vocab_size, pad_id):
    if args.model_type == "transformer":
        return Seq2SeqTransformer(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.ff_dim,
            dropout=args.dropout,
        )

    if args.model_type == "attention":
        return AttentionSeq2Seq(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            pad_id=pad_id,
        )

    return Seq2SeqGRU(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        pad_id=pad_id,
    )


def build_checkpoint(args, model, stoi, itos, epoch, train_loss, val_loss):
    checkpoint = {
        "model_type": args.model_type,
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(itos),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
    }

    if args.model_type == "transformer":
        checkpoint.update(
            {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "dim_feedforward": args.ff_dim,
                "dropout": args.dropout,
            }
        )
    else:
        checkpoint.update(
            {
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
            }
        )

    return checkpoint


def split_pairs(pairs, val_ratio, seed):
    if val_ratio <= 0 or len(pairs) < 2:
        return pairs, []

    val_size = int(len(pairs) * val_ratio)
    val_size = max(1, min(val_size, len(pairs) - 1))

    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[val_size:], shuffled[:val_size]


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src_batch, tgt_in_batch, tgt_out_batch in loader:
            src_batch = src_batch.to(device)
            tgt_in_batch = tgt_in_batch.to(device)
            tgt_out_batch = tgt_out_batch.to(device)

            logits = model(src_batch, tgt_in_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out_batch.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(loader)


def encode_text(text, stoi):
    ids = [stoi.get(char, stoi[UNK]) for char in text]
    ids.append(stoi[EOS])
    return ids


def select_non_special_token(logits, blocked_ids):
    sorted_ids = torch.argsort(logits, dim=-1, descending=True)
    for token_id in sorted_ids[0].tolist():
        if token_id not in blocked_ids:
            return token_id
    return sorted_ids[0, 0].item()


def greedy_generate_gru(model, text, stoi, itos, device):
    src = torch.tensor(encode_text(text, stoi), dtype=torch.long, device=device).unsqueeze(0)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        src_emb = model.embedding(src)
        _, hidden = model.encoder(src_emb)
        input_id = torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device)
        result = []

        for _ in range(len(text)):
            emb = model.embedding(input_id)
            output, hidden = model.decoder(emb, hidden)
            logits = model.fc(output[:, -1, :])
            next_id = select_non_special_token(logits, blocked_ids)
            result.append(itos[next_id])
            input_id = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return "".join(result)


def greedy_generate_attention(model, text, stoi, itos, device):
    src = torch.tensor(encode_text(text, stoi), dtype=torch.long, device=device).unsqueeze(0)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        encoder_outputs, hidden, src_mask = model.encode(src)
        current_token = torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device)
        result = []

        for _ in range(len(text)):
            logits = model.decode(current_token, encoder_outputs, hidden, src_mask)
            next_id = select_non_special_token(logits[:, -1, :], blocked_ids)
            result.append(itos[next_id])

            step_emb = model.embedding(current_token)
            context = model._attention_context(hidden, encoder_outputs, src_mask)
            decoder_input = torch.cat([step_emb, context], dim=-1)
            _, hidden = model.decoder(decoder_input, hidden)
            current_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return "".join(result)


def greedy_generate_transformer(model, text, stoi, itos, device):
    src = torch.tensor(encode_text(text, stoi), dtype=torch.long, device=device).unsqueeze(0)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        memory, src_key_padding_mask = model.encode(src)
        generated = torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device)
        result = []

        for _ in range(len(text)):
            logits = model.decode(generated, memory, src_key_padding_mask)
            next_id = select_non_special_token(logits[:, -1, :], blocked_ids)
            result.append(itos[next_id])
            generated = torch.cat([generated, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    return "".join(result)


def generate_samples(model, model_type, sample_texts, stoi, itos, device):
    model.eval()
    results = []

    for text in sample_texts:
        if model_type == "transformer":
            down = greedy_generate_transformer(model, text, stoi, itos, device)
        elif model_type == "attention":
            down = greedy_generate_attention(model, text, stoi, itos, device)
        else:
            down = greedy_generate_gru(model, text, stoi, itos, device)
        results.append((text, down))

    return results


def main():
    setup_utf8_stdio()
    args = parse_args()
    if args.label_smoothing < 0 or args.label_smoothing >= 1:
        raise ValueError("--label-smoothing 必须在 [0, 1) 区间内。")
    if args.grad_clip < 0:
        raise ValueError("--grad-clip 不能小于 0。")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_path = resolve_save_path(args)
    best_save_path = resolve_best_save_path(args, save_path)
    print("使用设备：", device)
    print("模型类型：", args.model_type)

    limit = None if args.limit == 0 else args.limit
    pairs = load_pairs_from_two_files(args.in_path, args.out_path, limit=limit)
    print("读取到对联数量：", len(pairs))

    if not pairs:
        raise ValueError("未读取到有效训练数据，请检查 --in-path 和 --out-path。")

    stoi, itos = build_vocab(pairs)
    print("词表大小：", len(itos))

    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    print("训练集数量：", len(train_pairs))
    print("验证集数量：", len(val_pairs))
    print("标签平滑：", args.label_smoothing)
    print("梯度裁剪：", args.grad_clip)

    train_dataset = CoupletDataset(train_pairs, stoi)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_pairs:
        val_dataset = CoupletDataset(val_pairs, stoi)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    model = build_model(args, vocab_size=len(itos), pad_id=stoi[PAD]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=stoi[PAD], label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = None
    best_checkpoint = None
    sample_texts = args.sample_text or DEFAULT_SAMPLE_TEXTS

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for src_batch, tgt_in_batch, tgt_out_batch in train_loader:
            src_batch = src_batch.to(device)
            tgt_in_batch = tgt_in_batch.to(device)
            tgt_out_batch = tgt_out_batch.to(device)

            logits = model(src_batch, tgt_in_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out_batch.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device) if val_loader is not None else None

        checkpoint = build_checkpoint(
            args=args,
            model=model,
            stoi=stoi,
            itos=itos,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)

        if val_loss is not None:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = copy.deepcopy(checkpoint)
                best_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_checkpoint, best_save_path)
        else:
            best_checkpoint = checkpoint
            torch.save(best_checkpoint, best_save_path)

        if val_loss is None:
            print(f"epoch {epoch + 1}, train_loss = {train_loss:.4f}")
        else:
            print(f"epoch {epoch + 1}, train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

        for sample_text, sample_down in generate_samples(model, args.model_type, sample_texts, stoi, itos, device):
            print(f"sample | 上联：{sample_text} | 下联：{sample_down}")

    print("latest checkpoint：", save_path)
    if best_checkpoint is not None:
        print("best checkpoint：", best_save_path)


if __name__ == "__main__":
    main()
