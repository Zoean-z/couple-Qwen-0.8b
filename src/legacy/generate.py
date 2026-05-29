import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .couplet_dataset import BOS, EOS, PAD, UNK, encode
    from .model import AttentionSeq2Seq, Seq2SeqGRU, Seq2SeqTransformer
except ImportError:
    from couplet_dataset import BOS, EOS, PAD, UNK, encode
    from model import AttentionSeq2Seq, Seq2SeqGRU, Seq2SeqTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "couplet_gru.pth"
DEFAULT_BEAM_SIZE = 3
DEFAULT_REPETITION_PENALTY = 0.6
DEFAULT_EXPLORE_K = 10
DEFAULT_EXPLORE_BEAMS = 1


def setup_utf8_stdio():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="加载对联模型并生成下联。")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="模型 checkpoint 路径。")
    parser.add_argument("--text", type=str, default=None, help="直接传入上联，单次生成后退出。")
    parser.add_argument("--device", default=None, help="手动指定设备，例如 cpu 或 cuda。")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE, help="beam search 宽度，1 表示贪心解码。")
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY, help="重复字惩罚系数，越大越不鼓励重复。")
    parser.add_argument("--diverse-beam", action="store_true", help="启用轻量多样性偏置，在高分候选外保留少量探索 beam。")
    parser.add_argument("--explore-k", type=int, default=DEFAULT_EXPLORE_K, help="多样性偏置时，从前 k 个候选中挑探索分支。")
    parser.add_argument("--explore-beams", type=int, default=DEFAULT_EXPLORE_BEAMS, help="多样性偏置时，每步额外保留的探索 beam。")
    parser.add_argument("--num-candidates", type=int, default=1, help="输出候选条数，仅在 beam search 下生效。")
    parser.add_argument("--no-rerank", action="store_true", help="关闭候选二次排序，直接按模型分数输出。")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint, device):
    model_type = checkpoint.get("model_type", "gru")
    stoi = checkpoint["stoi"]

    if model_type == "transformer":
        model = Seq2SeqTransformer(
            vocab_size=checkpoint["vocab_size"],
            pad_id=stoi[PAD],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_encoder_layers=checkpoint["num_encoder_layers"],
            num_decoder_layers=checkpoint["num_decoder_layers"],
            dim_feedforward=checkpoint["dim_feedforward"],
            dropout=checkpoint.get("dropout", 0.1),
        ).to(device)
    elif model_type == "attention":
        model = AttentionSeq2Seq(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            pad_id=stoi[PAD],
        ).to(device)
    else:
        model = Seq2SeqGRU(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            pad_id=stoi[PAD],
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, model_type


def decode_token_ids(token_ids, itos):
    return "".join(itos[token_id] for token_id in token_ids)


def format_candidates(candidates):
    lines = []
    for idx, candidate in enumerate(candidates, start=1):
        text = candidate["text"]
        score = candidate["score"]
        rerank_score = candidate.get("rerank_score", score)
        lines.append(f"{idx}. {text}  (model={score:.4f}, rerank={rerank_score:.4f})")
    return "\n".join(lines)


def rerank_candidates(src_text, candidates):
    src_chars = set(src_text)
    reranked = []

    for candidate in candidates:
        text = candidate["text"]
        model_score = candidate["score"]
        same_position_count = sum(src_char == tgt_char for src_char, tgt_char in zip(src_text, text))
        shared_char_count = sum(char in src_chars for char in text)
        repeat_count = len(text) - len(set(text))
        adjacent_repeat_count = sum(text[idx] == text[idx - 1] for idx in range(1, len(text)))
        unique_bonus = len(set(text)) / max(1, len(text))

        rerank_score = (
            model_score
            - 0.80 * same_position_count
            - 0.35 * shared_char_count
            - 0.60 * repeat_count
            - 0.90 * adjacent_repeat_count
            + 0.25 * unique_bonus
        )

        new_candidate = dict(candidate)
        new_candidate["rerank_score"] = rerank_score
        reranked.append(new_candidate)

    reranked.sort(key=lambda item: (item["rerank_score"], item["score"]), reverse=True)
    return reranked


def top_candidates(
    logits,
    blocked_ids,
    beam_size,
    used_token_ids=None,
    repetition_penalty=0.0,
    explore_k=None,
    explore_beams=None,
):
    log_probs = F.log_softmax(logits, dim=-1)
    if used_token_ids and repetition_penalty > 0:
        for token_id in used_token_ids:
            log_probs[0, token_id] -= repetition_penalty

    sorted_ids = torch.argsort(log_probs, dim=-1, descending=True)[0].tolist()
    candidates = []

    for token_id in sorted_ids:
        if token_id in blocked_ids:
            continue
        candidates.append((token_id, log_probs[0, token_id].item()))
        if len(candidates) >= beam_size:
            break

    if candidates:
        return candidates

    fallback_id = sorted_ids[0]
    return [(fallback_id, log_probs[0, fallback_id].item())]


def diverse_top_candidates(
    logits,
    blocked_ids,
    beam_size,
    used_token_ids=None,
    repetition_penalty=0.0,
    explore_k=10,
    explore_beams=1,
):
    log_probs = F.log_softmax(logits, dim=-1)
    if used_token_ids and repetition_penalty > 0:
        for token_id in used_token_ids:
            log_probs[0, token_id] -= repetition_penalty

    sorted_ids = torch.argsort(log_probs, dim=-1, descending=True)[0].tolist()
    pool = [token_id for token_id in sorted_ids if token_id not in blocked_ids][: max(explore_k, beam_size)]
    if not pool:
        fallback_id = sorted_ids[0]
        return [(fallback_id, log_probs[0, fallback_id].item())]

    main_keep = max(1, beam_size - max(0, explore_beams))
    selected = pool[:main_keep]

    tail_pool = pool[main_keep:max(explore_k, main_keep)]
    if tail_pool and explore_beams > 0:
        step = max(1, len(tail_pool) // explore_beams)
        for idx in range(0, len(tail_pool), step):
            token_id = tail_pool[idx]
            if token_id not in selected:
                selected.append(token_id)
            if len(selected) >= beam_size:
                break

    if len(selected) < beam_size:
        for token_id in pool:
            if token_id not in selected:
                selected.append(token_id)
            if len(selected) >= beam_size:
                break

    return [(token_id, log_probs[0, token_id].item()) for token_id in selected[:beam_size]]


def select_next_token(logits, blocked_ids, used_token_ids=None, repetition_penalty=0.0):
    return top_candidates(
        logits,
        blocked_ids,
        beam_size=1,
        used_token_ids=used_token_ids,
        repetition_penalty=repetition_penalty,
    )[0][0]


def generate_gru_greedy(model, up, stoi, itos, device, repetition_penalty):
    src = encode(up, stoi, add_eos=True)
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        src_emb = model.embedding(src)
        _, hidden = model.encoder(src_emb)
        input_id = torch.tensor([[stoi[BOS]]], dtype=torch.long).to(device)
        result = []
        used_ids = []

        for _ in range(len(up)):
            emb = model.embedding(input_id)
            output, hidden = model.decoder(emb, hidden)
            logits = model.fc(output[:, -1, :])
            next_id = select_next_token(logits, blocked_ids, used_token_ids=used_ids, repetition_penalty=repetition_penalty)
            result.append(itos[next_id])
            used_ids.append(next_id)
            input_id = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return "".join(result)


def generate_gru_beam(model, up, stoi, itos, device, beam_size, repetition_penalty, diverse_beam=False, explore_k=10, explore_beams=1):
    src = encode(up, stoi, add_eos=True)
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        src_emb = model.embedding(src)
        _, hidden = model.encoder(src_emb)
        beams = [([], 0.0, hidden, torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device))]

        for _ in range(len(up)):
            next_beams = []

            for token_ids, score, beam_hidden, input_id in beams:
                emb = model.embedding(input_id)
                output, next_hidden = model.decoder(emb, beam_hidden)
                logits = model.fc(output[:, -1, :])
                candidate_fn = diverse_top_candidates if diverse_beam else top_candidates

                for next_id, next_score in candidate_fn(
                    logits,
                    blocked_ids,
                    beam_size,
                    used_token_ids=token_ids,
                    repetition_penalty=repetition_penalty,
                    explore_k=explore_k,
                    explore_beams=explore_beams,
                ):
                    next_beams.append(
                        (
                            token_ids + [next_id],
                            score + next_score,
                            next_hidden.clone(),
                            torch.tensor([[next_id]], dtype=torch.long, device=device),
                        )
                    )

            next_beams.sort(key=lambda item: item[1], reverse=True)
            beams = next_beams[:beam_size]

    return [{"text": decode_token_ids(token_ids, itos), "score": score} for token_ids, score, _, _ in beams]


def generate_attention_greedy(model, up, stoi, itos, device, repetition_penalty):
    src = encode(up, stoi, add_eos=True)
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        encoder_outputs, hidden, src_mask = model.encode(src)
        current_token = torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device)
        result = []
        used_ids = []

        for _ in range(len(up)):
            logits = model.decode(current_token, encoder_outputs, hidden, src_mask)
            next_id = select_next_token(
                logits[:, -1, :], blocked_ids, used_token_ids=used_ids, repetition_penalty=repetition_penalty
            )
            result.append(itos[next_id])
            used_ids.append(next_id)

            step_emb = model.embedding(current_token)
            context = model._attention_context(hidden, encoder_outputs, src_mask)
            decoder_input = torch.cat([step_emb, context], dim=-1)
            _, hidden = model.decoder(decoder_input, hidden)
            current_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return "".join(result)


def generate_attention_beam(model, up, stoi, itos, device, beam_size, repetition_penalty, diverse_beam=False, explore_k=10, explore_beams=1):
    src = encode(up, stoi, add_eos=True)
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        encoder_outputs, hidden, src_mask = model.encode(src)
        beams = [([], 0.0, hidden, torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device))]

        for _ in range(len(up)):
            next_beams = []

            for token_ids, score, beam_hidden, current_token in beams:
                logits = model.decode(current_token, encoder_outputs, beam_hidden, src_mask)
                candidate_fn = diverse_top_candidates if diverse_beam else top_candidates

                for next_id, next_score in candidate_fn(
                    logits[:, -1, :],
                    blocked_ids,
                    beam_size,
                    used_token_ids=token_ids,
                    repetition_penalty=repetition_penalty,
                    explore_k=explore_k,
                    explore_beams=explore_beams,
                ):
                    step_emb = model.embedding(current_token)
                    context = model._attention_context(beam_hidden, encoder_outputs, src_mask)
                    decoder_input = torch.cat([step_emb, context], dim=-1)
                    _, next_hidden = model.decoder(decoder_input, beam_hidden)
                    next_beams.append(
                        (
                            token_ids + [next_id],
                            score + next_score,
                            next_hidden.clone(),
                            torch.tensor([[next_id]], dtype=torch.long, device=device),
                        )
                    )

            next_beams.sort(key=lambda item: item[1], reverse=True)
            beams = next_beams[:beam_size]

    return [{"text": decode_token_ids(token_ids, itos), "score": score} for token_ids, score, _, _ in beams]


def generate_transformer_greedy(model, up, stoi, itos, device, repetition_penalty):
    src = encode(up, stoi, add_eos=True)
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        memory, src_key_padding_mask = model.encode(src)
        generated = torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device)
        result = []
        used_ids = []

        for _ in range(len(up)):
            logits = model.decode(generated, memory, src_key_padding_mask)
            next_id = select_next_token(
                logits[:, -1, :], blocked_ids, used_token_ids=used_ids, repetition_penalty=repetition_penalty
            )
            result.append(itos[next_id])
            used_ids.append(next_id)
            generated = torch.cat([generated, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    return "".join(result)


def generate_transformer_beam(model, up, stoi, itos, device, beam_size, repetition_penalty, diverse_beam=False, explore_k=10, explore_beams=1):
    src = encode(up, stoi, add_eos=True)
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
    blocked_ids = {stoi[PAD], stoi[BOS], stoi[EOS], stoi[UNK]}

    with torch.no_grad():
        memory, src_key_padding_mask = model.encode(src)
        beams = [([stoi[BOS]], 0.0)]

        for _ in range(len(up)):
            next_beams = []

            for generated_ids, score in beams:
                generated = torch.tensor([generated_ids], dtype=torch.long, device=device)
                logits = model.decode(generated, memory, src_key_padding_mask)
                candidate_fn = diverse_top_candidates if diverse_beam else top_candidates

                for next_id, next_score in candidate_fn(
                    logits[:, -1, :],
                    blocked_ids,
                    beam_size,
                    used_token_ids=generated_ids[1:],
                    repetition_penalty=repetition_penalty,
                    explore_k=explore_k,
                    explore_beams=explore_beams,
                ):
                    next_beams.append((generated_ids + [next_id], score + next_score))

            next_beams.sort(key=lambda item: item[1], reverse=True)
            beams = next_beams[:beam_size]

    return [{"text": decode_token_ids(token_ids[1:], itos), "score": score} for token_ids, score in beams]


def generate(
    model,
    model_type,
    up,
    stoi,
    itos,
    device,
    beam_size,
    repetition_penalty,
    diverse_beam=False,
    explore_k=10,
    explore_beams=1,
):
    model.eval()

    if model_type == "transformer":
        if beam_size > 1:
            return generate_transformer_beam(
                model, up, stoi, itos, device, beam_size, repetition_penalty, diverse_beam, explore_k, explore_beams
            )
        return generate_transformer_greedy(model, up, stoi, itos, device, repetition_penalty)

    if model_type == "attention":
        if beam_size > 1:
            return generate_attention_beam(
                model, up, stoi, itos, device, beam_size, repetition_penalty, diverse_beam, explore_k, explore_beams
            )
        return generate_attention_greedy(model, up, stoi, itos, device, repetition_penalty)

    if beam_size > 1:
        return generate_gru_beam(
            model, up, stoi, itos, device, beam_size, repetition_penalty, diverse_beam, explore_k, explore_beams
        )
    return generate_gru_greedy(model, up, stoi, itos, device, repetition_penalty)


def generate_candidates(
    model,
    model_type,
    up,
    stoi,
    itos,
    device,
    beam_size,
    repetition_penalty,
    diverse_beam=False,
    explore_k=10,
    explore_beams=1,
    num_candidates=1,
):
    result = generate(
        model,
        model_type,
        up,
        stoi,
        itos,
        device,
        beam_size,
        repetition_penalty,
        diverse_beam,
        explore_k,
        explore_beams,
    )

    if beam_size > 1:
        return result[:num_candidates]

    return [{"text": result, "score": 0.0, "rerank_score": 0.0}]


def main():
    setup_utf8_stdio()
    args = parse_args()
    if args.beam_size < 1:
        raise ValueError("--beam-size 必须大于等于 1。")
    if args.explore_k < 1:
        raise ValueError("--explore-k 必须大于等于 1。")
    if args.explore_beams < 0:
        raise ValueError("--explore-beams 不能小于 0。")
    if args.num_candidates < 1:
        raise ValueError("--num-candidates 必须大于等于 1。")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    model, model_type = load_model_from_checkpoint(checkpoint, device)
    print("模型类型：", model_type)
    print("解码方式：", "beam search" if args.beam_size > 1 else "greedy")
    print("重复惩罚：", args.repetition_penalty)
    print("多样性偏置：", args.diverse_beam)
    print("候选数量：", args.num_candidates if args.beam_size > 1 else 1)
    print("候选重排：", not args.no_rerank)

    if args.text:
        candidates = generate_candidates(
            model,
            model_type,
            args.text.strip(),
            stoi,
            itos,
            device,
            args.beam_size,
            args.repetition_penalty,
            args.diverse_beam,
            args.explore_k,
            args.explore_beams,
            args.num_candidates,
        )
        if args.beam_size > 1 and not args.no_rerank:
            candidates = rerank_candidates(args.text.strip(), candidates)
        if len(candidates) == 1:
            print("下联：", candidates[0]["text"])
        else:
            print("候选下联：")
            print(format_candidates(candidates))
        return

    while True:
        up = input("请输入上联，输入 q 退出：").strip()
        if up.lower() == "q":
            break

        candidates = generate_candidates(
            model,
            model_type,
            up,
            stoi,
            itos,
            device,
            args.beam_size,
            args.repetition_penalty,
            args.diverse_beam,
            args.explore_k,
            args.explore_beams,
            args.num_candidates,
        )
        if args.beam_size > 1 and not args.no_rerank:
            candidates = rerank_candidates(up, candidates)
        if len(candidates) == 1:
            print("下联：", candidates[0]["text"])
        else:
            print("候选下联：")
            print(format_candidates(candidates))


if __name__ == "__main__":
    main()
