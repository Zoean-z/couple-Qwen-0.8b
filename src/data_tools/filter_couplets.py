import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from pypinyin import Style, pinyin


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_IN_PATH = BASE_DIR / "data" / "couplet" / "couplet" / "train" / "in.txt"
DEFAULT_OUT_PATH = BASE_DIR / "data" / "couplet" / "couplet" / "train" / "out.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "filtered_couplets"
DEFAULT_TOP_K = 10000

TRADITIONAL_IMAGERY = [
    "山", "水", "风", "月", "花", "云", "雨", "春", "秋", "雪", "梅", "竹", "柳", "松", "江", "海", "天", "日",
]

MODERN_WORDS = [
    "手机", "电脑", "网络", "互联网", "人工智能", "AI", "APP", "app", "程序", "代码", "芯片", "高铁", "地铁",
    "直播", "视频", "粉丝", "流量", "快递", "外卖", "咖啡", "老板", "工资", "房贷", "证券", "基金", "股票",
    "美元", "公司", "集团", "平台", "系统", "数据", "算法", "机器人", "火箭", "航天", "汽车", "电车",
]

BAD_CHAR_PATTERN = re.compile(r"[\uFFFD�]|[\x00-\x08\x0B\x0C\x0E-\x1F]")
PUNCT_PATTERN = re.compile(r"[，。！？；：、“”‘’（）《》【】,.;:!?\"'()<>[\]{}]")
CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def parse_args():
    parser = argparse.ArgumentParser(description="按规则给对联训练集打分并筛选适合训练的子集。")
    parser.add_argument("--in-path", type=Path, default=DEFAULT_IN_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def load_pairs(in_path, out_path):
    with in_path.open("r", encoding="utf-8") as file_in, out_path.open("r", encoding="utf-8") as file_out:
        for line_no, (up, down) in enumerate(zip(file_in, file_out), start=1):
            up = up.strip().replace(" ", "")
            down = down.strip().replace(" ", "")
            if not up or not down:
                continue
            yield line_no, up, down


def is_ping_tone(char):
    tones = pinyin(char, style=Style.TONE3, heteronym=True, errors=lambda _: [[]])[0]
    for tone in tones:
        if not tone:
            continue
        match = re.search(r"([1-4])$", tone)
        if match and match.group(1) in {"1", "2"}:
            return True
    return False


def contains_traditional_imagery(text):
    return any(token in text for token in TRADITIONAL_IMAGERY)


def contains_modern_word(text):
    return any(token in text for token in MODERN_WORDS)


def has_garble_or_punct_issue(text):
    punct_count = len(PUNCT_PATTERN.findall(text))
    chinese_count = len(CHINESE_CHAR_PATTERN.findall(text))
    if BAD_CHAR_PATTERN.search(text):
        return True
    if chinese_count == 0:
        return True
    return punct_count > max(1, chinese_count // 2)


def shared_char_bonus(up, down):
    up_set = set(up)
    down_set = set(down)
    overlap = len(up_set & down_set)
    ratio = overlap / max(1, len(up_set | down_set))
    if ratio <= 0.12:
        return 10
    if ratio <= 0.2:
        return 5
    return 0


def score_pair(up, down):
    score = 0
    reasons = []
    pair_len = len(up)

    if len(up) == len(down):
        score += 30
        reasons.append("字数一致:+30")

    if pair_len in {5, 7, 9, 11}:
        score += 10
        reasons.append(f"{pair_len}字常见长度:+10")

    if down and is_ping_tone(down[-1]):
        score += 10
        reasons.append("下联末字平声:+10")

    overlap_bonus = shared_char_bonus(up, down)
    if overlap_bonus:
        score += overlap_bonus
        reasons.append(f"上下联重复字少:+{overlap_bonus}")

    if contains_traditional_imagery(up) or contains_traditional_imagery(down):
        score += 5
        reasons.append("含传统意象:+5")

    if contains_modern_word(up) or contains_modern_word(down):
        score -= 30
        reasons.append("含现代词:-30")

    if has_garble_or_punct_issue(up) or has_garble_or_punct_issue(down):
        score -= 50
        reasons.append("乱码或标点混乱:-50")

    return score, reasons


def length_bucket(length):
    if length == 5:
        return "5"
    if length == 7:
        return "7"
    if length in {9, 11}:
        return "9_11"
    return None


def sort_key(item):
    return (
        item["score"],
        1 if item["bucket"] == "7" else 0,
        item["length"],
        -item["line_no"],
    )


def select_top(records, top_k):
    quotas = {
        "5": int(top_k * 0.3),
        "7": int(top_k * 0.5),
    }
    quotas["9_11"] = top_k - quotas["5"] - quotas["7"]

    grouped = {"5": [], "7": [], "9_11": []}
    for record in records:
        if record["bucket"] in grouped:
            grouped[record["bucket"]].append(record)

    for bucket in grouped:
        grouped[bucket].sort(key=sort_key, reverse=True)

    selected = []
    for bucket, quota in quotas.items():
        selected.extend(grouped[bucket][:quota])

    if len(selected) < top_k:
        selected_ids = {item["line_no"] for item in selected}
        leftovers = [item for item in records if item["line_no"] not in selected_ids and item["bucket"] in grouped]
        leftovers.sort(key=sort_key, reverse=True)
        selected.extend(leftovers[: top_k - len(selected)])

    selected.sort(key=sort_key, reverse=True)
    return selected[:top_k], quotas


def write_outputs(output_dir, scored_records, selected_records, quotas):
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_tsv = output_dir / "scored_pairs.tsv"
    with scored_tsv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(["line_no", "length", "bucket", "score", "up", "down", "reasons"])
        for record in scored_records:
            writer.writerow(
                [
                    record["line_no"],
                    record["length"],
                    record["bucket"] or "",
                    record["score"],
                    record["up"],
                    record["down"],
                    " | ".join(record["reasons"]),
                ]
            )

    selected_tsv = output_dir / "top_selected.tsv"
    with selected_tsv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(["rank", "line_no", "length", "bucket", "score", "up", "down", "reasons"])
        for rank, record in enumerate(selected_records, start=1):
            writer.writerow(
                [
                    rank,
                    record["line_no"],
                    record["length"],
                    record["bucket"],
                    record["score"],
                    record["up"],
                    record["down"],
                    " | ".join(record["reasons"]),
                ]
            )

    selected_in = output_dir / "top_selected_in.txt"
    selected_out = output_dir / "top_selected_out.txt"
    selected_json = output_dir / "selection_summary.json"

    selected_in.write_text("\n".join(record["up"] for record in selected_records), encoding="utf-8")
    selected_out.write_text("\n".join(record["down"] for record in selected_records), encoding="utf-8")

    summary = {
        "total_scored": len(scored_records),
        "total_selected": len(selected_records),
        "quota_target": quotas,
        "selected_bucket_counts": dict(Counter(record["bucket"] for record in selected_records)),
        "selected_length_counts": dict(Counter(record["length"] for record in selected_records)),
        "top10_preview": [
            {
                "line_no": record["line_no"],
                "length": record["length"],
                "score": record["score"],
                "up": record["up"],
                "down": record["down"],
            }
            for record in selected_records[:10]
        ],
    }
    selected_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()

    scored_records = []
    for line_no, up, down in load_pairs(args.in_path, args.out_path):
        bucket = length_bucket(len(up))
        score, reasons = score_pair(up, down)
        scored_records.append(
            {
                "line_no": line_no,
                "up": up,
                "down": down,
                "length": len(up),
                "bucket": bucket,
                "score": score,
                "reasons": reasons,
            }
        )

    candidate_records = [record for record in scored_records if record["bucket"] is not None]
    selected_records, quotas = select_top(candidate_records, args.top_k)
    write_outputs(args.output_dir, scored_records, selected_records, quotas)

    print(f"scored={len(scored_records)}")
    print(f"candidates={len(candidate_records)}")
    print(f"selected={len(selected_records)}")
    print(f"output_dir={args.output_dir}")
    print(f"bucket_counts={dict(Counter(record['bucket'] for record in selected_records))}")


if __name__ == "__main__":
    main()
