from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_PROMPT_TEMPLATE = "上联：{up}\n下联："


def load_pairs_from_two_files(in_path, out_path, limit=None):
    in_path = Path(in_path)
    out_path = Path(out_path)
    pairs = []

    with in_path.open("r", encoding="utf-8") as file_in, out_path.open("r", encoding="utf-8") as file_out:
        for up, down in zip(file_in, file_out):
            up = up.strip().replace(" ", "")
            down = down.strip().replace(" ", "")

            if not up or not down:
                continue

            if len(up) != len(down):
                continue

            pairs.append((up, down))

            if limit is not None and limit > 0 and len(pairs) >= limit:
                break

    return pairs


class HFCoupletDataset(Dataset):
    def __init__(self, pairs, tokenizer, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        up, down = self.pairs[idx]
        prompt_text = self.prompt_template.format(up=up)
        full_text = f"{prompt_text}{down}"

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if self.tokenizer.eos_token_id is not None:
            full_ids = full_ids + [self.tokenizer.eos_token_id]

        full_ids = full_ids[: self.max_length]
        attention_mask = [1] * len(full_ids)
        labels = full_ids.copy()

        prompt_len = min(len(prompt_ids), len(full_ids))
        for index in range(prompt_len):
            labels[index] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class HFCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            attention_masks.append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        item["labels"],
                        torch.full((pad_len,), -100, dtype=torch.long),
                    ]
                )
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }
