from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


def load_pairs(data_path):
    data_path = Path(data_path)
    pairs = []

    with data_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue

            if "|" in line:
                parts = line.split("|", maxsplit=1)
            else:
                parts = line.split()

            if len(parts) != 2:
                print("格式错误，已跳过：", line)
                continue

            up, down = parts
            if len(up) != len(down):
                continue

            pairs.append((up, down))

    return pairs


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

            if limit is not None and len(pairs) >= limit:
                break

    return pairs


def build_vocab(pairs):
    chars = set()

    for up, down in pairs:
        chars.update(up)
        chars.update(down)

    itos = [PAD, BOS, EOS, UNK] + sorted(chars)
    stoi = {char: index for index, char in enumerate(itos)}
    return stoi, itos


def encode(text, stoi, add_bos=False, add_eos=False):
    ids = []

    if add_bos:
        ids.append(stoi[BOS])

    for char in text:
        ids.append(stoi.get(char, stoi[UNK]))

    if add_eos:
        ids.append(stoi[EOS])

    return ids


class CoupletDataset(Dataset):
    def __init__(self, pairs, stoi):
        self.pairs = pairs
        self.stoi = stoi

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        up, down = self.pairs[idx]

        src = encode(up, self.stoi, add_eos=True)
        tgt_in = encode(down, self.stoi, add_bos=True)
        tgt_out = encode(down, self.stoi, add_eos=True)

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )


def collate_fn(batch):
    srcs, tgt_ins, tgt_outs = zip(*batch)

    srcs = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_ins = pad_sequence(tgt_ins, batch_first=True, padding_value=0)
    tgt_outs = pad_sequence(tgt_outs, batch_first=True, padding_value=0)

    return srcs, tgt_ins, tgt_outs
