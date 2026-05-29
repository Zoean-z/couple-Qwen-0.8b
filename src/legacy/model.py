import math

import torch
import torch.nn as nn


class Seq2SeqGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, pad_id=0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )

        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.decoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt_in):
        src_emb = self.embedding(src)
        _, hidden = self.encoder(src_emb)

        tgt_emb = self.embedding(tgt_in)
        output, _ = self.decoder(tgt_emb, hidden)

        logits = self.fc(output)
        return logits


class AttentionSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = nn.GRU(
            input_size=embed_dim + hidden_dim * 2,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.encoder_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src):
        src_mask = src.eq(self.pad_id)
        src_emb = self.embedding(src)
        encoder_outputs, hidden = self.encoder(src_emb)
        hidden = hidden.view(1, 2, src.size(0), self.hidden_dim).sum(dim=1)
        return encoder_outputs, hidden, src_mask

    def _attention_context(self, decoder_hidden, encoder_outputs, src_mask):
        query = self.attn_query(decoder_hidden[-1]).unsqueeze(1)
        keys = self.attn_key(encoder_outputs)
        scores = torch.bmm(query, keys.transpose(1, 2))
        scores = scores.masked_fill(src_mask.unsqueeze(1), -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)
        return context

    def decode(self, tgt_in, encoder_outputs, hidden, src_mask):
        tgt_emb = self.embedding(tgt_in)
        outputs = []
        decoder_hidden = hidden

        for step in range(tgt_emb.size(1)):
            step_emb = tgt_emb[:, step : step + 1, :]
            context = self._attention_context(decoder_hidden, encoder_outputs, src_mask)
            decoder_input = torch.cat([step_emb, context], dim=-1)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs.append(decoder_output)

        outputs = torch.cat(outputs, dim=1)
        return self.fc(outputs)

    def forward(self, src, tgt_in):
        encoder_outputs, hidden, src_mask = self.encode(src)
        return self.decode(tgt_in, encoder_outputs, hidden, src_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def _embed(self, tokens):
        return self.position(self.embedding(tokens) * math.sqrt(self.d_model))

    @staticmethod
    def _causal_mask(size, device):
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def encode(self, src):
        src_key_padding_mask = src.eq(self.pad_id)
        src_emb = self._embed(src)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, tgt_in, memory, src_key_padding_mask):
        tgt_key_padding_mask = tgt_in.eq(self.pad_id)
        tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)
        tgt_emb = self._embed(tgt_in)
        output = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc(output)

    def forward(self, src, tgt_in):
        memory, src_key_padding_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_key_padding_mask)
