"""
Transformer-based WAF Model
Classifies HTTP requests as: normal, sqli, xss, path_traversal, cmd_injection
"""

import torch
import torch.nn as nn
import math
import json
import os

# ── Vocabulary ──────────────────────────────────────────────────────────────
VOCAB = list(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\t\n"
)
VOCAB = ["<PAD>", "<UNK>", "<CLS>"] + VOCAB
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

LABELS = ["normal", "sqli", "xss", "path_traversal", "cmd_injection"]
NUM_CLASSES = len(LABELS)
MAX_LEN = 256


def encode(text: str) -> list[int]:
    tokens = [CHAR2IDX["<CLS>"]] + [
        CHAR2IDX.get(c, CHAR2IDX["<UNK>"]) for c in text[:MAX_LEN - 1]
    ]
    tokens += [CHAR2IDX["<PAD>"]] * (MAX_LEN - len(tokens))
    return tokens


# ── Positional Encoding ──────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LEN, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Transformer WAF Model ────────────────────────────────────────────────────
class WAFTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (batch, seq_len)
        emb = self.pos_enc(self.embedding(x))          # (batch, seq, d_model)
        out = self.transformer(emb, src_key_padding_mask=mask)  # (batch, seq, d_model)
        cls = out[:, 0, :]                              # CLS token
        return self.classifier(cls)                     # (batch, num_classes)


# ── Inference Helper ─────────────────────────────────────────────────────────
class WAFInferencer:
    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = WAFTransformer().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        self.model.eval()

    @torch.no_grad()
    def predict(self, request_text: str) -> dict:
        tokens = torch.tensor([encode(request_text)], dtype=torch.long).to(self.device)
        pad_mask = (tokens == 0)
        logits = self.model(tokens, pad_mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        idx = int(torch.argmax(logits, dim=-1).item())
        confidence = probs[idx]
        label = LABELS[idx]
        return {
            "label": label,
            "is_attack": label != "normal",
            "confidence": round(confidence, 4),
            "probabilities": {l: round(p, 4) for l, p in zip(LABELS, probs)},
        }
