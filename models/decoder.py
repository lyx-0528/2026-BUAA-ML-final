import torch
import torch.nn as nn
import torch.nn.functional as F

from config import D_MODEL, NUM_HEADS, NUM_DECODER_LAYERS, NUM_QUERIES, NUM_CHARS, DROPOUT, PATCH_SIZE, IMG_SIZE


def sinusoidal_position_embedding(num_positions, d_model, temperature=10000):
    position = torch.arange(num_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(temperature)) / d_model))
    pe = torch.zeros(num_positions, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, nhead: int = NUM_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout(self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)[0])

        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout(self.cross_attn(tgt2, memory, memory)[0])

        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout_ffn(self.linear2(F.gelu(self.linear1(tgt2))))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = NUM_DECODER_LAYERS,
        d_model: int = D_MODEL,
        nhead: int = NUM_HEADS,
        num_queries: int = NUM_QUERIES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        self.query_tokens = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.query_pos = nn.Parameter(torch.zeros(num_queries, d_model))
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory, return_aux=False):
        batch_size = memory.size(0)

        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.query_pos.unsqueeze(0)

        aux_outputs = []
        for layer in self.layers:
            queries = layer(queries, memory)
            if return_aux:
                aux_outputs.append(self.norm(queries))

        queries = self.norm(queries)

        if return_aux:
            return queries, aux_outputs
        return queries


class ColorHead(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class CharHead(nn.Module):
    def __init__(self, d_model: int = D_MODEL, num_chars: int = NUM_CHARS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_chars),
        )

    def forward(self, x):
        return self.mlp(x)
