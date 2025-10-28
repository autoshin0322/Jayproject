# posevit/model.py
import torch, torch.nn as nn, math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):  # x: (B,T,D)
        return x + self.pe[:, :x.size(1)]

class PoseSeqTransformer(nn.Module):
    def __init__(self, d_in, d_model=128, nhead=4, num_layers=4, num_classes=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos  = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):           # x: (B,T,F)
        h = self.proj(x)            # (B,T,D)
        h = self.pos(h)
        h = self.encoder(h)         # (B,T,D)
        logits = self.cls(h)        # (B,T,C)
        return logits
