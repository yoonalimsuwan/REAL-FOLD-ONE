"""HCSN CRISPR off-target predictor — input: 23-nt guide + 23-nt
candidate site (encoded as 2 channels), output: off-target probability.
Domain meaning of HCSN: weak measurement ≈ 'soft' mismatch readout that
doesn't fully discard partial matches — exactly the biology of off-targets.
Developers : PAI AND Yoon A Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026
"""
import torch
import torch.nn as nn

# [B2 FIX] Fallback definitions if model.py is not available
try:
    from model import RMSNorm, HCSNPass
except ImportError:
    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization."""
        def __init__(self, d_model: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d_model))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x / rms * self.weight

    class HCSNPass(nn.Module):
        """HCSN weak-measurement pass: gated soft attention on features."""
        def __init__(self, d_model: int):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
            self.transform = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            g = self.gate(x)
            return x + g * self.transform(x)


class OffTargetNet(nn.Module):
    # [B2a FIX] def init → def __init__
    def __init__(self, d_model=128, n_layers=4, n_heads=4, seq=23):
        # [B2b FIX] super().init() → super().__init__()
        super().__init__()
        self.embed = nn.Embedding(5, d_model)     # A,C,G,T,+mismatch-channel
        self.pos = nn.Parameter(torch.randn(1, seq, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, n_layers)
        self.hcsn = HCSNPass(d_model)             # hcsn_every=1: every layer
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def encode_pair(self, guide: str, site: str) -> torch.Tensor:
        b = {"A": 0, "C": 1, "G": 2, "T": 3}
        ids = []
        for g, s in zip(guide.upper(), site.upper()):
            if g in b and s in b:
                ids.append(b[s] + (4 if g != s else 0))  # +4 marks mismatch
            else:
                ids.append(0)  # unknown character → default to A
        return torch.tensor(ids)

    def forward(self, ids):                        # (B, 23)
        x = self.embed(ids) + self.pos
        for blk in self.backbone.layers:
            x = blk(x)
            x = self.hcsn(x)                       # always on: soft mismatch readout
        return self.head(self.norm(x.mean(dim=1))).squeeze(-1)


# --- smoke test ---
if __name__ == "__main__":
    torch.manual_seed(0)
    m = OffTargetNet()
    # [B2c/d FIX] removed trailing spaces from test strings
    on  = m.encode_pair("GAGTCCGAGCATAAAGCGC", "GAGTCCGAGCATAAAGCGC")[:23]
    off = m.encode_pair("GAGTCCGAGCATAAAGCGC", "GAGTCCGAGCATAAAGCGT")[:23]
    ids = torch.stack([on, off])
    logits = m(ids)
    logits.sum().backward()
    print("off-target OK:", logits.shape, "| params:", sum(p.numel() for p in m.parameters()))
    print(f"  perfect match logit: {logits[0].item():.4f}")
    print(f"  mismatch logit:      {logits[1].item():.4f}")
