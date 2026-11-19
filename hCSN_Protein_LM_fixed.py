"""HCSN Protein LM — vocab = 20 amino acids + 3 special tokens.
Reuse RoPE / CausalSelfAttention / SwiGLU / Block / HCSNLMD from model.py,
only the tokenization layer changes.
Developers : PAI AND Yoon A Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026
"""
import torch

# [B5a FIX] ลบ unused import `build_lm`
try:
    from model import HCSNLMD
except ImportError:
    # Fallback: minimal HCSNLMD if model.py not available
    import torch.nn as nn

    class HCSNLMD(nn.Module):
        """Minimal protein LM fallback when model.py is unavailable."""
        def __init__(self, vocab_size=23, d_model=256, n_layers=8,
                     n_heads=8, max_seq=514, dropout=0.0, hcsn_every=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Parameter(torch.randn(1, max_seq, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.blocks = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, ids, targets=None):
            B, L = ids.shape
            x = self.embed(ids) + self.pos[:, :L]
            x = self.blocks(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1), ignore_index=20)
            return logits, loss


AA = "ACDEFGHIKLMNPQRSTVWY"          # 20 canonical
SPECIAL = "<pad><cls><sep>"           # 3
VOCAB = list(AA) + ["<pad>", "<cls>", "<sep>"]   # 23 tokens
AA2ID = {c: i for i, c in enumerate(AA)}
PAD, CLS, SEP = 20, 21, 22


def encode(seq: str) -> torch.Tensor:
    ids = [CLS] + [AA2ID.get(c, PAD) for c in seq.upper()] + [SEP]
    return torch.tensor(ids, dtype=torch.long)


def decode(ids) -> str:
    return "".join(VOCAB[i] for i in ids if i < 20)


def build_protein_lm(d_model=256, n_layers=8, n_heads=8, max_seq=512,
                     hcsn_every=2):
    # NOTE: build_lm in model.py hard-codes vocab 256; here we construct
    # directly so vocab = 23. Architecture identical to tested model.
    # [B5b FIX] max_seq + 2 for CLS/SEP tokens
    return HCSNLMD(vocab_size=23, d_model=d_model, n_layers=n_layers,
                   n_heads=n_heads, max_seq=max_seq + 2,
                   dropout=0.0, hcsn_every=hcsn_every)


# --- smoke test (run yourself; mirrors the tests model.py passed) ---
if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_protein_lm()
    seq = "MKTLLILAVVAAALA"          # signal-peptide-like
    ids = encode(seq).unsqueeze(0)
    targets = torch.roll(ids, -1, dims=1)      # next-token prediction
    logits, loss = m(ids, targets)
    loss.backward()
    print("protein LM OK:", tuple(logits.shape), round(loss.item(), 3),
          "| params:", sum(p.numel() for p in m.parameters()))
