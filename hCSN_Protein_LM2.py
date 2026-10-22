"""HCSN Protein LM — next-token on amino acid sequences.
Domain meaning of HCSN: fractal scales ≈ residue 'depth' (surface->core),
weak measurement ≈ non-destructive readout of local structure propensity.

Developers : PAI AND Yoon A Limsuwan 
Organization : MSPS NETWORK 
Licence : MIT 
Year : 2026

"""
import torch
from model import HCSNLMD

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2ID = {c: i for i, c in enumerate(AA)}
PAD, CLS, SEP = 20, 21, 22
VOCAB = list(AA) + ["<pad>", "<cls>", "<sep>"]

def encode(seq: str) -> torch.Tensor:
    return torch.tensor([CLS] + [AA2ID.get(c, PAD) for c in seq.upper()] + [SEP])

def decode(ids) -> str:
    return "".join(VOCAB[i] for i in ids if i < 20)

def build_protein_lm(d_model=256, n_layers=8, n_heads=8, max_seq=512, hcsn_every=2):
    return HCSNLMD(vocab_size=23, d_model=d_model, n_layers=n_layers,
                   n_heads=n_heads, max_seq=max_seq, hcsn_every=hcsn_every)

# --- smoke test ---
if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_protein_lm()
    ids = encode("MKTLLILAVVAAALA").unsqueeze(0)
    targets = torch.roll(ids, -1, dims=1)
    logits, loss = m(ids, targets)
    loss.backward()
    print("protein OK:", tuple(logits.shape), round(loss.item(), 3),
          "| params:", sum(p.numel() for p in m.parameters()))
