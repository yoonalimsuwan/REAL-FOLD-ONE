"""generate.py — checkpoint load + interactive completion.

Developers   : PAI AND Yoon A Limsuwan
Organization : MSPS NETWORK
Licence      : MIT
Year         : 2026
"""

from __future__ import annotations

import argparse
import torch

from model import build_lm
from data import ByteDetokenizer


def load_model(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    model = build_lm(a["model_size"], vocab_size=256, max_seq=a["seq_len"],
                     hcsn_every=a["hcsn_every"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), a


@torch.no_grad()
def complete(model, prompt: str, max_new: int = 80, temperature: float = 0.8,
             top_k: int = 40, top_p: float = 0.95, device: str = "cpu"):
    ids = torch.frombuffer(bytearray(prompt.encode("utf-8")),
                           dtype=torch.uint8).long()
    ids = ids[:model.max_seq - max_new - 1].unsqueeze(0).to(device)
    out = model.generate(ids, max_new=max_new, temperature=temperature,
                         top_k=top_k, top_p=top_p)
    return ByteDetokenizer().decode(out[0].tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="The cloud ")
    p.add_argument("--max_new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    model, a = load_model(args.ckpt)
    print(complete(model, args.prompt, args.max_new, args.temperature))
