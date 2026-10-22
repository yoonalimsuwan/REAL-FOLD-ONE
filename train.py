"""train.py — single-GPU / DDP training loop with cosine LR + checkpointing.

Developers   : PAI AND Yoon A Limsuwan
Organization : MSPS NETWORK
Licence      : MIT
Year         : 2026
"""

from __future__ import annotations

import argparse
import math
import os
import time
import torch
from torch.utils.data import DataLoader

from model import build_lm
from data import PackedTokenDataset


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model_size", default="debug")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--ckpt_dir", default="./checkpoints")
    p.add_argument("--hcsn_every", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    return p


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PackedTokenDataset(args.data, seq_len=args.seq_len)
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, generator=g)

    model = build_lm(args.model_size, vocab_size=256, max_seq=args.seq_len,
                     hcsn_every=args.hcsn_every).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[train] {args.model_size} | {nparams/1e6:.2f}M params | {device}")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, args.warmup, args.steps))

    os.makedirs(args.ckpt_dir, exist_ok=True)
    step, t0, running = 0, time.time(), None
    model.train()

    while step < args.steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1

            running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()
            if step % args.log_every == 0:
                print(f"[step {step:6d}] loss={running:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}")
            if step % args.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args)},
                           os.path.join(args.ckpt_dir, "hcsn_lm.pt"))
            if step >= args.steps:
                break

    torch.save({"model": model.state_dict(), "step": step, "args": vars(args)},
               os.path.join(args.ckpt_dir, "hcsn_lm.pt"))
    print(f"[train] done. final loss: {running:.4f}")


if __name__ == "__main__":
    main()
