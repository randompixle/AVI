#!/usr/bin/env python3
import os
import sys
import time

import sentencepiece as spm
import torch

from torch_model import TinyGPT


MODEL_DIR = "trained_models"
SPM_MODEL = os.path.join(MODEL_DIR, "spm_complete.model")
MODEL_PATH = os.path.join(MODEL_DIR, "complete_transformer.pt")


def sample_next(probs, idx_history, top_k=40, rep_penalty=1.2, banned_ids=None):
    if idx_history:
        for idx in set(idx_history):
            probs[idx] /= rep_penalty
        probs = probs / probs.sum()
    if banned_ids:
        probs[list(banned_ids)] = 0
        s = probs.sum()
        if s > 0:
            probs = probs / s
    if top_k is not None and top_k > 0:
        topk = torch.topk(probs, k=min(top_k, probs.numel()))
        mask = torch.zeros_like(probs)
        mask[topk.indices] = topk.values
        probs = mask / mask.sum()
    return int(torch.multinomial(probs, 1).item())


def sample_next_top_p(probs, top_p=0.95):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    cutoff = cumulative > top_p
    if cutoff.any():
        first = int(cutoff.nonzero(as_tuple=False)[0].item())
        sorted_probs[first + 1:] = 0
    if sorted_probs.sum() > 0:
        sorted_probs = sorted_probs / sorted_probs.sum()
    choice = int(torch.multinomial(sorted_probs, 1).item())
    return int(sorted_idx[choice].item())


def generate(model, sp, prompt, max_new=50, temperature=0.9, ctx_limit=256, top_k=50, top_p=0.95, rep_penalty=1.1, progress=False, min_new=0):
    # GPT-3 style: the model is a raw completion engine; chat is external prompt framing.
    data = [sp.piece_to_id("<bos>")]
    prompt_ids = sp.encode(prompt, out_type=int)
    data.extend(prompt_ids)

    start = time.time()
    spinner = ["|", "/", "-", "\\"]
    short_prompt = len(prompt.split()) <= 3
    max_new = min(max_new, 30) if short_prompt else max_new

    banned_ids = set()
    newline_ids = set()

    for i in range(max_new):
        ctx = data[-ctx_limit:] if data else [sp.piece_to_id("<bos>")]
        x = torch.tensor(ctx, dtype=torch.long, device=next(model.parameters()).device)[None, :]
        logits = model(x)
        last = logits[0, -1] / max(temperature, 1e-6)
        probs = torch.softmax(last, dim=-1)
        # Apply repetition penalty before sampling to reduce loops while keeping raw text style.
        if data:
            for idx in set(data[-50:]):
                probs[idx] /= rep_penalty
            probs = probs / probs.sum()
        if top_k is not None and top_k > 0:
            topk = torch.topk(probs, k=min(top_k, probs.numel()))
            mask = torch.zeros_like(probs)
            mask[topk.indices] = topk.values
            probs = mask / mask.sum()
        if top_p is not None and top_p < 1.0:
            idx = sample_next_top_p(probs, top_p=top_p)
        else:
            idx = sample_next(probs, data[-50:], top_k=None, rep_penalty=rep_penalty, banned_ids=banned_ids)
        data.append(idx)
        if (i + 1) >= min_new and (idx == sp.piece_to_id("<eos>") or idx in newline_ids):
            break
        if progress and (i % 5 == 0):
            elapsed = time.time() - start
            s = spinner[(i // 5) % len(spinner)]
            sys.stdout.write(f"\r{s} generating... {elapsed:.2f}s")
            sys.stdout.flush()
    if progress:
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

    gen_tokens = data[len(prompt_ids) + 1 :]
    out = sp.decode([t for t in gen_tokens if t not in {
        sp.piece_to_id("<bos>"),
        sp.piece_to_id("<eos>"),
        sp.piece_to_id("<pad>")
    }]).strip()

    return out


def main():
    default_temp = 0.65
    default_max_new = 120
    default_ctx = 256
    default_top_k = 50
    default_top_p = 0.9
    default_rep = 1.15

    use_chat = "--chat" in sys.argv
    if use_chat:
        sys.argv = [arg for arg in sys.argv if arg != "--chat"]

    argi = 1
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".pt"):
        model_path = sys.argv[1]
        argi = 2
    else:
        model_path = MODEL_PATH

    temperature = float(sys.argv[argi]) if len(sys.argv) > argi else default_temp
    max_new = int(sys.argv[argi + 1]) if len(sys.argv) > argi + 1 else default_max_new
    ctx_len = int(sys.argv[argi + 2]) if len(sys.argv) > argi + 2 else default_ctx
    top_k = int(sys.argv[argi + 3]) if len(sys.argv) > argi + 3 else default_top_k
    top_p = float(sys.argv[argi + 4]) if len(sys.argv) > argi + 4 else default_top_p
    rep_penalty = float(sys.argv[argi + 5]) if len(sys.argv) > argi + 5 else default_rep

    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = TinyGPT(
        vocab_size=ckpt.get("vocab_size", sp.get_piece_size()),
        block_size=ckpt.get("block_size", 128),
        n_embd=ckpt.get("n_embd", 256),
        n_head=ckpt.get("n_head", 8),
        n_layer=ckpt.get("n_layer", 6),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("SentencePiece Transformer (raw completion)" if not use_chat else "SentencePiece Transformer (chat wrapper)")
    while True:
        try:
            user = input("you> " if use_chat else "").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue

        if use_chat:
            # GPT-3-style controller: strong document lock + asymmetric roles + one example.
            lock = (
                "This is a conversation. You are an AI that replies directly and briefly. "
                "Do not continue articles, do not invent speakers, and do not write anything except the reply."
            )
            example = "<|user|> Hello!\n<|assistant|> Hi there."
            prompt = f"{lock}\n\n{example}\n\n<|user|> {user}\n<|assistant|>"
        else:
            prompt = user

        start = time.time()
        # Ensure non-empty output by retrying a few times.
        reply = ""
        for _ in range(5):
            reply = generate(
                model,
                sp,
                prompt,
                max_new=max_new,
                temperature=temperature,
                ctx_limit=min(ctx_len, model.block_size),
                top_k=top_k,
                top_p=top_p,
                rep_penalty=rep_penalty,
                progress=True,
                min_new=6 if use_chat else 0,
            )
            if reply.strip():
                break
        _ = time.time() - start
        if use_chat:
            for stop in ("\n<|user|>", "\n<|assistant|>", "\n\n"):
                cut = reply.find(stop)
                if cut > 0:
                    reply = reply[:cut].rstrip()
                    break
            # Cut heading-like junk that sometimes appears in WikiText.
            head = reply.find("==")
            if head > 0:
                reply = reply[:head].rstrip()
            reply = reply.lstrip(" \t\n-:;,.\"'()`")
            print(f"bot> {reply}\n")
        else:
            print(reply)


if __name__ == "__main__":
    raise SystemExit(main())
