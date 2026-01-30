#!/usr/bin/env python3
import argparse
import time

import torch
import sentencepiece as spm
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch_model import TinyGPT

MODEL_DIR = "trained_models"
OUR_SPM = f"{MODEL_DIR}/spm_complete.model"
OUR_MODEL = f"{MODEL_DIR}/complete_transformer.pt"

PROMPTS = [
    "The city was silent except for",
    "I didnt realize anything was wrong until",
    "By the time the train arrived,",
    "The old computer in the basement still worked, but",
    "Nobody remembered who started the tradition, only that",
    "The internet became self-aware at 3:14 a.m.",
    "In the future, punctuation will be illegal because",
    "The manual clearly stated that time travel was forbidden, yet",
    "Every mirror in the building displayed a different version of",
    "The Roman Empire never officially ended because",
    "Early computers were dangerous not because of electricity, but",
    "Historians still argue whether the event actually happened, since",
    "The experiment failed in an unexpected way when",
    "$ make all\ngcc -Wall -O2 main.c -o app",
    "[ERROR] Failed to initialize subsystem:",
    "2023-11-04 22:13:51 WARNING:",
    "Loading modules...\n\u2713 core\n\u2713 renderer",
    "The rules were simple:\n1.",
    "There were three things he always carried with him:",
    "The letter contained only one sentence:",
    "She wrote the same word on every page:",
]


def load_our_model(device):
    sp = spm.SentencePieceProcessor()
    sp.load(OUR_SPM)

    ckpt = torch.load(OUR_MODEL, map_location=device, weights_only=True)
    model = TinyGPT(
        vocab_size=ckpt.get("vocab_size", sp.get_piece_size()),
        block_size=ckpt.get("block_size", 256),
        n_embd=ckpt.get("n_embd", 384),
        n_head=ckpt.get("n_head", 8),
        n_layer=ckpt.get("n_layer", 8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, sp


def sample_top_p(probs, top_p=0.95):
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


def generate_our(model, sp, prompt, max_new=120, temperature=0.9, top_k=50, top_p=0.95, rep_penalty=1.1):
    data = [sp.piece_to_id("<bos>")]
    data.extend(sp.encode(prompt, out_type=int))
    device = next(model.parameters()).device

    for _ in range(max_new):
        ctx = data[-model.block_size :]
        x = torch.tensor(ctx, dtype=torch.long, device=device)[None, :]
        logits = model(x)
        last = logits[0, -1] / max(temperature, 1e-6)
        probs = torch.softmax(last, dim=-1)
        # repetition penalty
        for idx in set(data[-50:]):
            probs[idx] /= rep_penalty
        probs = probs / probs.sum()
        # top-k filter
        if top_k is not None and top_k > 0:
            topk = torch.topk(probs, k=min(top_k, probs.numel()))
            mask = torch.zeros_like(probs)
            mask[topk.indices] = topk.values
            probs = mask / mask.sum()
        # top-p sample
        idx = sample_top_p(probs, top_p=top_p)
        data.append(idx)
        if idx == sp.piece_to_id("<eos>"):
            break

    return sp.decode([t for t in data if t not in {
        sp.piece_to_id("<bos>"),
        sp.piece_to_id("<eos>"),
        sp.piece_to_id("<pad>")
    }]).strip()


def generate_gpt2(model, tok, prompt, max_new=120, temperature=0.9, top_k=50, top_p=0.95):
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return tok.decode(out[0], skip_special_tokens=True)


def talk_loop(kind, device, max_new, temperature, top_k, top_p, rep_penalty):
    if kind == "gpt2":
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        model.eval()
        while True:
            try:
                prompt = input().strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not prompt:
                continue
            out = generate_gpt2(model, tok, prompt, max_new=max_new, temperature=temperature, top_k=top_k, top_p=top_p)
            print(out)
    else:
        model, sp = load_our_model(device)
        while True:
            try:
                prompt = input().strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not prompt:
                continue
            out = generate_our(model, sp, prompt, max_new=max_new, temperature=temperature, top_k=top_k, top_p=top_p, rep_penalty=rep_penalty)
            print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--talk", choices=["gpt2", "ours"], help="interactive mode with a single model")
    parser.add_argument("--max-new", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rep-penalty", type=float, default=1.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.talk:
        return talk_loop(
            args.talk,
            device,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
        )

    print("loading gpt2...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gpt2.eval()

    print("loading our model...")
    our_model, sp = load_our_model(device)

    for i, prompt in enumerate(PROMPTS, 1):
        print("=" * 80)
        print(f"Prompt {i}: {prompt}")

        t0 = time.time()
        ours = generate_our(
            our_model,
            sp,
            prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
        )
        t1 = time.time()
        gpt2_out = generate_gpt2(
            gpt2,
            tok,
            prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        t2 = time.time()

        print("\n-- Our model --")
        print(ours)
        print(f"[elapsed {t1 - t0:.2f}s]")

        print("\n-- GPT-2 --")
        print(gpt2_out)
        print(f"[elapsed {t2 - t1:.2f}s]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
