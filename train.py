#!/usr/bin/env python3
import os
import re
import sys
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

from torch_model import TinyGPT


MODEL_DIR = "trained_models"
SPM_PREFIX = os.path.join(MODEL_DIR, "spm_complete")
SPM_MODEL = os.path.join(MODEL_DIR, "spm_complete.model")
MODEL_PATH = os.path.join(MODEL_DIR, "complete_transformer.pt")

# Plain-text corpus only (GPT-2 style). No chats, no prompts, no roles.
WIKITEXT_DATASET = ("wikitext", "wikitext-103-raw-v1")


# Strip assistant/QA contamination to keep the corpus purely narrative.
_CHAT_CONTAMINATION = re.compile(
    r"(?:^|[\s\[\(])("
    r"q:|a:|user:|assistant:|system:|options:|the answer is|true or false|how can i help|"
    r"you>|bot>|assistant>|<user>|<bot>|<assistant>|<system>|role:|instruction:|answer:)"
    r"(?:$|[\s\]\),.:;!?])",
    re.IGNORECASE,
)


def iter_wikitext(max_samples):
    try:
        from datasets import load_dataset
    except Exception:
        raise RuntimeError("Missing 'datasets' package. Install with: pip install datasets")
    ds = load_dataset(WIKITEXT_DATASET[0], WIKITEXT_DATASET[1], split="train")
    count = 0
    for row in ds:
        text = row.get("text", "").strip()
        if not text:
            continue
        if _CHAT_CONTAMINATION.search(text):
            continue
        yield text
        count += 1
        if max_samples is not None and count >= max_samples:
            return


def progress_bar(step, total, remaining, width=30, label=""):
    frac = step / total if total else 1.0
    filled = int(width * frac)
    bar = "=" * filled + "-" * (width - filled)
    eta = time.strftime("%M:%S", time.gmtime(remaining))
    prefix = f"{label} " if label else ""
    return f"{prefix}[{bar}] {step}/{total} ETA {eta}"


def build_spm_corpus(path, max_samples):
    with open(path, "w", encoding="utf-8") as f:
        seen = 0
        start = time.time()
        for text in iter_wikitext(max_samples):
            # Keep the tokenizer clean: raw text only, no chat markers or UI tokens.
            f.write(text.replace("\n", " ").strip() + "\n")
            seen += 1
            if seen % 200 == 0:
                elapsed = time.time() - start
                rate = seen / elapsed if elapsed > 0 else 0.0
                remaining = (max_samples - seen) / rate if rate > 0 else 0.0
                line = progress_bar(seen, max_samples, remaining, label="Corpus")
                print(line, end="\r", flush=True)
    print()


def build_token_stream(sp, max_samples):
    # GPT-2 style: single continuous stream, no chat turns.
    stream = []
    seen = 0
    start = time.time()
    for text in iter_wikitext(max_samples):
        stream.extend(sp.encode(text, out_type=int))
        stream.append(sp.piece_to_id("\n"))
        seen += 1
        if seen % 200 == 0:
            elapsed = time.time() - start
            rate = seen / elapsed if elapsed > 0 else 0.0
            remaining = (max_samples - seen) / rate if rate > 0 else 0.0
            line = progress_bar(seen, max_samples, remaining, label="Encoding")
            print(line, end="\r", flush=True)
    print()
    return torch.tensor(stream, dtype=torch.long)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 120000

    if steps < 10000:
        print("clamping steps to 10000 (min)")
        steps = 10000
    if steps > 38000:
        print("clamping steps to 38000 (max)")
        steps = 38000

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(SPM_MODEL):
        print("Training SentencePiece tokenizer (clean text only)...")
        corpus_path = os.path.join(MODEL_DIR, "spm_corpus_complete.txt")
        build_spm_corpus(corpus_path, max_samples)
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=SPM_PREFIX,
            vocab_size=vocab_size,
            model_type="bpe",
            bos_id=0,
            eos_id=1,
            unk_id=2,
            pad_id=3,
        )
    else:
        print("Reusing existing SentencePiece model.")

    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    print(f"spm vocab size: {sp.get_piece_size()}")
    data = build_token_stream(sp, max_samples)
    print(f"token stream size: {len(data)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    block_size = 256
    batch_size = 6
    n_embd = 384
    n_head = 8
    n_layer = 8

    model = TinyGPT(
        vocab_size=sp.get_piece_size(),
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    start = time.time()
    model.train()
    for step in range(steps):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            elapsed = time.time() - start
            rate = (step + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (steps - (step + 1)) / rate if rate > 0 else 0.0
            line = progress_bar(step + 1, steps, remaining)
            print(f"{line} loss {loss.item():.4f}", end="\r", flush=True)
    print()

    torch.save(
        {
            "model": model.state_dict(),
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "vocab_size": sp.get_piece_size(),
        },
        MODEL_PATH,
    )
    print(f"saved {MODEL_PATH} and {SPM_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
