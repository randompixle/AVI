#!/usr/bin/env python3
import math
import os
import re
import sys
import time
from glob import glob

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
LOCAL_TEXT_GLOB = os.path.join("datasets", "text", "*.txt")
CACHE_DIR = os.path.join(MODEL_DIR, "cache")
CACHE_TOKENS = os.path.join(CACHE_DIR, "train_tokens.pt")
CACHE_META = os.path.join(CACHE_DIR, "train_tokens.meta")


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


def iter_local_texts():
    # Optional: mix in any local raw text files for more coverage.
    for path in sorted(glob(LOCAL_TEXT_GLOB)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        yield text
        except OSError:
            continue


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
        for text in iter_local_texts():
            f.write(text.replace("\n", " ").strip() + "\n")
            seen += 1
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
    for text in iter_local_texts():
        stream.extend(sp.encode(text, out_type=int))
        stream.append(sp.piece_to_id("\n"))
        seen += 1
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


def load_or_build_tokens(sp, max_samples, block_size):
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = f"samples={max_samples}|vocab={sp.get_piece_size()}|block={block_size}"
    if os.path.exists(CACHE_TOKENS) and os.path.exists(CACHE_META):
        try:
            with open(CACHE_META, "r", encoding="utf-8") as f:
                if f.read().strip() == meta:
                    return torch.load(CACHE_TOKENS)
        except OSError:
            pass
    data = build_token_stream(sp, max_samples)
    torch.save(data, CACHE_TOKENS)
    with open(CACHE_META, "w", encoding="utf-8") as f:
        f.write(meta)
    return data


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def main():
    experimental = "--experimental-testing" in sys.argv
    if experimental:
        sys.argv = [arg for arg in sys.argv if arg != "--experimental-testing"]
    staged = "--staged" in sys.argv
    if staged:
        sys.argv = [arg for arg in sys.argv if arg != "--staged"]

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else (2000 if experimental else 30000)
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 200000

    if not experimental:
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Colab/T4-friendly overrides via env vars (keeps root clean and avoids hardcoding).
    if experimental:
        block_size = int(os.getenv("AVI_BLOCK_SIZE", "128"))
        batch_size = int(os.getenv("AVI_BATCH_SIZE", "4"))
        n_embd = int(os.getenv("AVI_N_EMBD", "256"))
        n_head = int(os.getenv("AVI_N_HEAD", "8"))
        n_layer = int(os.getenv("AVI_N_LAYER", "6"))
        lr = float(os.getenv("AVI_LR", "3e-4"))
    else:
        block_size = int(os.getenv("AVI_BLOCK_SIZE", "512"))
        batch_size = int(os.getenv("AVI_BATCH_SIZE", "4"))
        n_embd = int(os.getenv("AVI_N_EMBD", "512"))
        n_head = int(os.getenv("AVI_N_HEAD", "8"))
        n_layer = int(os.getenv("AVI_N_LAYER", "10"))
        lr = float(os.getenv("AVI_LR", "6e-4"))
    grad_accum = int(os.getenv("AVI_GRAD_ACCUM", "1"))

    data = load_or_build_tokens(sp, max_samples, block_size)
    print(f"token stream size: {len(data)}")

    model = TinyGPT(
        vocab_size=sp.get_piece_size(),
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    start = time.time()
    model.train()
    total_steps = steps
    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(data, block_size, batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / grad_accum).backward()
            loss_accum += loss.item()
        # cosine decay
        t = step / max(total_steps - 1, 1)
        lr_t = lr * (0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * t))))
        for g in optimizer.param_groups:
            g["lr"] = lr_t
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            elapsed = time.time() - start
            rate = (step + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (total_steps - (step + 1)) / rate if rate > 0 else 0.0
            line = progress_bar(step + 1, total_steps, remaining)
            print(f"{line} loss {loss_accum/grad_accum:.4f} lr {lr_t:.2e}", end="\r", flush=True)
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
