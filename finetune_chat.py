#!/usr/bin/env python3
import json
import os
import sys
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

from torch_model import TinyGPT

MODEL_DIR = "trained_models"
SPM_MODEL = os.path.join(MODEL_DIR, "spm_complete.model")
BASE_MODEL = os.path.join(MODEL_DIR, "complete_transformer.pt")
CHAT_MODEL = os.path.join(MODEL_DIR, "chat_transformer.pt")

OASST_JSONL = os.path.join("datasets", "OpenAssistant", "2023-04-12_oasst_ready.messages.jsonl")
OASST_GZ = os.path.join("datasets", "OpenAssistant", "2023-04-12_oasst_ready.messages.jsonl.gz")
OASST_PATH = OASST_JSONL if os.path.exists(OASST_JSONL) else OASST_GZ

WIKITEXT_DATASET = ("wikitext", "wikitext-103-raw-v1")

BAD_PATTERNS = (
    "as an ai",
    "as a language model",
    "language model",
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i'm not sure",
    "i am not sure",
    "how can i help",
    "i do not have opinions",
    "i can't",
    "i cannot",
    "i don't know",
    "i do not know",
    "not sure",
    "how can i assist",
    "how may i help",
    "any other questions",
    "let me know if you have any other questions",
    "i am a human",
    "world war",
    "i'm here to help",
    "i am here to help",
    "i'm here to assist",
    "i am here to assist",
    "how can i assist you",
    "how can i help you",
    "open assistant",
    "i'm here to help",
    "i am here to help",
    "i'm here to assist",
    "i am here to assist",
    "is there anything else i can help",
    "is there anything else i can assist",
)

SOCIAL_DEFAULTS = [
    ("hello", "Hi."),
    ("hi", "Hello."),
    ("hey", "Hi."),
    ("yo", "Hey."),
    ("how are you?", "Doing well. How about you?"),
    ("how are you doing?", "Doing well. How about you?"),
    ("thanks", "You're welcome."),
    ("thank you", "You're welcome."),
    ("ok", "Got it."),
    ("okay", "Understood."),
    ("cool", "Great."),
    ("nice", "Sounds good."),
    ("huh?", "Could you clarify?"),
    ("what?", "Could you clarify?"),
    ("why?", "Could you clarify your question?"),
    ("help", "What do you need help with?"),
    ("bye", "Bye."),
    ("what is a chair?", "A chair is a piece of furniture designed for sitting."),
    ("what is water?", "Water is a clear liquid made of hydrogen and oxygen."),
    ("what is a dog?", "A dog is a domesticated animal kept as a pet."),
    ("what is a computer?", "A computer is a device that processes information."),
    ("what is a phone?", "A phone is a device used to communicate over distance."),
    ("what is the sun?", "The Sun is the star at the center of our solar system."),
]


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
        # Drop WikiText section headings and markup that leak into chat.
        if "==" in text or "= = =" in text:
            continue
        yield text
        count += 1
        if max_samples is not None and count >= max_samples:
            return


def iter_oasst_pairs(max_pairs):
    if not os.path.exists(OASST_PATH):
        return
    if OASST_PATH.endswith(".gz"):
        import gzip
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda p: open(p, "r", encoding="utf-8")

    with opener(OASST_PATH) as f:
        messages = []
        for line in f:
            obj = json.loads(line)
            if obj.get("lang") != "en":
                continue
            if obj.get("deleted"):
                continue
            if not obj.get("text"):
                continue
            messages.append(obj)

    by_id = {m["message_id"]: m for m in messages if "message_id" in m}

    count = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        parent_id = m.get("parent_id")
        if not parent_id:
            continue
        parent = by_id.get(parent_id)
        if not parent or parent.get("role") != "prompter":
            continue
        user = parent.get("text", "").strip()
        assistant = m.get("text", "").strip()
        if not user or not assistant:
            continue
        low = assistant.lower()
        if any(p in low for p in BAD_PATTERNS):
            continue
        if len(assistant.split()) > 18:
            continue
        if " = = =" in assistant or "==" in assistant:
            continue
        yield (user, assistant)
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


def mix_streams(raw_iter, chat_iter, chat_ratio=0.1):
    # pattern: 9 raw / 1 chat for 10% chat
    raw_count = int((1 - chat_ratio) * 10)
    chat_count = 10 - raw_count
    pattern = ["r"] * max(raw_count, 1) + ["c"] * max(chat_count, 1)

    while True:
        for p in pattern:
            try:
                if p == "r":
                    yield ("raw", next(raw_iter))
                else:
                    yield ("chat", next(chat_iter))
            except StopIteration:
                return


def build_stream(sp, max_raw, max_chat, chat_ratio=0.1):
    stream = []
    raw_iter = iter_wikitext(max_raw)
    chat_iter = iter_oasst_pairs(max_chat)
    mixed = mix_streams(iter(raw_iter), iter(chat_iter), chat_ratio=chat_ratio)

    seen = 0
    start = time.time()
    total = max_raw + max_chat
    for kind, item in mixed:
        if kind == "raw":
            text = item
            stream.extend(sp.encode(text, out_type=int))
            stream.append(sp.piece_to_id("\n"))
        else:
            u, a = item
            text = f"<|user|> {u}\n<|assistant|> {a}\n"
            stream.extend(sp.encode(text, out_type=int))
            stream.append(sp.piece_to_id("\n"))
        seen += 1
        if seen % 200 == 0:
            elapsed = time.time() - start
            rate = seen / elapsed if elapsed > 0 else 0.0
            remaining = (total - seen) / rate if rate > 0 else 0.0
            mins, secs = divmod(int(remaining), 60)
            print(f"[{'='*int(30*seen/total):<30}] {seen}/{total} ETA {mins:02d}:{secs:02d}", end="\r", flush=True)
    print()
    # Inject a tiny social-default set so greetings don't drift into encyclopedia tone.
    for u, a in SOCIAL_DEFAULTS:
        text = f"<|user|> {u}\n<|assistant|> {a}\n"
        stream.extend(sp.encode(text, out_type=int))
        stream.append(sp.piece_to_id("\n"))
    return torch.tensor(stream, dtype=torch.long)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    max_raw = int(sys.argv[2]) if len(sys.argv) > 2 else 90000
    max_chat = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    chat_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

    if steps < 5000:
        steps = 5000
    if chat_ratio > 0.1:
        chat_ratio = 0.1

    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(BASE_MODEL, map_location=device, weights_only=True)
    # Allow Colab/T4 overrides via env vars.
    block_size = int(os.getenv("AVI_BLOCK_SIZE", str(ckpt.get("block_size", 256))))
    n_embd = int(os.getenv("AVI_N_EMBD", str(ckpt.get("n_embd", 384))))
    n_head = int(os.getenv("AVI_N_HEAD", str(ckpt.get("n_head", 8))))
    n_layer = int(os.getenv("AVI_N_LAYER", str(ckpt.get("n_layer", 8))))
    batch_size = int(os.getenv("AVI_BATCH_SIZE", "6"))

    model = TinyGPT(
        vocab_size=ckpt.get("vocab_size", sp.get_piece_size()),
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    print("building mixed stream...")
    data = build_stream(sp, max_raw, max_chat, chat_ratio=chat_ratio)
    print(f"token stream size: {len(data)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    start = time.time()
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
            mins, secs = divmod(int(remaining), 60)
            print(f"[{(step+1):>6}/{steps}] ETA {mins:02d}:{secs:02d} loss {loss.item():.4f}", end="\r", flush=True)
    print()

    torch.save(
        {
            "model": model.state_dict(),
            "block_size": ckpt.get("block_size", 256),
            "n_embd": ckpt.get("n_embd", 384),
            "n_head": ckpt.get("n_head", 8),
            "n_layer": ckpt.get("n_layer", 8),
            "vocab_size": sp.get_piece_size(),
        },
        CHAT_MODEL,
    )
    print(f"saved {CHAT_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
