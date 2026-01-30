import json
import time
from collections import Counter


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<bot>"]
SPECIAL_OFFSET = 256


def train_bpe(byte_seq, vocab_size, special_ids, progress_every=200, log_fn=None):
    vocab_size = max(vocab_size, 256 + len(special_ids))
    next_id = 256 + len(special_ids)
    merges = []
    total_merges = vocab_size - next_id
    start = time.time()

    seq = byte_seq[:]
    while next_id < vocab_size:
        pairs = Counter()
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a in special_ids or b in special_ids:
                continue
            pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        merges.append((a, b, next_id))
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(next_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq
        next_id += 1

        if progress_every and (len(merges) % progress_every == 0):
            elapsed = time.time() - start
            rate = len(merges) / elapsed if elapsed > 0 else 0.0
            remaining = (total_merges - len(merges)) / rate if rate > 0 else 0.0
            if log_fn:
                log_fn(len(merges), total_merges, remaining)

    return merges


def build_ranks(merges):
    ranks = {}
    for i, (a, b, new_id) in enumerate(merges):
        ranks[(a, b)] = (i, new_id)
    return ranks


def bpe_encode(tokens, ranks):
    while True:
        best = None
        best_rank = None
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in ranks:
                rank, new_id = ranks[pair]
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best = (pair, new_id)
        if best is None:
            break
        pair, new_id = best
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


def build_decoder(merges):
    decoder = {}
    for a, b, new_id in merges:
        decoder[new_id] = (a, b)
    return decoder


def decode_tokens(tokens, decoder):
    out = []
    stack = list(tokens)
    while stack:
        t = stack.pop(0)
        if t < 256:
            out.append(t)
        elif t in decoder:
            a, b = decoder[t]
            stack = [a, b] + stack
        else:
            continue
    return bytes(out)


def save_bpe(path, merges, vocab_size):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"merges": merges, "vocab_size": vocab_size, "special": SPECIAL_TOKENS}, f)


def load_bpe(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merges = [tuple(m) for m in data["merges"]]
    return merges, data["vocab_size"], data["special"]
