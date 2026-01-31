#!/usr/bin/env python3
import os
import re
import sys

START_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK", re.IGNORECASE)


def clean_text(text: str) -> str:
    lines = text.splitlines()
    start = 0
    end = len(lines)
    for i, line in enumerate(lines):
        if START_RE.search(line):
            start = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if END_RE.search(lines[i]):
            end = i
            break
    body = lines[start:end]
    # remove extra blank lines
    cleaned = "\n".join([ln.rstrip() for ln in body]).strip()
    return cleaned


def main():
    if len(sys.argv) < 2:
        print("usage: strip_gutenberg.py <dir>")
        return 1
    root = sys.argv[1]
    for name in os.listdir(root):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            cleaned = clean_text(raw)
            with open(path, "w", encoding="utf-8") as f:
                f.write(cleaned + "\n")
        except OSError:
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
