# AVI
An experimental, local language model project.

## What this is
- Base model is a **pure completion LM** (SentencePiece + decoder-only Transformer).
- Optional **small chat finetune** that adds turn-taking without poisoning the base model.

## Quick start

Train the base completion model (pure text):

```bash
python3 train.py 30000 12000 120000
```

Chat wrapper (base model):

```bash
python3 chat.py --chat
```

Small chat finetune (keep base intact):

```bash
python3 finetune_chat.py 20000 90000 10000 0.02
python3 chat.py --chat trained_models/chat_transformer.pt
```

## Files
- `train.py` pure completion training (WikiText)
- `finetune_chat.py` light chat finetune (2–5% chat)
- `chat.py` completion + `--chat` wrapper
- `compare_gpt2.py` side-by-side GPT‑2 vs our model

## Notes
- Models and datasets are **ignored** by git (see `.gitignore`).
- Use Colab for longer training if needed.
