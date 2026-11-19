"""Shared pieces: dataset, prompt, and the scoring function used by every model."""
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"          # the small model (0.5 billion params)
ADAPTER_DIR = ROOT / "models" / "lora-adapter"       # where the LoRA weights are saved
REPORTS = ROOT / "reports"
LABELS = ["clean", "offensive"]                      # 0 = clean, 1 = offensive

SYSTEM = "You are a strict chat moderator for an online gaming community."
INSTRUCTION = (
    "Is this chat message offensive? Offensive means insults, slurs, harassment, "
    "threats or profanity. Answer with exactly one word: offensive or clean.\n\n"
    "Message: {text}"
)


def load_data():
    """OffensEval (tweet_eval/offensive): short, messy, informal messages like game chat.
    train 11,916 | validation 1,324 | test 860 — all labelled by humans."""
    return load_dataset("cardiffnlp/tweet_eval", "offensive")


def messages(text: str):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION.format(text=text)}]


def prompt_text(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(messages(text), tokenize=False,
                                         add_generation_prompt=True)


def answer_token_ids(tokenizer):
    """First token of each answer word — we compare their probabilities."""
    return [tokenizer.encode(w, add_special_tokens=False)[0] for w in LABELS]


@torch.no_grad()
def score_local(model, tokenizer, texts, batch_size=32):
    """Return P(offensive) for each text, read straight from the model's next-token logits.
    No text generation needed: one forward pass per batch -> fast and deterministic."""
    tokenizer.padding_side = "left"
    ids = answer_token_ids(tokenizer)
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = [prompt_text(tokenizer, t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        two = torch.softmax(logits[:, ids].float(), dim=-1)
        probs.extend(two[:, 1].tolist())
    return probs
