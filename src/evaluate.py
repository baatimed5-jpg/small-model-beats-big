"""Evaluate one contender on the SAME 860 human-labelled test messages.

  python src/evaluate.py --model base     # small model, no training (zero-shot)
  python src/evaluate.py --model lora     # small model + our LoRA fine-tune
  python src/evaluate.py --model api      # big model via any OpenAI-compatible API

For --model api set these environment variables (defaults = Groq, free tier):
  BIG_API_KEY        your key
  BIG_BASE_URL       https://api.groq.com/openai/v1
  BIG_MODEL          openai/gpt-oss-120b
  BIG_PRICE_IN       USD per 1M input tokens   (default 0.15)
  BIG_PRICE_OUT      USD per 1M output tokens  (default 0.60)
The key can also be put in a .env file next to README.md (BIG_API_KEY=...).
"""
import argparse
import json
import os
import time

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from common import ADAPTER_DIR, BASE_MODEL, REPORTS, load_data, messages

GPU_WATTS = 60          # laptop RTX 3050 under load (approx.)
KWH_PRICE = 0.15        # USD per kWh (approx.)


def run_local(name, texts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common import score_local

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if name == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_DIR).merge_and_unload()
    model.eval()
    score_local(model, tok, texts[:8])                      # warm-up
    t = time.perf_counter()
    probs = score_local(model, tok, texts)
    secs = time.perf_counter() - t
    usd = secs / 3600 * GPU_WATTS / 1000 * KWH_PRICE        # electricity only
    return probs, secs, usd, f"{BASE_MODEL}" + (" + LoRA" if name == "lora" else " (zero-shot)")


def load_env():
    env = ADAPTER_DIR.parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def run_api(texts, limit_rpm):
    from openai import OpenAI

    load_env()

    client = OpenAI(api_key=os.environ["BIG_API_KEY"],
                    base_url=os.environ.get("BIG_BASE_URL", "https://api.groq.com/openai/v1"))
    model = os.environ.get("BIG_MODEL", "openai/gpt-oss-120b")
    p_in = float(os.environ.get("BIG_PRICE_IN", 0.15))
    p_out = float(os.environ.get("BIG_PRICE_OUT", 0.60))
    probs, tok_in, tok_out, busy = [], 0, 0, 0.0
    for i, text in enumerate(texts):
        for attempt in range(6):
            try:
                t = time.perf_counter()
                r = client.chat.completions.create(model=model, messages=messages(text),
                                                   temperature=0, max_tokens=300,
                                                   # gpt-oss is a reasoning model: think briefly, hide the thinking
                                                   extra_body={"reasoning_effort": "low",
                                                               "include_reasoning": False})
                busy += time.perf_counter() - t
                break
            except Exception as e:                       # rate limit / network: wait & retry
                wait = 10 * (attempt + 1)
                print(f"  retry in {wait}s ({type(e).__name__})")
                time.sleep(wait)
        else:
            raise RuntimeError("API keeps failing")
        answer = (r.choices[0].message.content or "").strip().lower()
        probs.append(1.0 if answer.startswith("off") else 0.0)
        tok_in += r.usage.prompt_tokens
        tok_out += r.usage.completion_tokens
        if limit_rpm:
            time.sleep(60 / limit_rpm)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(texts)}")
    usd = tok_in / 1e6 * p_in + tok_out / 1e6 * p_out
    return probs, busy, usd, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "lora", "api"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N test messages")
    ap.add_argument("--rpm", type=int, default=28, help="API requests per minute cap")
    args = ap.parse_args()

    test = load_data()["test"]
    if args.limit:
        test = test.select(range(args.limit))
    texts, y = list(test["text"]), list(test["label"])

    if args.model == "api":
        probs, secs, usd, label = run_api(texts, args.rpm)
    else:
        probs, secs, usd, label = run_local(args.model, texts)
    pred = [int(p >= 0.5) for p in probs]

    res = {
        "contender": args.model,
        "model": label,
        "n": len(y),
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "precision_offensive": round(precision_score(y, pred, zero_division=0), 4),
        "recall_offensive": round(recall_score(y, pred, zero_division=0), 4),
        "ms_per_message": round(secs / len(y) * 1000, 2),
        "usd_per_1M_messages": round(usd / len(y) * 1e6, 2),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"results_{args.model}.json").write_text(json.dumps(res, indent=2))
    with open(REPORTS / f"predictions_{args.model}.jsonl", "w", encoding="utf-8") as f:
        for t, yy, p in zip(texts, y, probs):
            f.write(json.dumps({"text": t, "label": yy, "p_offensive": round(p, 4)}) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
