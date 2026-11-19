"""Improvement 1: pick the decision threshold on the VALIDATION set (never on test),
then apply it once to the saved test predictions.

Run:  python src/tune_threshold.py [--adapter models/lora-adapter] [--tag lora]
Writes reports/results_<tag>_tuned.json  (original results files are untouched)
"""
import argparse
import json

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ADAPTER_DIR, BASE_MODEL, REPORTS, ROOT, load_data, score_local

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=str(ADAPTER_DIR))
ap.add_argument("--tag", default="lora")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

val = load_data()["validation"]
p_val = np.array(score_local(model, tok, list(val["text"])))
y_val = np.array(val["label"])

grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
f1s = [f1_score(y_val, p_val >= t, average="macro") for t in grid]
best_t = float(grid[int(np.argmax(f1s))])
print(f"validation: F1 @0.50 = {f1_score(y_val, p_val >= .5, average='macro'):.4f} | "
      f"best threshold {best_t} -> F1 {max(f1s):.4f}")

pred_file = REPORTS / f"predictions_{args.tag}.jsonl"
if not pred_file.exists():  # score test set if this adapter has no saved predictions yet
    test = load_data()["test"]
    p_test = np.array(score_local(model, tok, list(test["text"])))
    y = np.array(test["label"])
else:
    rows = [json.loads(l) for l in pred_file.read_text(encoding="utf-8").splitlines()]
    p_test = np.array([r["p_offensive"] for r in rows])
    y = np.array([r["label"] for r in rows])

pred = (p_test >= best_t).astype(int)
res = {
    "contender": f"{args.tag}_tuned", "model": f"{BASE_MODEL} + LoRA ({args.tag}), threshold {best_t}",
    "threshold": best_t, "n": int(len(y)),
    "accuracy": round(accuracy_score(y, pred), 4),
    "macro_f1": round(f1_score(y, pred, average="macro"), 4),
    "precision_offensive": round(precision_score(y, pred), 4),
    "recall_offensive": round(recall_score(y, pred), 4),
}
(REPORTS / f"results_{args.tag}_tuned.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
