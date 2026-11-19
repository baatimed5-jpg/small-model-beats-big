"""Build the final scoreboard (reports/RESULTS.md + reports/scoreboard.png)
from whatever results_*.json files exist.  Run:  python src/compare.py"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REPORTS

NAMES = {"base": "Small, zero-shot", "lora": "Small + LoRA (ours)", "api": "Big model (API)"}

rows = [json.loads((REPORTS / f"results_{k}.json").read_text())
        for k in NAMES if (REPORTS / f"results_{k}.json").exists()]
if not rows:
    raise SystemExit("No results yet - run src/evaluate.py first.")

lines = ["# Scoreboard - offensive chat detection (860 human-labelled test messages)\n",
         "| Contender | Model | Macro F1 | Accuracy | ms / message | USD / 1M messages |",
         "|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {NAMES[r['contender']]} | {r['model']} | **{r['macro_f1']:.3f}** | "
                 f"{r['accuracy']:.1%} | {r['ms_per_message']} | ${r['usd_per_1M_messages']:,} |")
by = {r["contender"]: r for r in rows}
if "lora" in by and "api" in by:
    l, a = by["lora"], by["api"]
    cheaper = a["usd_per_1M_messages"] / max(l["usd_per_1M_messages"], 0.01)
    lines += ["", f"- F1 gap vs big model: **{l['macro_f1'] - a['macro_f1']:+.3f}**",
              f"- Cost: our model is **~{cheaper:,.0f}x cheaper** per message "
              "(local electricity vs API tokens)",
              f"- Size: 0.5B parameters vs {a['model']}"]
(REPORTS / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
labels = [NAMES[r["contender"]] for r in rows]
colors = ["#9aa5b1", "#2f7ed8", "#e8743b"][:len(rows)]
ax[0].bar(labels, [r["macro_f1"] for r in rows], color=colors)
ax[0].set_title("Quality (macro F1, higher is better)")
ax[0].set_ylim(0, 1)
ax[1].bar(labels, [r["usd_per_1M_messages"] for r in rows], color=colors)
ax[1].set_title("Cost per 1M messages (USD, lower is better)")
for a in ax:
    a.tick_params(axis="x", labelrotation=12)
plt.tight_layout()
plt.savefig(REPORTS / "scoreboard.png", dpi=130)
print("Saved reports/RESULTS.md and reports/scoreboard.png")
