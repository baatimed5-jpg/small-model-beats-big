# Scoreboard - offensive chat detection (860 human-labelled test messages)

| Contender | Model | Macro F1 | Accuracy | ms / message | USD / 1M messages |
|---|---|---|---|---|---|
| Small, zero-shot | Qwen/Qwen2.5-0.5B-Instruct (zero-shot) | **0.568** | 57.0% | 33.9 | $0.08 |
| Small + LoRA (ours) | Qwen/Qwen2.5-0.5B-Instruct + LoRA | **0.802** | 85.7% | 41.94 | $0.1 |
| Big model (API) | openai/gpt-oss-120b | **0.784** | 83.3% | 472.85 | $47.7 |

- F1 gap vs big model: **+0.018**
- Cost: our model is **~477x cheaper** per message (local electricity vs API tokens)
- Size: 0.5B parameters vs openai/gpt-oss-120b
