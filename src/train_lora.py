"""Fine-tune Qwen2.5-0.5B-Instruct with LoRA to detect offensive chat messages.

LoRA = instead of changing all 500M weights, we freeze them and train small
"adapter" matrices (~1% of the size) plugged into each layer. Cheap, fast,
fits on a 4 GB laptop GPU, and the result is a ~35 MB file.

Run:  python src/train_lora.py            (about 30-40 min on an RTX 3050 4 GB)
      python src/train_lora.py --epochs 2 --limit 2000   (quick test)
"""
import argparse

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from common import ADAPTER_DIR, BASE_MODEL, LABELS, ROOT, load_data, prompt_text

MAX_LEN = 192


def build_example(tokenizer, text, label):
    """Prompt tokens get label -100 (ignored), so the model only learns the answer."""
    p = tokenizer(prompt_text(tokenizer, text), add_special_tokens=False)["input_ids"][-(MAX_LEN - 4):]
    a = tokenizer(LABELS[label] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}


def collate(tokenizer):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        ids = [b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--limit", type=int, default=0, help="use only N training rows")
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data = load_data()
    train = data["train"].shuffle(seed=42)
    if args.limit:
        train = train.select(range(args.limit))
    train = train.map(lambda r: build_example(tok, r["text"], r["label"]),
                      remove_columns=train.column_names)
    val = data["validation"].select(range(300)).map(
        lambda r: build_example(tok, r["text"], r["label"]),
        remove_columns=data["validation"].column_names)

    targs = TrainingArguments(
        output_dir=str(ROOT / "models" / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,          # effective batch = 16
        per_device_eval_batch_size=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        logging_steps=25,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
            data_collator=collate(tok)).train()

    model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print(f"LoRA adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
