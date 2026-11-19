"""Web demo of the fine-tuned moderator.  Run:  python app.py  (needs the trained adapter)"""
import sys
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import ADAPTER_DIR, BASE_MODEL, score_local  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
model = PeftModel.from_pretrained(model, ADAPTER_DIR).merge_and_unload().eval()


def moderate(text):
    p = score_local(model, tok, [text])[0]
    verdict = "OFFENSIVE - block / warn" if p >= 0.5 else "CLEAN - allow"
    return verdict, {"offensive": p, "clean": 1 - p}


demo = gr.Interface(
    fn=moderate,
    inputs=gr.Textbox(lines=3, label="Chat message"),
    outputs=[gr.Textbox(label="Verdict"), gr.Label(label="Confidence")],
    title="Gaming Chat Moderator - 0.5B model fine-tuned with LoRA",
    description="Runs fully on this laptop. No API, no cost per message.",
    examples=[["gg wp, that last round was insane"], ["uninstall the game you useless idiot"],
              ["anyone want to queue ranked later?"]],
)

if __name__ == "__main__":
    demo.launch()
