"""API + React demo server.  Run:  python api.py   ->  http://127.0.0.1:8002"""
import json
import sys
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from common import ADAPTER_DIR, BASE_MODEL, REPORTS, score_local  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
model = PeftModel.from_pretrained(model, ADAPTER_DIR).merge_and_unload().eval()
score_local(model, tok, ["warm up"])

app = FastAPI(title="Gaming Chat Moderator API")


class Msg(BaseModel):
    text: str


@app.post("/api/moderate")
def moderate(m: Msg):
    t = time.perf_counter()
    p = score_local(model, tok, [m.text])[0]
    return {"offensive": p >= 0.5, "p_offensive": round(p, 4),
            "ms": round((time.perf_counter() - t) * 1000, 1), "device": device}


@app.get("/api/scoreboard")
def scoreboard():
    return [json.loads((REPORTS / f"results_{k}.json").read_text())
            for k in ("base", "lora", "api") if (REPORTS / f"results_{k}.json").exists()]


app.mount("/", StaticFiles(directory=ROOT / "web" / "dist", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
