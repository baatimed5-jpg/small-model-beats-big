import { useEffect, useRef, useState } from "react";

const PLAYERS = [
  { name: "you", color: "#4ade80" },
  { name: "xX_Viper_Xx", color: "#f472b6" },
  { name: "n00bslayer", color: "#60a5fa" },
  { name: "Kira", color: "#facc15" },
];

const QUICK = [
  "gg wp, that last round was insane",
  "uninstall the game you useless idiot",
  "anyone want to queue ranked later?",
  "you're trash, report this noob",
  "nice clutch kira!!",
];

const NAMES = { base: "Qwen 0.5B · no training", lora: "Qwen 0.5B + LoRA (ours)", api: "GPT-OSS-120B (OpenAI)" };

async function moderate(text) {
  return fetch("/api/moderate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }).then((r) => r.json());
}

function Message({ m, onReveal }) {
  const p = PLAYERS.find((x) => x.name === m.author) || PLAYERS[0];
  return (
    <li className={`msg ${m.res?.offensive ? "flagged" : ""} ${m.res ? "" : "pending"}`}>
      <span className="author" style={{ color: p.color }}>{m.author}</span>
      {m.res?.offensive && !m.revealed ? (
        <button className="hidden-msg" onClick={onReveal}>
          ⛔ message hidden by moderator, click to reveal
        </button>
      ) : (
        <span className="body">{m.text}</span>
      )}
      <span className="badge">
        {!m.res ? "scanning…" : m.res.offensive ? `blocked · ${Math.round(m.res.p_offensive * 100)}%` : `clean · ${m.res.ms} ms`}
      </span>
    </li>
  );
}

export default function App() {
  const [msgs, setMsgs] = useState([]);
  const [text, setText] = useState("");
  const [author, setAuthor] = useState("you");
  const [board, setBoard] = useState([]);
  const [last, setLast] = useState(null);
  const listRef = useRef(null);

  useEffect(() => {
    fetch("/api/scoreboard").then((r) => r.json()).then(setBoard).catch(() => {});
  }, []);
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [msgs]);

  async function send(t = text, who = author) {
    if (!t.trim()) return;
    const id = Date.now() + Math.random();
    setMsgs((m) => [...m, { id, author: who, text: t }]);
    setText("");
    const res = await moderate(t);
    setLast(res);
    setMsgs((m) => m.map((x) => (x.id === id ? { ...x, res } : x)));
  }

  const flagged = msgs.filter((m) => m.res?.offensive).length;
  const maxCost = Math.max(...board.map((b) => b.usd_per_1M_messages), 1);

  return (
    <div className="page">
      <header>
        <div className="logo">
          <span className="mark">LG</span>
          <div>
            <h1>LOBBY GUARD</h1>
            <p>Live toxic-chat moderation · 0.5B model fine-tuned with LoRA</p>
          </div>
        </div>
        <div className="status">
          <span className="live" /> {last?.device === "cuda" ? "GPU · RTX 3050" : "local model"} · offline
        </div>
      </header>

      <main>
        <section className="chat panel">
          <div className="panel-head">
            <span># ranked-lobby-7</span>
            <span className="counts">{msgs.length} msgs · <b>{flagged}</b> blocked</span>
          </div>
          <ul className="list" ref={listRef}>
            {msgs.length === 0 && (
              <li className="welcome">
                Type a message as any player, or use a quick message below.<br />Every line is checked by the model before it is shown.
              </li>
            )}
            {msgs.map((m) => (
              <Message key={m.id} m={m} onReveal={() => setMsgs((all) => all.map((x) => (x.id === m.id ? { ...x, revealed: true } : x)))} />
            ))}
          </ul>
          <div className="quick">
            {QUICK.map((q, i) => (
              <button key={q} onClick={() => send(q, PLAYERS[(i % 3) + 1].name)}>{q}</button>
            ))}
          </div>
          <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
            <select value={author} onChange={(e) => setAuthor(e.target.value)} aria-label="Player">
              {PLAYERS.map((p) => <option key={p.name}>{p.name}</option>)}
            </select>
            <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Say something in chat…" />
            <button type="submit" disabled={!text.trim()}>SEND</button>
          </form>
        </section>

        <aside>
          <section className="panel">
            <div className="panel-head"><span>Last verdict</span></div>
            {last ? (
              <div className={`meter ${last.offensive ? "bad" : "good"}`}>
                <div className="meter-top">
                  <b>{last.offensive ? "TOXIC" : "CLEAN"}</b>
                  <span>{Math.round(last.p_offensive * 100)}% toxic</span>
                </div>
                <div className="track"><i style={{ width: `${last.p_offensive * 100}%` }} /></div>
                <small>decided in {last.ms} ms on this laptop</small>
              </div>
            ) : (
              <p className="dim">Send a message to see the score.</p>
            )}
          </section>

          <section className="panel">
            <div className="panel-head"><span>Small vs big · 860 test messages</span></div>
            <ul className="board">
              {board.map((b) => (
                <li key={b.contender} className={b.contender}>
                  <div className="b-name">{NAMES[b.contender]}{b.contender === "lora" && <em>WINNER</em>}</div>
                  <div className="b-row"><span>F1</span><div className="track"><i style={{ width: `${b.macro_f1 * 100}%` }} /></div><code>{b.macro_f1.toFixed(3)}</code></div>
                  <div className="b-row"><span>$/1M</span><div className="track cost"><i style={{ width: `${Math.max(2, (b.usd_per_1M_messages / maxCost) * 100)}%` }} /></div><code>${b.usd_per_1M_messages}</code></div>
                  <div className="b-row"><span>speed</span><div className="spacer" /><code>{Math.round(b.ms_per_message)} ms</code></div>
                </li>
              ))}
            </ul>
          </section>
        </aside>
      </main>
      <footer>Qwen2.5-0.5B-Instruct + LoRA (8.8M trainable params) · trained in 33 min on a 4 GB laptop GPU · PyTorch · PEFT · FastAPI · React</footer>
    </div>
  );
}
