# Optional UI: app/streamlit_app.py
import json, os, re, time, requests
from typing import Any, Dict, List, Optional
import pandas as pd, numpy as np, matplotlib.pyplot as plt, streamlit as st

def ensure_datasets():
    try:
        import datasets  # noqa: F401
        import pyarrow   # noqa: F401
        return True
    except Exception as e:
        st.error("Please install 'datasets' and 'pyarrow' to read Arrow datasets. pip install datasets pyarrow")
        return False

ARTICLES = {"a","an","the"}
PUNCT_RE = re.compile(r"[!\\\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~]")
def normalize_answer(s: str) -> str:
    s = s.lower(); s = PUNCT_RE.sub(" ", s)
    return " ".join(w for w in s.split() if w not in ARTICLES).strip()
def exact_match(pred: str, golds: List[str]) -> float:
    npred = normalize_answer(pred); return 1.0 if any(npred == normalize_answer(g) for g in golds) else 0.0
def f1_score(pred: str, golds: List[str]) -> float:
    ptoks = normalize_answer(pred).split()
    if not ptoks: return 0.0
    best = 0.0
    for g in golds:
        gtoks = normalize_answer(g).split()
        if not gtoks: continue
        num_same = sum(min(ptoks.count(t), gtoks.count(t)) for t in set(ptoks))
        if num_same == 0: f1 = 0.0
        else:
            prec = num_same/len(ptoks); rec = num_same/len(gtoks)
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        best = max(best, f1)
    return best
def ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None):
    url = base_url.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    if options: payload["options"] = options
    r = requests.post(url, json=payload, timeout=600); r.raise_for_status()
    return r.json()
def ask_multihop(base_url: str, model: str, question: str, contexts: List[str], k_ctx: int = 3, num_predict: int = 128) -> str:
    ctx = "\\n\\n".join(contexts[:k_ctx])
    sys_prompt = ("You answer multi-hop questions using only the provided context. "
                  "Cite minimal spans if needed. Answer with a single short sentence.")
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Context:\\n{ctx}\\n\\nQuestion: {question}\\n\\nAnswer:"}]
    resp = ollama_chat(base_url, model, messages, options={"temperature": 0.0, "num_predict": num_predict})
    ans = resp.get("message", {}).get("content", "").strip()
    return ans.splitlines()[0].strip()

st.set_page_config(page_title="Mistral Bench — EM‑LLM compatible", layout="wide")
st.title("Mistral Bench (Ollama/CPU) — EM‑LLM compatible outputs")

with st.sidebar:
    base_url = st.text_input("Ollama URL", value="http://localhost:11434")
    model = st.text_input("Model", value="mistral")
    k_ctx = st.number_input("Contexts per question", 1, 10, 3)
    n = st.number_input("Sample first N (0 = all)", 0, 1000000, 50)
    num_predict = st.number_input("num_predict", 16, 1024, 128)

tab1, tab2 = st.tabs(["Upload JSON/JSONL", "Use Arrow directory"])

def run_and_render(rows):
    preds=[]; prog = st.progress(0)
    cap = len(rows) if int(n)==0 else min(len(rows), int(n))
    for i, r in enumerate(rows[:cap], 1):
        q = r.get("question","") or r.get("input","")
        gts = r.get("answers", []) or r.get("answer", [])
        if isinstance(gts, str): gts=[gts]
        ctxs = r.get("contexts") or r.get("documents")
        if not ctxs:
            ctx = r.get("context",""); ctxs=[p.strip() for p in ctx.split("\\n\\n") if p.strip()] or ([ctx] if ctx else [])
        import time
        t0=time.time(); pred = ask_multihop(base_url, model, q, ctxs, k_ctx=int(k_ctx), num_predict=int(num_predict)); t1=time.time()
        preds.append({"id": r.get("id", r.get("_id", f"q{i}")), "question": q, "answers": gts, "contexts": ctxs,
                      "pred": pred, "total_ppl": None, "block_sizes": None, "mean_block_size": None, "generation_time": round(t1-t0,3)})
        prog.progress(i/cap)
    import pandas as pd
    df = pd.DataFrame(preds)
    df["EM"] = [exact_match(p or "", a or []) for p,a in zip(df["pred"], df["answers"])]
    df["F1"] = [f1_score(p or "", a or []) for p,a in zip(df["pred"], df["answers"])]
    st.metric("EM", f"{df['EM'].mean():.3f}")
    st.metric("F1", f"{df['F1'].mean():.3f}")
    # Plots
    fig1 = plt.figure(); plt.bar(["EM_mean","F1_mean"], [float(df["EM"].mean()), float(df["F1"].mean())]); plt.title("Aggregate QA metrics"); plt.ylabel("score"); plt.ylim(0,1); plt.tight_layout(); st.pyplot(fig1)
    fig2 = plt.figure(); df["F1"].hist(bins=15); plt.title("F1 distribution"); plt.xlabel("F1"); plt.ylabel("count"); plt.tight_layout(); st.pyplot(fig2)
    fig3 = plt.figure(); df["EM"].hist(bins=3); plt.title("EM distribution"); plt.xlabel("EM"); plt.ylabel("count"); plt.tight_layout(); st.pyplot(fig3)
    fig4 = plt.figure(); df["generation_time"].hist(bins=15); plt.title("Generation time (s)"); plt.xlabel("seconds"); plt.ylabel("count"); plt.tight_layout(); st.pyplot(fig4)
    st.dataframe(df[["id","question","answers","pred","EM","F1","generation_time"]])

    # Downloads (EM‑LLM compatible)
    emllm_items = [{
        "id": r["id"],
        "pred": r["pred"],
        "answers": r["answers"],
        "all_classes": None,
        "length": len(" ".join(r.get("contexts", []) or [])),
        "token_length": None,
        "chunk_ppl": None,
        "total_ppl": None,
        "block_sizes": None,
        "mean_block_size": None,
        "generation_time": r["generation_time"],
    } for r in preds]
    res = {"hotpotqa": {"score": round(float(df["F1"].mean())*100.0,2), "len_predictions": int(len(df)),
                        "ppl_mean": None, "ppl_std": None, "block_sizes_mean": None, "block_sizes_std": None}}
    st.download_button("hotpotqa.jsonl", data="\n".join(json.dumps(o, ensure_ascii=False) for o in emllm_items), file_name="hotpotqa.jsonl")
    st.download_button("hotpotqa.json", data=json.dumps(emllm_items, ensure_ascii=False, indent=2), file_name="hotpotqa.json")
    st.download_button("res.json", data=json.dumps(res, ensure_ascii=False, indent=2), file_name="res.json")
    st.download_button("results.csv", data=df.to_csv(index=False).encode("utf-8"), file_name="results.csv")
    st.download_button("predictions.json", data=json.dumps(preds, ensure_ascii=False, indent=2), file_name="predictions.json")

with tab1:
    st.write("Upload HotpotQA-style JSON/JSONL")
    qa_file = st.file_uploader("Dataset", type=["json","jsonl"])
    if st.button("Run (JSON/JSONL)"):
        if not qa_file:
            st.error("Please upload a dataset file.")
        else:
            rows = []
            text = qa_file.getvalue().decode("utf-8").strip()
            bad=False
            for line in text.splitlines():
                line=line.strip()
                if not line: continue
                try: rows.append(json.loads(line))
                except Exception: bad=True; rows=[]; break
            if bad or not rows:
                rows = json.loads(text)
                rows = rows if isinstance(rows, list) else [rows]
            # normalize fields
            for i, r in enumerate(rows, 1):
                r.setdefault("id", r.get("_id", f"q{i}"))
                r.setdefault("question", r.get("input", r.get("question","")))
                if "contexts" not in r:
                    ctx = r.get("context", "")
                    r["contexts"] = [p.strip() for p in ctx.split("\\n\\n") if p.strip()] or ([ctx] if ctx else [])
            run_and_render(rows)

with tab2:
    st.write("Enter a folder path that contains an Arrow dataset (e.g., LongBench hotpotqa/test)")
    data_dir = st.text_input("Arrow dataset directory path")
    if st.button("Run (Arrow)"):
        if not data_dir or not os.path.exists(data_dir):
            st.error("Please enter a valid directory path.")
        elif not ensure_datasets():
            pass
        else:
            from datasets import Dataset
            if os.path.isdir(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith(".arrow")]
                if not files:
                    st.error("No .arrow file found in the directory."); st.stop()
                arrow_file = os.path.join(data_dir, files[0])
            else:
                arrow_file = data_dir
            ds: Dataset = Dataset.from_file(arrow_file)
            rows = []
            for i in range(len(ds)):
                row = ds[i]
                q = row.get("input", "")
                ctx = row.get("context", "")
                ctxs = [p.strip() for p in ctx.split("\\n\\n") if p.strip()] or ([ctx] if ctx else [])
                gts = row.get("answers", [])
                rid = row.get("_id", f"q{i+1}")
                rows.append({"id": rid, "question": q, "answers": gts, "contexts": ctxs})
            run_and_render(rows)