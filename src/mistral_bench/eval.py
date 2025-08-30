# src/mistral_bench/eval.py
import os, re, time, json, logging
from typing import Any, Dict, List, Optional
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional HF datasets for Arrow loading (LongBench-style)
try:
    from datasets import Dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

logger = logging.getLogger("mistral_bench")

ARTICLES = {"a", "an", "the"}
PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~]")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    return " ".join(w for w in s.split() if w not in ARTICLES).strip()


def exact_match(pred: str, golds: List[str]) -> float:
    npred = normalize_answer(pred)
    return 1.0 if any(npred == normalize_answer(g) for g in golds) else 0.0


def f1_score(pred: str, golds: List[str]) -> float:
    ptoks = normalize_answer(pred).split()
    if not ptoks:
        return 0.0
    best = 0.0
    for g in golds:
        gtoks = normalize_answer(g).split()
        if not gtoks:
            continue
        num_same = sum(min(ptoks.count(t), gtoks.count(t)) for t in set(ptoks))
        if num_same == 0:
            f1 = 0.0
        else:
            prec = num_same / len(ptoks)
            rec = num_same / len(gtoks)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        best = max(best, f1)
    return best


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    text = open(path, "r", encoding="utf-8").read().strip()
    rows, bad = [], False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad = True
            rows = []
            break
    if not bad and rows:
        return rows
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def load_longbench_arrow(data_path: str) -> List[Dict[str, Any]]:
    if not _HAS_DATASETS:
        raise RuntimeError("Please install 'datasets' and 'pyarrow' to read Arrow files.")
    # Accept a directory with an Arrow file, or a direct .arrow file
    if os.path.isdir(data_path):
        files = [f for f in os.listdir(data_path) if f.endswith(".arrow")]
        if not files:
            raise FileNotFoundError("No .arrow file found in the directory.")
        arrow_file = os.path.join(data_path, files[0])
    else:
        arrow_file = data_path
    ds: Dataset = Dataset.from_file(arrow_file)
    # Expected: input (question), context (string), answers (list[str]), _id
    out = []
    for i in range(len(ds)):
        row = ds[i]
        q = row.get("input", "")
        ctx = row.get("context", "")
        ctxs = [p.strip() for p in ctx.split("\n\n") if p.strip()] or ([ctx] if ctx else [])
        gts = row.get("answers", [])
        rid = row.get("_id", f"q{i+1}")
        out.append({"id": rid, "question": q, "answers": gts, "contexts": ctxs})
    return out


def load_any(data_path: str) -> List[Dict[str, Any]]:
    # Directory or .arrow => Arrow loader; else JSON/JSONL
    if os.path.isdir(data_path) or data_path.endswith(".arrow"):
        logger.info("Loading Arrow dataset from %s", data_path)
        return load_longbench_arrow(data_path)
    logger.info("Loading JSON/JSONL dataset from %s", data_path)
    return load_json_or_jsonl(data_path)


def ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]],
                options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    logger.debug("POST %s (model=%s)", url, model)
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def ask_multihop(base_url: str, model: str, question: str, contexts: List[str],
                 k_ctx: int = 3, num_predict: int = 128) -> str:
    ctx = "\n\n".join(contexts[:k_ctx])
    sys_prompt = (
        "You answer multi-hop questions using only the provided context. "
        "Cite minimal spans if needed. Answer with a single short sentence."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"},
    ]
    ans = ollama_chat(
        base_url, model, messages, options={"temperature": 0.0, "num_predict": num_predict}
    ).get("message", {}).get("content", "").strip()
    return ans.splitlines()[0].strip()


def _write_plots(df: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    # Aggregate bar
    plt.figure()
    plt.bar(["EM_mean", "F1_mean"], [float(df["EM"].mean()), float(df["F1"].mean())])
    plt.title("Aggregate QA metrics")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "agg_metrics_bar.png"))
    plt.close()
    # F1 histogram
    plt.figure()
    df["F1"].hist(bins=15)
    plt.title("F1 distribution")
    plt.xlabel("F1")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "f1_hist.png"))
    plt.close()
    # EM histogram
    plt.figure()
    df["EM"].hist(bins=3)
    plt.title("EM distribution")
    plt.xlabel("EM")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "em_hist.png"))
    plt.close()
    # Generation time
    if "generation_time" in df.columns:
        plt.figure()
        df["generation_time"].hist(bins=15)
        plt.title("Generation time (s)")
        plt.xlabel("seconds")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "time_hist.png"))
        plt.close()


def evaluate(
    data_path: str,
    out_dir: str,
    base_url: str = "http://localhost:11434",
    model: str = "mistral",
    n: int = 50,
    k_ctx: int = 3,
    num_predict: int = 128,
    log_every: int = 10,
) -> Dict[str, str]:
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    rows = load_any(data_path)
    total = len(rows)
    cap = total if not n or n <= 0 else min(total, n)

    logger.info(
        "Starting evaluation: total=%d, cap=%d, model=%s, k_ctx=%d, num_predict=%d",
        total, cap, model, k_ctx, num_predict
    )

    preds: List[Dict[str, Any]] = []
    for i, r in enumerate(rows[:cap], 1):
        q = r.get("question", "") or r.get("input", "")
        golds = r.get("answers", []) or r.get("answer", [])
        if isinstance(golds, str):
            golds = [golds]
        ctxs = r.get("contexts", r.get("documents", []))
        if not ctxs:
            ctx = r.get("context", "")
            ctxs = [p.strip() for p in ctx.split("\n\n") if p.strip()] or ([ctx] if ctx else [])

        t0 = time.time()
        pred = ask_multihop(base_url, model, q, ctxs, k_ctx=k_ctx, num_predict=num_predict)
        t1 = time.time()

        rec = {
            "id": r.get("id", r.get("_id", f"q{i}")),
            "question": q,
            "answers": golds,
            "contexts": ctxs[:k_ctx],
            "pred": pred,
            # ppl-related fields are None because Ollama doesn't expose logprobs
            "total_ppl": None,
            "block_sizes": None,
            "mean_block_size": None,
            "generation_time": round(t1 - t0, 3),
        }
        preds.append(rec)

        if (i % max(1, log_every) == 0) or (i == cap):
            elapsed = time.time() - t_start
            avg = elapsed / i
            eta = avg * (cap - i)
            logger.info(
                "Progress %d/%d | last_id=%s | last_time=%.3fs | avg=%.2fs | ETA≈%.1fs",
                i, cap, rec["id"], rec["generation_time"], avg, eta
            )

    # Save detailed predictions
    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    # Metrics
    df = pd.DataFrame(preds)
    df["EM"] = [exact_match(p or "", a or []) for p, a in zip(df["pred"], df["answers"])]
    df["F1"] = [f1_score(p or "", a or []) for p, a in zip(df["pred"], df["answers"])]

    # EM-LLM style res.json (top-level "hotpotqa")
    res = {
        "hotpotqa": {
            "score": round(float(df["F1"].mean()) * 100.0, 2),
            "len_predictions": int(len(df)),
            "ppl_mean": None,
            "ppl_std": None,
            "block_sizes_mean": None,
            "block_sizes_std": None,
        }
    }
    with open(os.path.join(out_dir, "res.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # CSV table
    df[["id", "question", "answers", "pred", "EM", "F1", "generation_time"]].to_csv(
        os.path.join(out_dir, "results.csv"), index=False
    )

    # EM-LLM compatible per-item outputs
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

    with open(os.path.join(out_dir, "hotpotqa.jsonl"), "w", encoding="utf-8") as f:
        for o in emllm_items:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "hotpotqa.json"), "w", encoding="utf-8") as f:
        json.dump(emllm_items, f, ensure_ascii=False, indent=2)

    # Plots
    _write_plots(df, os.path.join(out_dir, "plots"))

    elapsed_total = time.time() - t_start
    logger.info("Done in %.1fs — wrote outputs to: %s", elapsed_total, out_dir)

    return {
        "predictions_path": os.path.join(out_dir, "predictions.json"),
        "res_path": os.path.join(out_dir, "res.json"),
        "hotpotqa_jsonl": os.path.join(out_dir, "hotpotqa.jsonl"),
        "hotpotqa_json": os.path.join(out_dir, "hotpotqa.json"),
        "csv_path": os.path.join(out_dir, "results.csv"),
        "plots_dir": os.path.join(out_dir, "plots"),
    }
