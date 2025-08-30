#!/usr/bin/env python
# compare_results.py
import json, re, argparse, os, itertools
import pandas as pd
import matplotlib.pyplot as plt

ARTICLES = {"a","an","the"}
PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~]")

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    return " ".join(w for w in s.split() if w not in ARTICLES).strip()

def exact_match(pred: str, golds):
    npred = normalize_answer(pred or "")
    for g in (golds or []):
        if npred == normalize_answer(g):
            return 1.0
    return 0.0

def f1_score(pred: str, golds):
    ptoks = normalize_answer(pred or "").split()
    best = 0.0
    for g in (golds or []):
        gtoks = normalize_answer(g).split()
        if not ptoks or not gtoks:
            continue
        common = sum(min(ptoks.count(t), gtoks.count(t)) for t in set(ptoks))
        if common == 0: 
            f1 = 0.0
        else:
            prec = common/len(ptoks); rec = common/len(gtoks)
            f1 = 2*prec*rec/(prec+rec)
        best = max(best, f1)
    return best

def load_items(dirpath: str):
    """Return DataFrame with id, answers, pred; tries hotpotqa.jsonl first, then predictions.json."""
    jl = os.path.join(dirpath, "hotpotqa.jsonl")
    js = os.path.join(dirpath, "hotpotqa.json")
    pr = os.path.join(dirpath, "predictions.json")
    rows = []
    if os.path.exists(jl):
        with open(jl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        # hotpotqa.jsonl: has id, pred, answers
        df = pd.DataFrame(rows)
        # best effort: populate question if available in predictions.json
        if os.path.exists(pr):
            preds = pd.read_json(pr)
            df = df.merge(preds[["id","question"]], on="id", how="left")
        return df.rename(columns={"pred":"pred_text"})
    elif os.path.exists(js):
        rows = json.load(open(js, "r", encoding="utf-8"))
        df = pd.DataFrame(rows).rename(columns={"pred":"pred_text"})
        if os.path.exists(pr):
            preds = pd.read_json(pr)
            df = df.merge(preds[["id","question"]], on="id", how="left")
        return df
    elif os.path.exists(pr):
        rows = json.load(open(pr, "r", encoding="utf-8"))
        df = pd.DataFrame(rows)
        return df.rename(columns={"pred":"pred_text"})
    else:
        raise FileNotFoundError(f"No hotpotqa.jsonl/json or predictions.json in {dirpath}")

def main():
    ap = argparse.ArgumentParser(description="Compare EM/F1 between EM-LLM outputs and Mistral-bench outputs.")
    ap.add_argument("--em_dir", required=True, help="Folder with EM-LLM outputs (hotpotqa.jsonl/json or predictions.json)")
    ap.add_argument("--ours_dir", required=True, help="Folder with our Mistral outputs")
    ap.add_argument("--out", default="compare_out", help="Folder to write plots and CSVs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    em = load_items(args.em_dir)
    ours = load_items(args.ours_dir)

    # minimal columns
    keep = ["id","answers","pred_text","question"]
    for df in (em, ours):
        if "answers" not in df.columns:
            # last resort: try "answers" inside predictions.json shape
            pass
        if "question" not in df.columns:
            df["question"] = None

    em = em[ [c for c in keep if c in em.columns] ].copy()
    ours = ours[ [c for c in keep if c in ours.columns] ].copy()
    em = em.rename(columns={"pred_text":"pred_em"})
    ours = ours.rename(columns={"pred_text":"pred_mistral"})

    df = em.merge(ours, on=["id","answers"], how="inner", suffixes=("_em","_mistral"))
    # compute EM/F1 with the same code on both sides
    df["EM_em"] = [exact_match(p, a) for p,a in zip(df["pred_em"], df["answers"])]
    df["F1_em"] = [f1_score(p, a) for p,a in zip(df["pred_em"], df["answers"])]
    df["EM_mistral"] = [exact_match(p, a) for p,a in zip(df["pred_mistral"], df["answers"])]
    df["F1_mistral"] = [f1_score(p, a) for p,a in zip(df["pred_mistral"], df["answers"])]

    df["ΔEM"] = df["EM_mistral"] - df["EM_em"]
    df["ΔF1"] = df["F1_mistral"] - df["F1_em"]

    # save per-item diff table
    out_csv = os.path.join(args.out, "comparison_per_item.csv")
    df_out = df[["id","question","answers","pred_em","pred_mistral","EM_em","EM_mistral","F1_em","F1_mistral","ΔEM","ΔF1"]]
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Wrote {out_csv}")

    # print aggregates
    agg = {
        "n_overlap": len(df),
        "EM_em_mean": float(df["EM_em"].mean()),
        "EM_mistral_mean": float(df["EM_mistral"].mean()),
        "F1_em_mean": float(df["F1_em"].mean()),
        "F1_mistral_mean": float(df["F1_mistral"].mean()),
        "ΔEM_mean": float(df["ΔEM"].mean()),
        "ΔF1_mean": float(df["ΔF1"].mean()),
    }
    print(json.dumps(agg, indent=2))

    # plots
    plt.figure()
    plt.bar(["EM_em","EM_mistral"], [agg["EM_em_mean"], agg["EM_mistral_mean"]])
    plt.ylim(0,1); plt.ylabel("mean"); plt.title("EM mean (EM-LLM vs Mistral)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"em_mean_bar.png")); plt.close()

    plt.figure()
    plt.bar(["F1_em","F1_mistral"], [agg["F1_em_mean"], agg["F1_mistral_mean"]])
    plt.ylim(0,1); plt.ylabel("mean"); plt.title("F1 mean (EM-LLM vs Mistral)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"f1_mean_bar.png")); plt.close()

    plt.figure()
    plt.hist(df["ΔF1"], bins=21)
    plt.title("ΔF1 distribution (Mistral - EM-LLM)")
    plt.xlabel("ΔF1"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(os.path.join(args.out,"delta_f1_hist.png")); plt.close()

    # simple bootstrap CI on ΔF1 mean
    import numpy as np
    rng = np.random.default_rng(0)
    diffs = df["ΔF1"].to_numpy()
    boots = []
    for _ in range(2000):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(sample.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    with open(os.path.join(args.out,"delta_f1_bootstrap.txt"), "w", encoding="utf-8") as f:
        f.write(f"ΔF1 mean={agg['ΔF1_mean']:.4f}, 95% CI=({lo:.4f},{hi:.4f}), n={len(diffs)}\n")
    print(f"Wrote plots and CI to {args.out}")

if __name__ == "__main__":
    main()
