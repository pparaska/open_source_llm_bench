# Mistral Bench (EM‑LLM compatible; Arrow/JSONL; Ollama/CPU)

This project evaluates **Mistral via Ollama** on HotpotQA/LongBench-style datasets and saves **EM‑LLM compatible outputs**:
- `res.json` (top-level `hotpotqa` metrics block)
- `hotpotqa.jsonl` and `hotpotqa.json` (per‑item predictions with EM‑LLM fields)
- Plus: `results.csv` and plots

Works with:
- **Arrow** folders or files from LongBench (e.g., `data-00000-of-00001.arrow`, `dataset_info.json`, `state.json`)
- **JSON/JSONL** files using HotpotQA-like fields

## Quick start (Windows PowerShell)

```powershell
# 1) Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install
pip install -r requirements.txt
python -m pip install -e .

# 3) Ensure Ollama & model
ollama pull mistral

# 4) Run on Arrow directory (whole set)
mistral-bench --data C:\path\to\hotpotqa\test --out runs\mistral_longbench --n 0 --k_ctx 3 --num_predict 128
```

You can also pass a single `.arrow` file, or a `.json`/`.jsonl` file. The tool auto-detects.

## Outputs
```
runs\mistral_longbench\
  res.json                # {"hotpotqa": {"score": ..., "len_predictions": ..., "ppl_mean": null, ...}}
  hotpotqa.jsonl          # JSONL per item (EM‑LLM-like keys; ppl fields null)
  hotpotqa.json           # Same as above but as a JSON array
  predictions.json        # Our full record (question, contexts, etc.)
  results.csv             # Tabular view (id, question, answers, pred, EM, F1, time)
  plots\
    agg_metrics_bar.png   # EM & F1 means
    f1_hist.png           # Distribution of F1
    em_hist.png           # Distribution of EM
    time_hist.png         # Generation time distribution
```

## Schema mapping
- **LongBench/Arrow**: `question ← input`, `contexts ← context` split on blank lines, `answers ← answers`, `id ← _id` (fallback index)
- **JSON/JSONL**: accepts either our schema (question/answers/contexts) or HotpotQA-like (`input`, `context`, `answers`)

## Notes
- **Perplexity**: Ollama’s public API doesn’t expose token logprobs, so ppl-related fields are `null`. This keeps files comparable to EM‑LLM, without claiming ppl.
- `--n 0` = run entire dataset.
- Use the Streamlit UI if you prefer (optional).

## Optional UI
```powershell
streamlit run app\streamlit_app.py
```