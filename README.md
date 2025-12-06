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

# If you get execution policy errors, use this to bypass for the session:
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force; .\.venv\Scripts\Activate.ps1

# Alternative: activate without bypass (if your system allows)
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt
pip install -e .

# 3) Ensure Ollama is installed and running
# Download from: https://ollama.ai/
# Verify installation (may need to restart terminal or add to PATH):
ollama --version

# If ollama is not in PATH, use full path:
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version

# Pull the Mistral model:
ollama pull mistral
# Or with full path:
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull mistral

# 4) Run on Arrow directory
# Example: Run on HotpotQA (whole set)
mistral-bench --data data\hotpotqa --out runs\mistral_longbench --n 0 --k_ctx 3 --num_predict 128

# Example: Run on LCC dataset (488 samples for comparison)
mistral-bench --data data\lcc --out runs\lcc_mistral --n 488 --k_ctx 0 --num_predict 128

# If mistral-bench command is not recognized (venv not activated), use:
.\.venv\Scripts\python.exe -m mistral_bench.cli --data data\lcc --out runs\lcc_mistral --n 488 --k_ctx 0 --num_predict 128
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

## Configuration Parameters
- `--data`: Path to dataset (Arrow directory, .arrow file, or .json/.jsonl file)
- `--out`: Output directory for results
- `--n`: Number of samples to process (0 = all samples)
- `--k_ctx`: Number of context blocks to use (0 = use all available contexts)
- `--num_predict`: Maximum tokens to generate per prediction
- `--model`: Ollama model name (default: mistral)
- `--url`: Ollama API URL (default: http://localhost:11434)
- `--log-every`: Print progress every K items (default: 10)
- `--verbose` / `--quiet`: Control logging level

## Troubleshooting

### PowerShell Execution Policy Error
If you get `cannot be loaded. A certificate was explicitly revoked` error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force; .\.venv\Scripts\Activate.ps1
```

### Ollama Not Found
After installing Ollama, you may need to:
1. Restart your terminal/PowerShell
2. Or use the full path: `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"`

### Command Not Recognized
If `mistral-bench` is not found:
- Make sure virtual environment is activated (you should see `(.venv)` in prompt)
- Or run directly: `.\.venv\Scripts\python.exe -m mistral_bench.cli [args]`

## Notes
- **Perplexity**: Ollama's public API doesn't expose token logprobs, so ppl-related fields are `null`. This keeps files comparable to EM‑LLM, without claiming ppl.
- `--n 0` = run entire dataset.
- Use the Streamlit UI if you prefer (optional).
- For comparison with EM-LLM results, use matching `--n` values (e.g., 488 for LCC dataset).

## Optional UI
```powershell
streamlit run app\streamlit_app.py
```