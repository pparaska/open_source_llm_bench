import argparse
from .eval import evaluate

def build_parser():
    ap = argparse.ArgumentParser(description="Mistral (Ollama) evaluator — EM‑LLM compatible outputs; supports JSON/JSONL and HF Arrow directories.")
    ap.add_argument("--data", required=True, help="JSON/JSONL file, .arrow file, or directory containing Arrow dataset")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--url", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default="mistral", help="Model name (ollama tag)")
    ap.add_argument("--n", type=int, default=50, help="Sample first N (0 = all)")
    ap.add_argument("--k_ctx", type=int, default=3, help="How many contexts per question")
    ap.add_argument("--num_predict", type=int, default=128, help="Max tokens to generate")
    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    evaluate(args.data, args.out, base_url=args.url, model=args.model, n=args.n, k_ctx=args.k_ctx, num_predict=args.num_predict)