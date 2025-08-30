# src/mistral_bench/cli.py
import argparse, logging, sys
from .eval import evaluate

def main():
    ap = argparse.ArgumentParser(
        description="Mistral (Ollama) evaluator — EM-LLM compatible outputs; JSON/JSONL & HF Arrow."
    )
    ap.add_argument("--data", required=True, help="JSON/JSONL file, .arrow file, or directory containing Arrow dataset")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--url", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default="mistral", help="Model name (ollama tag)")
    ap.add_argument("--n", type=int, default=50, help="Sample first N (0 = all)")
    ap.add_argument("--k_ctx", type=int, default=3, help="How many contexts per question")
    ap.add_argument("--num_predict", type=int, default=128, help="Max tokens to generate")
    ap.add_argument("--log-every", type=int, default=10, dest="log_every", help="Print progress every K items")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--verbose", action="store_true", help="Enable INFO logs (default)")
    g.add_argument("--quiet", action="store_true", help="Only show warnings/errors")

    args = ap.parse_args()

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(stream=sys.stdout, level=level, format="%(asctime)s | %(levelname)s | %(message)s")

    logging.info(
        "Args: data=%s out=%s url=%s model=%s n=%d k_ctx=%d num_predict=%d log_every=%d",
        args.data, args.out, args.url, args.model, args.n, args.k_ctx, args.num_predict, args.log_every
    )

    evaluate(
        args.data, args.out,
        base_url=args.url, model=args.model,
        n=args.n, k_ctx=args.k_ctx, num_predict=args.num_predict,
        log_every=args.log_every
    )

if __name__ == "__main__":
    main()
