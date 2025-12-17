from dotenv import load_dotenv
load_dotenv()

import argparse
from eval_framework.config import load_settings
from eval_framework.dataset import load_jsonl
from eval_framework.llm_client import OllamaLLMClient
from eval_framework.runner import run_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scorer", default="llm_judge", choices=["exact", "llm_judge"])
    ap.add_argument("--out", default="outputs/runs")
    args = ap.parse_args()

    settings = load_settings()
    tasks = load_jsonl(args.dataset)
    llm = OllamaLLMClient(settings)

    out = run_eval(tasks, llm, args.scorer, args.out)
    print(f"Saved run to {out}")

if __name__ == "__main__":
    main()
