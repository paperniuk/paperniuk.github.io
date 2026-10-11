"""Pull the decode-vs-context series out of the ds4 benchmark runs.

Usage: python3 tools/extract_bench.py ../../dstar-results/results > data/depth.json
The runs are the same ones the blog post charts: one chat grown to its full
context, reasoning xhigh, MTP on for the Qwen runs.
"""
import json
import sys
from pathlib import Path

RUNS = [
    ("q2", "Qwen3.8-Flash-Next Q2_0, 35 GiB", "q2-depth-xhigh"),
    ("iq3", "Qwen3.8-Flash-Next IQ3_XXS, 44 GiB", "iq3-depth-xhigh-4"),
    ("ds", "DeepSeek V4 Flash Q2, 81 GiB, streamed from SSD", "ds-depth-xhigh"),
]


def series(path):
    rows = json.loads(path.read_text())["rows"]
    return [
        {
            "ctx": r["prompt_tokens"],
            "decode": round(r["decode_tok_s"], 1),
            "prefill": round(r["prefill_tok_s"]),
        }
        for r in rows
    ]


def main(results_dir):
    out = {
        key: {"label": label, "points": series(Path(results_dir) / f"{run}.json")}
        for key, label, run in RUNS
    }
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
