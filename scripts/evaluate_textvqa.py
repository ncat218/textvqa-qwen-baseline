import argparse
import json
from pathlib import Path

from textvqa_baseline.scoring import compare_runs, score_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--clean-run", type=Path)
    parser.add_argument("--degraded-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.run:
        _, report = score_run(args.run)
    elif args.clean_run and args.degraded_run and args.output_dir:
        report = compare_runs(args.clean_run, args.degraded_run, args.output_dir)
    else:
        parser.error("Supply --run OR --clean-run, --degraded-run and --output-dir")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
