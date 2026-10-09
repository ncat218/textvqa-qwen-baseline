import argparse
import json
from pathlib import Path

from textvqa_baseline.inference import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, output_dir=args.output_dir, offline=args.offline), indent=2))


if __name__ == "__main__":
    main()
