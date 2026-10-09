import argparse
import json
from pathlib import Path

from textvqa_baseline.io_utils import load_config, validate_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Smoke testing only")
    args = parser.parse_args()
    config = load_config(args.config)
    _, summary = validate_manifest(config, args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
