import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from textvqa_baseline.inference import PREDICTION_COLUMNS, run
from textvqa_baseline.io_utils import load_config, validate_manifest
from textvqa_baseline.model import QwenBaseline
from textvqa_baseline.scoring import compare_runs, score_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-config", type=Path, default=Path("configs/base_clean.yaml"))
    parser.add_argument("--degraded-config", type=Path, default=Path("configs/base_realistic_mix_l2.yaml"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    clean_config = load_config(args.clean_config)
    degraded_config = load_config(args.degraded_config)
    clean_frame, _ = validate_manifest(clean_config, 16)
    degraded_frame, _ = validate_manifest(degraded_config, 16)
    if clean_frame.question_id.tolist() != degraded_frame.question_id.tolist():
        raise ValueError("Smoke question IDs differ")
    root = args.output_root or Path("outputs") / ("smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    model = QwenBaseline(clean_config, offline=True)
    clean = run(args.clean_config, limit=16, output_dir=root / "base_clean", model=model, offline=True)
    degraded = run(args.degraded_config, limit=16, output_dir=root / "base_realistic_mix_l2", model=model, offline=True)
    for name in ("base_clean", "base_realistic_mix_l2"):
        import pandas as pd
        predictions = pd.read_csv(root / name / "predictions.csv")
        if list(predictions.columns) != PREDICTION_COLUMNS or len(predictions) != 16:
            raise ValueError(f"Invalid smoke output schema: {name}")
        score_run(root / name)
    comparison = compare_runs(root / "base_clean", root / "base_realistic_mix_l2", root / "comparison")
    estimate_seconds = 1000 * (clean["latency_seconds"]["mean"] + degraded["latency_seconds"]["mean"])
    print(json.dumps({"smoke_root": str(root), "question_ids": clean_frame.question_id.tolist(),
                      "estimated_full_pair_seconds": estimate_seconds, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
