import os
import platform
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import (atomic_write, load_config, sha256_file, sha256_json, validate_manifest,
                       write_csv, write_json, write_jsonl)
from .model import QwenBaseline
from .prompting import PROMPT


PREDICTION_COLUMNS = ["image_id", "question_id", "question", "image_path", "condition_name",
                      "model_id", "model_revision", "prompt_template_id", "raw_prediction",
                      "normalized_prediction", "generation_seconds", "preprocess_seconds",
                      "total_seconds", "status", "error_message"]


def run(config_path: Path, *, limit: int | None = None, output_dir: Path | None = None,
        model: QwenBaseline | None = None, offline: bool = False) -> dict:
    config = load_config(config_path)
    frame, manifest_meta = validate_manifest(config, limit)
    target = Path(output_dir or config["output_dir"])
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    raw_manifest = Path(config["manifest_path"]).read_bytes()
    atomic_write(target / "manifest_used.csv", raw_manifest if limit is None else frame.drop(columns=["_resolved_clean", "_resolved_degraded"]).to_csv(index=False).encode())
    if limit is not None:
        write_csv(target / "smoke_manifest.csv", frame.drop(columns=["_resolved_clean", "_resolved_degraded"]))
    import torch
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    engine = model or QwenBaseline(config, offline=offline)
    records = []
    for row in frame.to_dict("records"):
        start = time.perf_counter()
        image_path = row["_resolved_clean"] if config["image_column"] == "clean_image_path" else row["_resolved_degraded"]
        prediction = dict(image_id=row["image_id"], question_id=row["question_id"], question=row["question"],
                          image_path=image_path, condition_name=config["experiment_name"], model_id=config["model_id"],
                          model_revision=engine.model_revision, prompt_template_id=config["prompt_template_id"],
                          raw_prediction="", normalized_prediction="", generation_seconds=0.0,
                          preprocess_seconds=0.0, total_seconds=0.0, status="failed", error_message="")
        try:
            raw, preprocessing, generation = engine.predict(row["question"], image_path)
            prediction.update(raw_prediction=raw, normalized_prediction=re.sub(r"\s+", " ", raw).strip(),
                              generation_seconds=generation, preprocess_seconds=preprocessing, status="ok")
        except Exception as exc:
            prediction["error_message"] = f"{type(exc).__name__}: {exc}"
        prediction["total_seconds"] = time.perf_counter() - start
        records.append(prediction)
    prediction_frame = pd.DataFrame(records, columns=PREDICTION_COLUMNS)
    write_csv(target / "predictions.csv", prediction_frame)
    write_jsonl(target / "predictions.jsonl", records)
    write_jsonl(target / "errors.jsonl", [r for r in records if r["status"] == "failed"])
    write_csv(target / "runtime.csv", prediction_frame[["image_id", "question_id", "preprocess_seconds", "generation_seconds", "total_seconds", "status"]])
    successful = prediction_frame[prediction_frame.status == "ok"]
    latency = successful.total_seconds
    statistics = dict(mean=float(latency.mean()) if len(latency) else None,
                      median=float(latency.median()) if len(latency) else None,
                      p95=float(latency.quantile(.95)) if len(latency) else None)
    metadata = dict(**manifest_meta, config_sha256=sha256_file(config_path), resolved_config_sha256=sha256_json(config), config=config,
                    model_id=config["model_id"], model_revision=engine.model_revision,
                    prompt_template_id=config["prompt_template_id"],
                    prompt_sha256=sha256_json(PROMPT),
                    generation_settings=dict(do_sample=False, num_beams=1, max_new_tokens=config["max_new_tokens"]),
                    processor_max_pixels=config["max_pixels"], image_column=config["image_column"],
                    runtime=engine.metadata(), hostname=platform.node(), timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    offline=offline, environment=dict(HF_HOME=os.environ.get("HF_HOME"), TRANSFORMERS_CACHE=os.environ.get("TRANSFORMERS_CACHE")))
    write_json(target / "run_metadata.json", metadata)
    summary = dict(row_count=len(frame), successful_rows=len(successful), failed_rows=len(frame)-len(successful), latency_seconds=statistics)
    write_json(target / "summary.json", summary)
    if summary["failed_rows"]:
        raise RuntimeError(f"{summary['failed_rows']} rows failed; see {target / 'errors.jsonl'}")
    return summary
