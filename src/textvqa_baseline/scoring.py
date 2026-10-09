import json
from pathlib import Path

import pandas as pd

from .evaluator import score_row
from .io_utils import sha256_file, write_csv, write_json


def score_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    manifest = pd.read_csv(run_dir / "manifest_used.csv", dtype=str, keep_default_na=False)
    predictions = pd.read_csv(run_dir / "predictions.csv", dtype={"image_id": str, "question_id": str}, keep_default_na=False)
    if len(manifest) != len(predictions) or len(predictions) != meta["row_count"]:
        raise ValueError("Prediction count differs from manifest")
    if predictions.status.ne("ok").any():
        raise ValueError("Failed predictions present; no score will be reported")
    if predictions.question_id.tolist() != manifest.question_id.tolist() or predictions.image_id.tolist() != manifest.image_id.tolist():
        raise ValueError("Prediction order/IDs differ from manifest")
    if len(manifest) == 1000 and sha256_file(run_dir / "manifest_used.csv") != meta["manifest_sha256"]:
        raise ValueError("Manifest bytes changed after inference")
    scores = [score_row(pred, ref) for pred, ref in zip(predictions.to_dict("records"), manifest.to_dict("records"))]
    predictions["score"] = scores
    write_csv(run_dir / "scored_predictions.csv", predictions)
    latency = predictions.total_seconds.astype(float)
    report = dict(overall_soft_accuracy=float(pd.Series(scores).mean()), successful_rows=len(scores), failed_rows=0,
                  latency_seconds=dict(mean=float(latency.mean()), median=float(latency.median()), p95=float(latency.quantile(.95))),
                  manifest_sha256=meta["manifest_sha256"], config_sha256=meta["config_sha256"],
                  model_id=meta["model_id"], model_revision=meta["model_revision"],
                  prompt_template_id=meta["prompt_template_id"], generation_settings=meta["generation_settings"],
                  processor_max_pixels=meta["processor_max_pixels"], scoring_formula="min(matching_answers/3,1)")
    write_json(run_dir / "score_report.json", report)
    return predictions, report


def compare_runs(clean_dir: Path, degraded_dir: Path, output_dir: Path) -> dict:
    clean_dir, degraded_dir, output_dir = map(Path, (clean_dir, degraded_dir, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    cm = json.loads((clean_dir / "run_metadata.json").read_text())
    dm = json.loads((degraded_dir / "run_metadata.json").read_text())
    for key in ("manifest_sha256", "model_id", "model_revision", "prompt_template_id", "generation_settings", "processor_max_pixels"):
        if cm[key] != dm[key]:
            raise ValueError(f"Runs differ in {key}")
    if cm.get("prompt_sha256") != dm.get("prompt_sha256"):
        raise ValueError("Runs differ in prompt text")
    if cm["image_column"] != "clean_image_path" or dm["image_column"] != "degraded_image_path":
        raise ValueError("Expected clean and degraded image columns")
    differences = {"experiment_name", "image_column", "expected_condition", "expected_level", "output_dir"}
    if {k: v for k, v in cm["config"].items() if k not in differences} != {k: v for k, v in dm["config"].items() if k not in differences}:
        raise ValueError("Fairness-sensitive configs differ")
    if cm["runtime"]["gpu_name"] != dm["runtime"]["gpu_name"] or cm["runtime"]["gpu_memory_bytes"] != dm["runtime"]["gpu_memory_bytes"]:
        raise ValueError("Hardware differs")
    if cm["runtime"]["attention_implementation"] != dm["runtime"]["attention_implementation"] or cm["runtime"]["dtype"] != dm["runtime"]["dtype"]:
        raise ValueError("Attention implementation or dtype differs")
    for key in ("torch_version", "transformers_version", "qwen_vl_utils_version", "cuda_version"):
        if cm["runtime"].get(key) != dm["runtime"].get(key):
            raise ValueError(f"Software runtime differs in {key}")
    clean, cr = score_run(clean_dir)
    degraded, dr = score_run(degraded_dir)
    if clean.question_id.tolist() != degraded.question_id.tolist() or clean.image_id.tolist() != degraded.image_id.tolist():
        raise ValueError("Paired question/image IDs differ")
    paired = pd.DataFrame(dict(image_id=clean.image_id, question_id=clean.question_id,
                               clean_prediction=clean.normalized_prediction, degraded_prediction=degraded.normalized_prediction,
                               clean_score=clean.score, degraded_score=degraded.score,
                               clean_seconds=clean.total_seconds, degraded_seconds=degraded.total_seconds))
    paired["score_difference"] = paired.degraded_score - paired.clean_score
    write_csv(output_dir / "paired_scores.csv", paired)
    clean_accuracy, degraded_accuracy = cr["overall_soft_accuracy"], dr["overall_soft_accuracy"]
    report = dict(clean_accuracy=clean_accuracy, degraded_accuracy=degraded_accuracy,
                  difference_percentage_points=(degraded_accuracy-clean_accuracy)*100,
                  relative_performance_drop_percent=((clean_accuracy-degraded_accuracy)/clean_accuracy*100 if clean_accuracy else None),
                  clean_higher_count=int((paired.clean_score > paired.degraded_score).sum()),
                  equal_score_count=int((paired.clean_score == paired.degraded_score).sum()),
                  degraded_higher_count=int((paired.clean_score < paired.degraded_score).sum()),
                  question_count=len(paired), manifest_sha256=cm["manifest_sha256"])
    write_json(output_dir / "comparison_report.json", report)
    write_csv(output_dir / "latency_comparison.csv", pd.DataFrame([
        dict(condition="clean", **cr["latency_seconds"]), dict(condition="degraded", **dr["latency_seconds"])]))
    return report
