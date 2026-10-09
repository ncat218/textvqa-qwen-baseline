# TextVQA clean/degraded Qwen baseline

Command-line, inference-only paired baseline for the 1,000 questions in `main/final_main_manifest.csv` from [Data_suy_thoai (Text VQA)](https://www.kaggle.com/datasets/anhthu128/data-suy-thoai-text-vqa). The degraded condition is `realistic_mix/L2`. The model is exactly `Qwen/Qwen2.5-VL-3B-Instruct`; both runs use one image, the same question, the same prompt, and greedy decoding. No model training or image restoration occurs.

## Data layout and the clean-image prerequisite

The published degradation dataset contains `textvqa_realistic_v2/main/final_main_manifest.csv` and `textvqa_realistic_v2/main/images/realistic_mix/L2/*.png`. Its manifest also has `clean_image_path`, but those values describe the **original TextVQA image source at generation time**. The degradation dataset does not bundle a `main/images_clean` directory. Obtain the same original TextVQA clean images separately and point `TEXTVQA_CLEAN_ROOT` to their directory. Validation will reject missing or ambiguous clean images before loading the model.

The manifest's legacy `clean_path` and `blur_path` columns are never used. The baseline reads `clean_image_path` and `degraded_image_path` only. The path resolver maps an old `/.../main/images/realistic_mix/L2/name.png` to the mounted degradation dataset and locates a clean image by exact filename under `TEXTVQA_CLEAN_ROOT`. A duplicate filename in the clean source is an error. The manifest bytes are never edited for a full run.

Set these paths on the **Linux H200 server**:

```bash
export TEXTVQA_DATASET_ROOT=/data/data-suy-thoai-text-vqa
export TEXTVQA_CLEAN_ROOT=/data/original-textvqa-images
export HF_HOME=/data/hf-cache
```

The first variable must contain the `textvqa_realistic_v2` directory. The second must recursively contain the 1,000 matching original image filenames. Make sure the images really correspond to the IDs; filenames alone do not prove content identity.

Expected manifest columns: `image_id`, `question_id`, `question`, `answers_json` or `answers` (10 strings), `clean_image_path`, `degraded_image_path`, `condition`, `level`, `config_sha256`, `source_manifest_sha256`. Other columns are retained. A full manifest must have exactly 1,000 rows and unique `question_id`. Each row must be `realistic_mix/L2`.

## Setup

Use Linux, Python 3.11, CUDA and one H200. From this project directory:

```bash
bash run.sh setup
```

`setup` uses `uv` if present or a local `.venv` with pip. Direct dependencies are version-pinned in `pyproject.toml`; for a fully frozen transitive environment, generate and commit a lock file on the target Linux/Python platform before the experiment. FlashAttention 2 is optional; when importable and usable it is selected, otherwise SDPA is used. Record the selected implementation for both runs and compare it before interpreting results.

Download the model **explicitly** before running the baseline. The run commands use offline mode and cannot fetch it implicitly:

```bash
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", revision="main")'
```

For strict reproducibility, replace `model_revision: main` in **both** YAML files with the same immutable Hugging Face commit hash before download and inference. The comparison checks the resolved commit. The default `max_pixels` is 2,048 visual patches × 28 × 28 = 1,605,632 pixels, chosen to retain small scene text; both configs use the same value. The Qwen [model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [Transformers Qwen2.5-VL docs](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/qwen2_5_vl) describe the chat template, image preprocessing and generation pattern used here.

## Commands

```bash
bash run.sh validate       # both image columns, all 1,000 rows; no model load
bash run.sh smoke          # first 16 paired rows, one model load, scores and time estimate
bash run.sh base_clean     # 1,000 clean rows
bash run.sh base_degraded  # same 1,000 rows on realistic_mix L2
bash run.sh score_baseline # score both and compare paired questions
```

Run the two full baselines only after checking the smoke outputs. `score_baseline` does not invoke inference. Reruns require a new `--output-dir` through `scripts/run_inference.py`, or moving the prior output out of the way. The CLI refuses to overwrite existing outputs.

The smoke scores are a functionality check, not a basis for changing the fixed prompt, model or preprocessing settings.

## Outputs

Each run writes `run_metadata.json`, an exact `manifest_used.csv`, `predictions.jsonl`, `predictions.csv`, `runtime.csv`, `errors.jsonl`, and `summary.json`. Scoring adds `score_report.json` and `scored_predictions.csv`. A failed row is written with `status=failed`, retained in the predictions, and logged in `errors.jsonl`; the process exits nonzero and scoring refuses to report an overall result.

Prediction fields: `image_id`, `question_id`, `question`, `image_path`, `condition_name`, `model_id`, `model_revision`, `prompt_template_id`, `raw_prediction`, `normalized_prediction`, `generation_seconds`, `preprocess_seconds`, `total_seconds`, `status`, `error_message`. Only whitespace is normalized in saved predictions. The evaluator separately lowercases, removes VQA-style punctuation/articles, normalizes number words and common contractions, then computes `min(matching_reference_answers/3, 1)` over the 10 human answers, as requested in the experiment specification. This formula is a simplified VQA consensus score and should be named as such in reports.

Comparison writes `paired_scores.csv`, `latency_comparison.csv`, and `comparison_report.json`: clean/degraded soft accuracies, percentage-point difference, relative drop, counts of lower/equal/higher degraded scores, and paired latency. It rejects mismatched manifests, IDs, model revisions, prompts, generation settings, processor resolution, relevant config fields, and hardware/attention implementation.

## Checks possible without the H200

```bash
PYTHONPATH=src python3.11 -m unittest discover -s tests -v
```

These tests use generated images and a fake manifest; they do not download Qwen or execute CUDA. Full model loading, GPU behavior, real data paths, and performance remain unverified until the Linux H200 smoke test succeeds.
