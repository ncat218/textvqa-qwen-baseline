import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from textvqa_baseline.evaluator import normalize_vqa, score_answer
from textvqa_baseline.io_utils import PathResolver, parse_answers, sha256_file, validate_manifest
from textvqa_baseline.prompting import PROMPT, make_messages
from textvqa_baseline.scoring import compare_runs
from textvqa_baseline.io_utils import write_json, write_csv


class BaselineTests(unittest.TestCase):
    def test_prompt_is_fixed_and_has_image(self):
        messages = make_messages("What?", str(Path.cwd() / "x.jpg"))
        self.assertEqual(messages[0]["content"][1]["text"], PROMPT.format(question="What?"))
        self.assertTrue(messages[0]["content"][0]["image"].startswith("file:///"))

    def test_evaluator(self):
        self.assertEqual(normalize_vqa("The, TWO dogs!"), "2 dogs")
        self.assertEqual(score_answer("Cat", ["cat"] * 3 + ["dog"] * 7), 1.0)
        self.assertAlmostEqual(score_answer("Cat", ["cat"] * 2 + ["dog"] * 8), 2 / 3)
        fixture = pd.read_csv(Path(__file__).parent / "fixtures/fake_manifest.csv", dtype=str, keep_default_na=False)
        self.assertEqual(len(parse_answers(fixture.iloc[0].to_dict())), 10)

    def test_path_resolution_and_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            clean_root = root / "source" / "val_images"
            degraded = root / "dataset" / "main" / "images" / "realistic_mix" / "L2"
            clean_root.mkdir(parents=True)
            degraded.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(clean_root / "image_a.jpg")
            Image.new("RGB", (8, 8)).save(degraded / "image_a.png")
            resolver = PathResolver(root / "dataset", root / "source")
            self.assertEqual(resolver.resolve("/old/clean/image_a.jpg", "clean_image_path"), (clean_root / "image_a.jpg").resolve())
            self.assertEqual(resolver.resolve("/kaggle/working/textvqa_realistic_v2/main/images/realistic_mix/L2/image_a.png", "degraded_image_path"), (degraded / "image_a.png").resolve())
            fixture = pd.read_csv(Path(__file__).parent / "fixtures/fake_manifest.csv", dtype=str, keep_default_na=False)
            rows = []
            for i in range(1000):
                row = fixture.iloc[0].to_dict()
                row["question_id"] = str(i)
                rows.append(row)
            manifest = root / "final_main_manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            config = {"manifest_path": str(manifest), "data_root": str(root / "dataset"), "clean_root": str(root / "source")}
            frame, meta = validate_manifest(config)
            self.assertEqual(len(frame), 1000)
            self.assertEqual(meta["manifest_sha256"], sha256_file(manifest))
            self.assertEqual(frame["_resolved_clean"].nunique(), 1)
            frame_smoke, _ = validate_manifest(config, 16)
            self.assertEqual(frame_smoke.question_id.tolist(), [str(i) for i in range(16)])
            (clean_root / "image_a.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_manifest(config, 16)

    def test_paired_comparison_rejects_mismatched_settings(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            manifest = pd.read_csv(Path(__file__).parent / "fixtures/fake_manifest.csv", dtype=str, keep_default_na=False)
            common = dict(manifest_sha256="fake_full_manifest_hash", row_count=1,
                          model_id="Qwen/Qwen2.5-VL-3B-Instruct", model_revision="commit1",
                          prompt_template_id="textvqa_short_v1", generation_settings={"do_sample": False, "num_beams": 1, "max_new_tokens": 20},
                          processor_max_pixels=1605632,
                          runtime={"gpu_name": "H200", "gpu_memory_bytes": 100, "attention_implementation": "sdpa", "dtype": "bfloat16"})
            for name, column, answer in (("clean", "clean_image_path", "hello"), ("degraded", "degraded_image_path", "wrong")):
                folder = root / name
                folder.mkdir()
                write_csv(folder / "manifest_used.csv", manifest)
                pred = pd.DataFrame([dict(image_id="image_a", question_id="question_a", normalized_prediction=answer,
                                          status="ok", total_seconds=1.0)])
                write_csv(folder / "predictions.csv", pred)
                write_json(folder / "run_metadata.json", dict(**common, image_column=column,
                    config={"image_column": column, "experiment_name": name, "expected_condition": None if name == "clean" else "realistic_mix",
                            "expected_level": None if name == "clean" else "L2", "output_dir": name, "model_id": common["model_id"]},
                    config_sha256=name))
            report = compare_runs(root / "clean", root / "degraded", root / "comparison")
            self.assertEqual(report["clean_accuracy"], 1.0)
            self.assertEqual(report["degraded_accuracy"], 0.0)
            self.assertEqual(report["clean_higher_count"], 1)
            meta_path = root / "degraded" / "run_metadata.json"
            meta = json.loads(meta_path.read_text())
            meta["model_revision"] = "different_commit"
            write_json(meta_path, meta)
            with self.assertRaisesRegex(ValueError, "model_revision"):
                compare_runs(root / "clean", root / "degraded", root / "comparison_2")


if __name__ == "__main__":
    unittest.main()
