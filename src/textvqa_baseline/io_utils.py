import ast
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import yaml


REQUIRED = {"image_id", "question_id", "question", "clean_image_path", "degraded_image_path", "condition", "level", "config_sha256", "source_manifest_sha256"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value) -> None:
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode())


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write(path, frame.to_csv(index=False, lineterminator="\n").encode())


def write_jsonl(path: Path, records: list[dict]) -> None:
    atomic_write(path, b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in records))


def load_config(path: Path) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"\$\{([^}]+)\}", raw)) - set(os.environ))
    if missing:
        raise ValueError(f"Missing environment variables in config: {missing}")
    raw = os.path.expandvars(raw)
    config = yaml.safe_load(raw)
    for key in ("manifest_path", "data_root", "clean_root", "output_dir"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    if config["model_id"] != "Qwen/Qwen2.5-VL-3B-Instruct":
        raise ValueError("Model ID must be Qwen/Qwen2.5-VL-3B-Instruct")
    if config["image_column"] not in {"clean_image_path", "degraded_image_path"}:
        raise ValueError("Invalid image column")
    if config["image_column"] == "clean_image_path" and (config["expected_condition"] is not None or config["expected_level"] is not None):
        raise ValueError("Clean run must have null expected condition and level")
    if config["image_column"] == "degraded_image_path" and (config["expected_condition"], config["expected_level"]) != ("realistic_mix", "L2"):
        raise ValueError("Degraded run must expect realistic_mix L2")
    if config["batch_size"] != 1:
        raise ValueError("Only batch_size=1 is implemented")
    if config["device"] != "cuda":
        raise ValueError("This baseline requires CUDA")
    return config


def parse_answers(row: dict) -> list[str]:
    for key in ("answers_json", "answers"):
        value = row.get(key, "")
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
        if isinstance(parsed, list) and len(parsed) == 10 and all(isinstance(x, str) for x in parsed):
            return parsed
    raise ValueError(f"question_id={row.get('question_id')} needs exactly 10 string reference answers")


class PathResolver:
    def __init__(self, data_root: Path, clean_root: Path):
        self.data_root = Path(data_root)
        self.clean_root = Path(clean_root)
        self._clean_index = None

    def _index_clean(self):
        if not self.clean_root.is_dir():
            raise FileNotFoundError(f"Clean image source directory is missing: {self.clean_root}")
        index = {}
        for path in self.clean_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index.setdefault(path.name.lower(), []).append(path)
        self._clean_index = index

    def resolve(self, value: str, column: str) -> Path:
        original = Path(value)
        if original.is_file():
            return original.resolve()
        if column == "degraded_image_path":
            parts = original.as_posix().split("/main/")
            if len(parts) != 2 or not parts[-1].startswith("images/realistic_mix/L2/"):
                raise FileNotFoundError(f"Cannot remap degraded path: {value}")
            candidate = self.data_root / "main" / parts[-1]
            if not candidate.is_file():
                raise FileNotFoundError(f"Missing degraded image: {candidate}")
            return candidate.resolve()
        if self._clean_index is None:
            self._index_clean()
        matches = self._clean_index.get(original.name.lower(), [])
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one clean image for {value}, found {len(matches)} under {self.clean_root}")
        return matches[0].resolve()


def validate_manifest(config: dict, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    path = Path(config["manifest_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(frame)
    if missing or not ({"answers_json", "answers"} & set(frame)):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}; answers column required")
    if len(frame) != 1000:
        raise ValueError(f"Expected exactly 1000 rows, got {len(frame)}")
    if frame.question_id.duplicated().any() or frame[["image_id", "question_id"]].duplicated().any():
        raise ValueError("Duplicate question_id or (image_id, question_id)")
    if any(frame[col].str.strip().eq("").any() for col in ("image_id", "question_id", "question", "clean_image_path", "degraded_image_path")):
        raise ValueError("Empty required manifest value")
    if not frame.condition.eq("realistic_mix").all() or not frame.level.eq("L2").all():
        raise ValueError("Final manifest must contain only realistic_mix L2")
    for row in frame.to_dict("records"):
        parse_answers(row)
    if limit is not None:
        if limit < 1 or limit > len(frame):
            raise ValueError("Invalid smoke limit")
        frame = frame.iloc[:limit].copy()
    resolver = PathResolver(Path(config["data_root"]), Path(config["clean_root"]))
    clean_paths, degraded_paths = [], []
    for row in frame.itertuples(index=False):
        clean_paths.append(str(resolver.resolve(row.clean_image_path, "clean_image_path")))
        degraded_paths.append(str(resolver.resolve(row.degraded_image_path, "degraded_image_path")))
    frame["_resolved_clean"] = clean_paths
    frame["_resolved_degraded"] = degraded_paths
    return frame, {"manifest_sha256": sha256_file(path), "manifest_path": str(path), "row_count": len(frame)}
