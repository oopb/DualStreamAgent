from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import VisualClawRound, VisualClawScenario

DATASET_ID = "UCSC-VLAA/VisualClawArena"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(item)
    return rows


def _resolve_inside(root: Path, value: str | Path) -> Path:
    target = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    root = root.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Dataset path escapes root: {value}")
    return target


def locate_dataset_root(path: str | Path) -> Path:
    """Locate the unpacked VisualClawArena release root."""
    source = Path(path).expanduser().resolve()
    candidates = [source]
    if source.exists() and source.is_dir():
        candidates.extend(child for child in source.iterdir() if child.is_dir())
    for candidate in candidates:
        if (
            (candidate / "manifest.json").is_file()
            and (candidate / "manifests").is_dir()
            and (candidate / "scenarios").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not locate an unpacked VisualClawArena release under {source}. "
        "Expected manifest.json, manifests/, and scenarios/."
    )


class VisualClawDataset:
    """Reader for the Hugging Face VisualClawArena release layout."""

    def __init__(self, root: str | Path):
        self.root = locate_dataset_root(root)
        self.manifest = _read_json(self.root / "manifest.json")
        if not isinstance(self.manifest, dict):
            raise ValueError("manifest.json must contain an object")
        self._scenario_rows = _read_jsonl(self.root / "manifests" / "scenarios.jsonl")
        self._scenario_index = {
            str(row.get("scenario_id") or row.get("id")): row
            for row in self._scenario_rows
            if row.get("scenario_id") or row.get("id")
        }

    @property
    def version(self) -> str:
        return str(self.manifest.get("version") or "unknown")

    def list_scenario_ids(self) -> list[str]:
        if self._scenario_rows:
            return [
                str(row.get("scenario_id") or row.get("id"))
                for row in self._scenario_rows
                if row.get("scenario_id") or row.get("id")
            ]
        return sorted(path.name for path in (self.root / "scenarios").iterdir() if path.is_dir())

    def _scenario_paths(self, scenario_id: str) -> tuple[dict[str, Any], Path, Path, Path]:
        row = dict(self._scenario_index.get(scenario_id, {}))
        scenario_root = self.root / "scenarios" / scenario_id
        if not scenario_root.is_dir():
            raise FileNotFoundError(f"VisualClawArena scenario not found: {scenario_id}")
        data_value = row.get("scenario_data_path") or row.get("data_path")
        spec_value = row.get("scenario_spec_path") or row.get("spec_path")
        data_dir = _resolve_inside(self.root, data_value) if data_value else scenario_root / "data"
        spec_dir = _resolve_inside(self.root, spec_value) if spec_value else scenario_root / "spec"
        return row, scenario_root, data_dir.resolve(), spec_dir.resolve()

    def load_scenario(
        self,
        scenario_id: str,
        *,
        release_only: bool = True,
        include_deprecated: bool = False,
        video_required_only: bool = False,
        round_types: Iterable[str] | None = None,
        round_ids: Iterable[str] | None = None,
        max_rounds: int = 0,
    ) -> VisualClawScenario:
        row, scenario_root, data_dir, spec_dir = self._scenario_paths(scenario_id)
        questions_path = spec_dir / "questions.json"
        if not questions_path.is_file():
            raise FileNotFoundError(f"Missing VisualClawArena questions file: {questions_path}")
        questions = _read_json(questions_path)
        raw_rounds = questions.get("rounds", []) if isinstance(questions, dict) else questions
        if not isinstance(raw_rounds, list):
            raise ValueError(f"{questions_path} must contain a rounds list")

        wanted_types = {str(value) for value in round_types or []}
        wanted_ids = {str(value) for value in round_ids or []}
        rounds: list[VisualClawRound] = []
        for index, item in enumerate(raw_rounds):
            if not isinstance(item, dict):
                continue
            round_item = VisualClawRound.from_dict(item, index=index)
            if release_only and not round_item.included_in_release_eval:
                continue
            if not include_deprecated and round_item.deprecated:
                continue
            if video_required_only and not round_item.video_required:
                continue
            if wanted_types and round_item.round_type not in wanted_types:
                continue
            if wanted_ids and round_item.round_id not in wanted_ids:
                continue
            rounds.append(round_item)
            if max_rounds > 0 and len(rounds) >= max_rounds:
                break

        if wanted_ids:
            found = {item.round_id for item in rounds}
            missing = wanted_ids - found
            if missing:
                raise ValueError(f"Round ids not found after filtering in {scenario_id}: {sorted(missing)}")

        clip_values = row.get("clip_paths") or row.get("clips") or []
        if isinstance(clip_values, str):
            clip_values = [clip_values]
        clip_paths: list[Path] = []
        for value in clip_values:
            candidate = _resolve_inside(self.root, str(value))
            if candidate.is_file():
                clip_paths.append(candidate)
        if not clip_paths:
            clip_dir = data_dir / "clip"
            for pattern in ("*.mp4", "*.mov", "*.mkv", "*.webm", "*.avi"):
                clip_paths.extend(sorted(clip_dir.glob(pattern)) if clip_dir.exists() else [])

        return VisualClawScenario(
            scenario_id=scenario_id,
            dataset_root=self.root,
            scenario_root=scenario_root.resolve(),
            data_dir=data_dir,
            spec_dir=spec_dir,
            workspace_dir=(data_dir / "workspace").resolve(),
            sessions_dir=(data_dir / "sessions").resolve(),
            updates_dir=(data_dir / "updates").resolve(),
            clip_paths=clip_paths,
            rounds=rounds,
            source_bucket=str(row.get("source_bucket") or row.get("bucket") or ""),
            metadata=row,
        )

    def validate(self, *, max_scenarios: int = 0) -> dict[str, Any]:
        scenario_ids = self.list_scenario_ids()
        if max_scenarios > 0:
            scenario_ids = scenario_ids[:max_scenarios]
        errors: list[str] = []
        round_count = 0
        clip_count = 0
        for scenario_id in scenario_ids:
            try:
                scenario = self.load_scenario(
                    scenario_id,
                    release_only=False,
                    include_deprecated=True,
                )
                round_count += len(scenario.rounds)
                clip_count += len(scenario.clip_paths)
                if not scenario.workspace_dir.exists():
                    errors.append(f"{scenario_id}: missing data/workspace")
                if not scenario.clip_paths:
                    errors.append(f"{scenario_id}: no clip found")
            except Exception as exc:
                errors.append(f"{scenario_id}: {exc}")
        return {
            "dataset": str(self.manifest.get("dataset") or "VisualClawArena"),
            "version": self.version,
            "root": str(self.root),
            "scenarios_checked": len(scenario_ids),
            "rounds_loaded": round_count,
            "clips_found": clip_count,
            "errors": errors,
            "valid": not errors,
        }


def download_visualclaw_arena(
    local_dir: str | Path,
    *,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    """Download the public dataset through optional ``huggingface_hub``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the arena extra first: pip install -e '.[arena]'"
        ) from exc
    destination = Path(local_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=str(destination),
    )
    return locate_dataset_root(destination)
