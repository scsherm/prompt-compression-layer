from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, path: Path, *, echo: bool = True, mirror_path: Path | None = None):
        self.path = path
        self.mirror_path = mirror_path
        self.echo = echo
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        if self.mirror_path:
            self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
            self.mirror_path.write_text("", encoding="utf-8")

    def event(self, event: str, **fields: Any) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **{key: _jsonable(value) for key, value in fields.items()},
        }
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._lock:
            for path in self._paths():
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            if self.echo:
                print(_human_line(row), file=sys.stderr, flush=True)

    def _paths(self) -> tuple[Path, ...]:
        if self.mirror_path and self.mirror_path != self.path:
            return (self.path, self.mirror_path)
        return (self.path,)


def _human_line(row: dict[str, Any]) -> str:
    event = row["event"]
    if event == "candidate_evaluated":
        return (
            f"[{row.get('stage')}] candidate={row.get('candidate_id')} "
            f"objective={row.get('objective_score')} drift={row.get('avg_semantic_drift')} "
            f"tok_red={row.get('token_reduction')} "
            f"cost=${row.get('estimated_cost_usd')} elapsed={row.get('elapsed_seconds')}s"
        )
    if event == "proposal_round_start":
        return (
            f"[round {row.get('round')}] proposing {row.get('batch_size')} complete prompts; "
            f"frontier={row.get('frontier_count')} trials={row.get('trial_count')}"
        )
    if event == "proposal_batch_generated":
        return f"[round {row.get('round')}] generated {row.get('candidate_count')} compressed prompts"
    if event == "search_round_complete":
        return (
            f"[round {row.get('round')}] frontier={row.get('frontier_count')} "
            f"best_reduction={row.get('best_token_reduction')} "
            f"best_behavior_loss={row.get('best_behavior_loss')}"
        )
    if event == "search_converged":
        return f"[search] converged after round {row.get('round')}: {row.get('reason')}"
    if event == "epoch_start":
        return f"[epoch {row.get('epoch')}] evaluating {row.get('candidate_count')} candidates on {row.get('example_count')} examples"
    if event == "epoch_frontier":
        return f"[epoch {row.get('epoch')}] frontier={row.get('frontier_count')} next_population={row.get('next_population_count')}"
    if event == "reference_build_start":
        return f"[reference] building behavioral references for {row.get('example_count')} inputs"
    if event == "reference_build_done":
        return f"[reference] built {row.get('reference_count')} behavioral references"
    if event == "population_seeded":
        return f"[chunk/explore] seeded {row.get('candidate_count')} candidates; original_instruction_tokens={row.get('original_instruction_tokens')}"
    if event == "population_seed_start":
        return (
            f"[chunk/explore] generating up to {row.get('population_size')} candidates "
            f"with proposer={row.get('proposer')}; original_instruction_tokens={row.get('original_instruction_tokens')}"
        )
    if event == "candidate_seeded":
        return (
            f"[seed] {row.get('candidate_count')}/{row.get('population_size')} "
            f"candidate={row.get('candidate_id')} chunker={row.get('chunker')} operator={row.get('operator')} "
            f"instruction_tokens={row.get('instruction_tokens')} tok_red={row.get('token_reduction')}"
        )
    if event == "chunking_plan":
        return f"[chunking] active={row.get('active_chunkers')} chunks={row.get('chunk_counts')}"
    if event == "mutation_done":
        return f"[epoch {row.get('epoch')}] mutated frontier into {row.get('next_population_count')} next candidates"
    if event in {"dev_start", "holdout_start"}:
        return f"[{event.removesuffix('_start')}] evaluating {row.get('candidate_count')} candidates on {row.get('example_count')} examples"
    if event in {"run_start", "dataset_split", "run_done"}:
        details = ", ".join(f"{key}={value}" for key, value in row.items() if key not in {"ts", "event"})
        return f"[{event}] {details}"
    return f"[{event}]"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
