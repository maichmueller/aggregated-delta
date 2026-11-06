from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pymimir as mm
import pymimir_rl

from loguru import logger


SUPPORTED_GENERATOR_DOMAINS = {
    "blocksworld-ipc",
    "childsnack-ipc",
    "ferry-ipc",
    "floortile-ipc",
    "miconic-ipc",
    "spanner-ipc",
}

FALLBACK_GENERATOR_DOMAINS = {
    "rovers-ipc",
    "satellite-ipc",
    "sokoban-ipc",
    "transport-ipc",
}


@dataclass(frozen=True)
class ProblemRecord:
    path: Path
    problem: mm.Problem
    object_count: int
    goal_size: int


@dataclass(frozen=True)
class ValidationEvaluationResult:
    evaluation: list[float | int]
    solved_by_path: dict[str, bool]
    total_validation: int
    validation_total: int
    validation_solved_count: int
    test_total: int
    test_solved_count: int
    test_evaluated: bool
    timed_out_by_path: dict[str, bool] = field(default_factory=dict)
    timed_out_count: int = 0


@dataclass(frozen=True)
class CurriculumExpansion:
    expanded: bool
    message: str | None = None
    previous_total: int = 0
    new_total: int = 0


@dataclass(frozen=True)
class CurriculumSnapshot:
    requested_strategy: str
    effective_strategy: str
    fallback_reason: str | None
    active_problem_paths: tuple[Path, ...]
    frontier_problem_paths: tuple[Path, ...]


class ValidationEvaluator(Protocol):
    def evaluate(
        self, records: Sequence[ProblemRecord]
    ) -> ValidationEvaluationResult: ...


class FrontierAwarePolicyEvaluation:
    def __init__(
        self,
        criterias: Sequence[pymimir_rl.EvaluationCriteria],
        trajectory_sampler: pymimir_rl.TrajectorySampler,
        horizon: int,
        pack_size: int = 3,
    ) -> None:
        self.criterias = list(criterias)
        self.trajectory_sampler = trajectory_sampler
        self.horizon = horizon
        self.pack_size = pack_size

    def evaluate(
        self, records: Sequence[ProblemRecord]
    ) -> ValidationEvaluationResult:
        if not records:
            return ValidationEvaluationResult([], {}, 0, 0, 0, 0, 0, False)
        if self.pack_size < 1:
            raise ValueError("pack_size must be at least 1.")

        ordered_records = sorted(records, key=_validation_pack_key)
        successful_trajectories = []
        solved_by_path: dict[str, bool] = {}
        for idx in range(0, len(ordered_records), self.pack_size):
            pack = ordered_records[idx : idx + self.pack_size]
            state_goals = [
                (
                    record.problem.get_initial_state(),
                    record.problem.get_goal_condition(),
                )
                for record in pack
            ]
            trajectories = self.trajectory_sampler.sample(
                state_goals, self.horizon
            )
            solved_in_pack = 0
            for record, trajectory in zip(pack, trajectories, strict=True):
                solved = trajectory.is_solution()
                solved_by_path[str(record.path)] = solved
                if solved:
                    successful_trajectories.append(trajectory)
                    solved_in_pack += 1
            if solved_in_pack != len(pack):
                break
        evaluation = [
            criteria.evaluate(successful_trajectories)
            for criteria in self.criterias
        ]
        return ValidationEvaluationResult(
            evaluation=evaluation,
            solved_by_path=solved_by_path,
            total_validation=len(records),
            validation_total=len(records),
            validation_solved_count=sum(
                1 for solved in solved_by_path.values() if solved
            ),
            test_total=0,
            test_solved_count=0,
            test_evaluated=False,
        )


def detect_pddl_generators_root() -> Path | None:
    candidates: list[Path] = []
    env_root = os.environ.get("PDDL_GENERATORS_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    repo_root = Path(__file__).resolve().parent
    candidates.extend(
        [
            repo_root / "pddl-generators",
            repo_root.parent / "pddl-generators",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def compare_evaluations(
    criterias: Sequence[pymimir_rl.EvaluationCriteria],
    current: Sequence[float | int],
    best: Sequence[float | int] | None,
) -> bool:
    if best is None:
        return True
    for index, criteria in enumerate(criterias):
        comparison = int(criteria.compare(current[index], best[index]))
        if comparison < 0:
            return False
        if comparison > 0:
            return True
    return False


class ValidationCurriculumManager:
    STATE_VERSION = 3

    def __init__(
        self,
        requested_strategy: str,
        domain_name: str,
        domain: mm.Domain,
        validation_path: Path | None,
        test_path: Path | None,
        cache_dir: Path,
        frontier_size: int,
        frontier_threshold: float,
        seed: int,
        pddl_generators_root: Path | None = None,
    ) -> None:
        self.requested_strategy = requested_strategy
        self.domain_name = domain_name
        self.domain = domain
        self.validation_path = (
            validation_path.resolve() if validation_path else None
        )
        self.test_path = test_path.resolve() if test_path else None
        self.cache_dir = cache_dir.resolve()
        self.generated_dir = self.cache_dir / "generated"
        self.state_path = self.cache_dir / "state.json"
        self.frontier_size = frontier_size
        self.frontier_threshold = frontier_threshold
        self.seed = seed
        self.pddl_generators_root = (
            pddl_generators_root.resolve() if pddl_generators_root else None
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self._record_cache: dict[Path, ProblemRecord] = {}
        self._validation_seed_records: list[ProblemRecord] = []
        self._test_records: list[ProblemRecord] = []
        self._state: dict[str, object] | None = None
        self._adapter: GeneratorDomainAdapter | None = None

        self._load_support_data()
        self._load_or_initialize_state()

    @property
    def effective_strategy(self) -> str:
        assert self._state is not None
        return str(self._state["effective_strategy"])

    @property
    def fallback_reason(self) -> str | None:
        assert self._state is not None
        value = self._state.get("fallback_reason")
        return str(value) if value else None

    def snapshot(self) -> CurriculumSnapshot:
        frontier_paths = tuple(
            record.path for record in self.frontier_records()
        )
        return CurriculumSnapshot(
            requested_strategy=self.requested_strategy,
            effective_strategy=self.effective_strategy,
            fallback_reason=self.fallback_reason,
            active_problem_paths=tuple(self.active_problem_paths()),
            frontier_problem_paths=frontier_paths,
        )

    def active_problem_paths(self) -> list[Path]:
        return [record.path for record in self.active_records()]

    def active_records(self) -> list[ProblemRecord]:
        return [
            self._get_record(Path(path)) for path in self._state_active_paths()
        ]

    def frontier_records(self) -> list[ProblemRecord]:
        if self.effective_strategy == "test_curriculum":
            return []
        batches = self._state_batches()
        if not batches:
            return []
        return [self._get_record(Path(path)) for path in batches[-1]]

    def maybe_expand(
        self, solved_by_path: Mapping[str, bool]
    ) -> CurriculumExpansion:
        if self.effective_strategy == "test_curriculum":
            return CurriculumExpansion(False)
        frontier = self.frontier_records()
        if not frontier:
            return CurriculumExpansion(False)
        solved_count = sum(
            1
            for record in frontier
            if solved_by_path.get(str(record.path), False)
        )
        ratio = solved_count / len(frontier)
        if ratio < self.frontier_threshold:
            return CurriculumExpansion(False)
        previous_total = len(self.active_problem_paths())
        expansion = self._expand_generator_curriculum()
        if expansion.expanded:
            return CurriculumExpansion(
                expanded=True,
                message=expansion.message,
                previous_total=previous_total,
                new_total=len(self.active_problem_paths()),
            )
        return expansion

    def evaluate_with_frontier(
        self,
        evaluator: ValidationEvaluator,
    ) -> ValidationEvaluationResult:
        if self.effective_strategy == "test_curriculum":
            validation_records = [
                self._get_record(Path(path)) for path in self._state_base_paths()
            ]
            validation_result = evaluator.evaluate(validation_records)
            validation_solved_count = sum(
                1
                for record in validation_records
                if validation_result.solved_by_path.get(str(record.path), False)
            )
            test_records = self._test_records
            test_total = len(test_records)
            test_evaluated = bool(test_records) and (
                validation_solved_count == len(validation_records)
            )
            if test_evaluated:
                test_result = evaluator.evaluate(test_records)
                combined_evaluation = _sum_evaluations(
                    validation_result.evaluation, test_result.evaluation
                )
                solved_by_path = dict(validation_result.solved_by_path)
                solved_by_path.update(test_result.solved_by_path)
                timed_out_by_path = dict(validation_result.timed_out_by_path)
                timed_out_by_path.update(test_result.timed_out_by_path)
                test_solved_count = sum(
                    1
                    for record in test_records
                    if test_result.solved_by_path.get(str(record.path), False)
                )
                total_validation = len(validation_records) + test_total
                return ValidationEvaluationResult(
                    evaluation=combined_evaluation,
                    solved_by_path=solved_by_path,
                    total_validation=total_validation,
                    validation_total=len(validation_records),
                    validation_solved_count=validation_solved_count,
                    test_total=test_total,
                    test_solved_count=test_solved_count,
                    test_evaluated=True,
                    timed_out_by_path=timed_out_by_path,
                    timed_out_count=(
                        validation_result.timed_out_count
                        + test_result.timed_out_count
                    ),
                )
            return ValidationEvaluationResult(
                evaluation=list(validation_result.evaluation),
                solved_by_path=dict(validation_result.solved_by_path),
                total_validation=len(validation_records),
                validation_total=len(validation_records),
                validation_solved_count=validation_solved_count,
                test_total=test_total,
                test_solved_count=0,
                test_evaluated=False,
                timed_out_by_path=dict(validation_result.timed_out_by_path),
                timed_out_count=validation_result.timed_out_count,
            )

        active_records = self.active_records()
        result = evaluator.evaluate(active_records)
        frontier = self.frontier_records()
        if not frontier:
            return result
        solved_count = sum(
            1
            for record in frontier
            if result.solved_by_path.get(str(record.path), False)
        )
        ratio = solved_count / len(frontier)
        return ValidationEvaluationResult(
            evaluation=result.evaluation,
            solved_by_path=result.solved_by_path,
            total_validation=result.total_validation,
            validation_total=result.total_validation,
            validation_solved_count=sum(
                1 for solved in result.solved_by_path.values() if solved
            ),
            test_total=0,
            test_solved_count=0,
            test_evaluated=False,
            timed_out_by_path=result.timed_out_by_path,
            timed_out_count=result.timed_out_count,
        )

    def _load_support_data(self) -> None:
        if self.validation_path is not None and self.validation_path.exists():
            self._validation_seed_records = self._collect_records(
                self.validation_path
            )
        if self.test_path is not None and self.test_path.exists():
            self._test_records = self._collect_records(self.test_path)

    def _load_or_initialize_state(self) -> None:
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text())
                if self._state_matches_request(payload):
                    self._state = payload
                    return
            except Exception as exc:
                logger.warning(
                    "Ignoring unreadable validation curriculum state '{}': {}",
                    self.state_path,
                    exc,
                )
        self._state = self._initialize_state()
        self._persist_state()

    def _state_matches_request(self, payload: Mapping[str, object]) -> bool:
        return (
            int(payload.get("state_version", -1)) == self.STATE_VERSION
            and payload.get("requested_strategy") == self.requested_strategy
            and payload.get("domain_name") == self.domain_name
            and payload.get("validation_path")
            == self._path_text(self.validation_path)
            and payload.get("test_path") == self._path_text(self.test_path)
            and int(payload.get("frontier_size", -1)) == self.frontier_size
            and float(payload.get("frontier_threshold", -1.0))
            == self.frontier_threshold
            and int(payload.get("seed", -1)) == self.seed
            and payload.get("pddl_generators_root")
            == self._path_text(self.pddl_generators_root)
        )

    def _initialize_state(self) -> dict[str, object]:
        effective_strategy = self.requested_strategy
        fallback_reason = None
        if self.requested_strategy == "generator_curriculum":
            if self.domain_name in FALLBACK_GENERATOR_DOMAINS:
                effective_strategy = "test_curriculum"
                fallback_reason = (
                    f"Falling back to test_curriculum because '{self.domain_name}' "
                    "is not supported by the first-pass generator curriculum."
                )
                logger.warning(fallback_reason)
            elif self.domain_name not in SUPPORTED_GENERATOR_DOMAINS:
                effective_strategy = "test_curriculum"
                fallback_reason = (
                    f"Falling back to test_curriculum because '{self.domain_name}' "
                    "has no generator adapter."
                )
                logger.warning(fallback_reason)
        if effective_strategy == "generator_curriculum":
            batches = [
                [str(record.path) for record in batch]
                for batch in _group_by_object_count(
                    self._validation_seed_records
                )
            ]
            next_target = (
                max(
                    record.object_count
                    for record in self._validation_seed_records
                )
                + 1
                if self._validation_seed_records
                else 1
            )
            state = {
                "state_version": self.STATE_VERSION,
                "requested_strategy": self.requested_strategy,
                "effective_strategy": effective_strategy,
                "domain_name": self.domain_name,
                "validation_path": self._path_text(self.validation_path),
                "test_path": self._path_text(self.test_path),
                "frontier_size": self.frontier_size,
                "frontier_threshold": self.frontier_threshold,
                "seed": self.seed,
                "pddl_generators_root": self._path_text(
                    self.pddl_generators_root
                ),
                "fallback_reason": fallback_reason,
                "base_problem_paths": [],
                "batches": batches,
                "awaiting_validation_mastery": False,
                "next_test_bucket_index": None,
                "next_target_object_count": next_target,
                "skipped_object_counts": [],
            }
            self._ensure_generator_adapter()
            return state
        if not self._test_records:
            raise RuntimeError(
                "test_curriculum requires a non-empty test split."
            )
        if self._validation_seed_records:
            return {
                "state_version": self.STATE_VERSION,
                "requested_strategy": self.requested_strategy,
                "effective_strategy": "test_curriculum",
                "domain_name": self.domain_name,
                "validation_path": self._path_text(self.validation_path),
                "test_path": self._path_text(self.test_path),
                "frontier_size": self.frontier_size,
                "frontier_threshold": self.frontier_threshold,
                "seed": self.seed,
                "pddl_generators_root": self._path_text(
                    self.pddl_generators_root
                ),
                "fallback_reason": fallback_reason,
                "base_problem_paths": [
                    str(record.path) for record in self._validation_seed_records
                ],
                "batches": [],
                "next_target_object_count": None,
                "skipped_object_counts": [],
            }
        logger.warning(
            "test_curriculum was requested without a validation split; "
            "treating the test split as the primary validation set."
        )
        return {
            "state_version": self.STATE_VERSION,
            "requested_strategy": self.requested_strategy,
            "effective_strategy": "test_curriculum",
            "domain_name": self.domain_name,
            "validation_path": self._path_text(self.validation_path),
            "test_path": self._path_text(self.test_path),
            "frontier_size": self.frontier_size,
            "frontier_threshold": self.frontier_threshold,
            "seed": self.seed,
            "pddl_generators_root": self._path_text(self.pddl_generators_root),
            "fallback_reason": fallback_reason,
            "base_problem_paths": [
                str(record.path) for record in self._test_records
            ],
            "batches": [],
            "next_target_object_count": None,
            "skipped_object_counts": [],
        }

    def _ensure_generator_adapter(self) -> None:
        if self._adapter is not None:
            return
        if self.pddl_generators_root is None:
            raise RuntimeError(
                "generator_curriculum requires --pddl_generators_root, "
                "$PDDL_GENERATORS_ROOT, or a repo-local pddl-generators checkout "
                "at ./pddl-generators or ../pddl-generators."
            )
        self._adapter = GeneratorDomainAdapter(
            domain_name=self.domain_name,
            domain=self.domain,
            pddl_generators_root=self.pddl_generators_root,
            output_dir=self.generated_dir,
            validation_seed_records=self._validation_seed_records,
            base_seed=self.seed,
        )

    def _state_batches(self) -> list[list[str]]:
        assert self._state is not None
        return [
            [str(path) for path in batch]
            for batch in list(self._state.get("batches", []))
        ]

    def _state_base_paths(self) -> list[str]:
        assert self._state is not None
        return [
            str(path)
            for path in list(self._state.get("base_problem_paths", []))
        ]

    def _state_active_paths(self) -> list[str]:
        paths: list[str] = list(self._state_base_paths())
        for batch in self._state_batches():
            paths.extend(batch)
        return paths

    def _persist_state(self) -> None:
        assert self._state is not None
        self.state_path.write_text(json.dumps(self._state, indent=2))

    def _expand_generator_curriculum(self) -> CurriculumExpansion:
        assert self._state is not None
        self._ensure_generator_adapter()
        assert self._adapter is not None
        seen_signatures = {
            canonical_problem_text(Path(path).read_text())
            for path in self._state_active_paths()
        }
        target = int(self._state.get("next_target_object_count", 1))
        max_target = target + 128
        skipped: list[int] = list(self._state.get("skipped_object_counts", []))
        while target <= max_target:
            if not self._adapter.is_feasible_size(target):
                skipped.append(target)
                target += 1
                continue
            try:
                generated_paths = self._adapter.generate_batch(
                    target_object_count=target,
                    batch_size=self.frontier_size,
                    existing_signatures=seen_signatures,
                    batch_index=len(self._state_batches()),
                )
            except RuntimeError as exc:
                logger.warning(
                    "Validation generation failed at size {}: {}",
                    target,
                    exc,
                )
                skipped.append(target)
                target += 1
                continue
            if len(generated_paths) < self.frontier_size:
                logger.warning(
                    "Generated only {} / {} validation problems at size {}; "
                    "skipping to the next size.",
                    len(generated_paths),
                    self.frontier_size,
                    target,
                )
                skipped.append(target)
                target += 1
                continue
            self._state["batches"].append(
                [str(path) for path in generated_paths]
            )
            self._state["next_target_object_count"] = target + 1
            self._state["skipped_object_counts"] = skipped
            self._persist_state()
            return CurriculumExpansion(
                True,
                f"Expanded generator curriculum with {len(generated_paths)} problem(s) at size {target}.",
            )
        self._state["skipped_object_counts"] = skipped
        self._persist_state()
        logger.warning(
            "No feasible generator validation size found after trying {}..{}.",
            int(self._state.get("next_target_object_count", 1)),
            max_target,
        )
        return CurriculumExpansion(False)

    def _collect_records(self, input_path: Path) -> list[ProblemRecord]:
        if input_path.is_file():
            paths = [input_path.resolve()]
        else:
            paths = sorted(
                path.resolve()
                for path in input_path.glob("*.pddl")
                if path.name != "domain.pddl"
            )
        return sorted(
            (self._get_record(path) for path in paths), key=_sorted_record_key
        )

    def _get_record(self, path: Path) -> ProblemRecord:
        path = path.resolve()
        cached = self._record_cache.get(path)
        if cached is not None:
            return cached
        problem = mm.Problem(self.domain, str(path))
        record = ProblemRecord(
            path=path,
            problem=problem,
            object_count=len(problem.get_objects()),
            goal_size=len(problem.get_goal_condition()),
        )
        self._record_cache[path] = record
        return record

    @staticmethod
    def _path_text(path: Path | None) -> str | None:
        return str(path.resolve()) if path is not None else None


def _sum_evaluations(
    left: Sequence[float | int], right: Sequence[float | int]
) -> list[float | int]:
    if len(left) != len(right):
        raise ValueError(
            "Cannot combine validation and test evaluations with different lengths."
        )
    combined: list[float | int] = []
    for lhs, rhs in zip(left, right, strict=True):
        combined.append(lhs + rhs)
    return combined


class GeneratorDomainAdapter:
    def __init__(
        self,
        domain_name: str,
        domain: mm.Domain,
        pddl_generators_root: Path,
        output_dir: Path,
        validation_seed_records: Sequence[ProblemRecord],
        base_seed: int,
    ) -> None:
        self.domain_name = domain_name
        self.domain = domain
        self.root = pddl_generators_root.resolve()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_seed = base_seed
        self.reference_specs = [
            ReferenceSpec.from_record(domain_name, record)
            for record in validation_seed_records
        ]
        if not self.reference_specs:
            raise RuntimeError(
                "generator_curriculum requires validation seed problems."
            )

    def is_feasible_size(self, object_count: int) -> bool:
        return self._choose_parameters(object_count, 0) is not None

    def generate_batch(
        self,
        target_object_count: int,
        batch_size: int,
        existing_signatures: set[str],
        batch_index: int,
    ) -> list[Path]:
        generated: list[Path] = []
        signatures = set(existing_signatures)
        attempts = 0
        max_attempts = max(batch_size * 20, 40)
        while len(generated) < batch_size and attempts < max_attempts:
            params = self._choose_parameters(target_object_count, attempts)
            if params is None:
                break
            instance_index = len(generated)
            seed = _derived_seed(
                self.base_seed, target_object_count, batch_index, attempts
            )
            problem_name = (
                f"{self.domain_name}-dyn-{target_object_count:03d}-"
                f"{batch_index:02d}-{instance_index:02d}"
            )
            text = self._generate_problem_text(
                target_object_count, params, seed, problem_name
            )
            signature = canonical_problem_text(text)
            if signature in signatures:
                attempts += 1
                continue
            path = (
                self.output_dir
                / f"size_{target_object_count:03d}"
                / f"{problem_name}.pddl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            if not self._validate_generated_problem(path, target_object_count):
                path.unlink(missing_ok=True)
                attempts += 1
                continue
            signatures.add(signature)
            generated.append(path.resolve())
            attempts += 1
        return generated

    def _validate_generated_problem(
        self, problem_path: Path, target_object_count: int
    ) -> bool:
        try:
            problem = mm.Problem(self.domain, str(problem_path))
        except Exception as exc:
            logger.warning(
                "Generated validation problem '{}' did not parse: {}",
                problem_path,
                exc,
            )
            return False
        if len(problem.get_objects()) != target_object_count:
            logger.warning(
                "Generated validation problem '{}' has {} objects, expected {}.",
                problem_path,
                len(problem.get_objects()),
                target_object_count,
            )
            return False
        if problem.get_goal_condition().holds(problem.get_initial_state()):
            logger.warning(
                "Generated validation problem '{}' is trivial; discarding.",
                problem_path,
            )
            return False
        return True

    def _choose_reference(self, variant_index: int) -> "ReferenceSpec":
        return self.reference_specs[variant_index % len(self.reference_specs)]

    def _choose_parameters(
        self, target_object_count: int, variant_index: int
    ) -> dict[str, int | float] | None:
        ref = self._choose_reference(variant_index)
        match self.domain_name:
            case "blocksworld-ipc":
                towers = ref.goal_counts.get("on-table") or ref.goal_counts.get(
                    "clear", 1
                )
                towers = max(
                    1,
                    min(
                        target_object_count,
                        round(
                            towers
                            * target_object_count
                            / max(ref.object_count, 1)
                        ),
                    ),
                )
                return {"blocks": target_object_count, "towers": towers}
            case "childsnack-ipc":
                candidates = []
                for children in range(1, target_object_count + 1):
                    for trays in range(1, 5):
                        sandwiches = (
                            target_object_count - (3 * children) - trays - 3
                        )
                        if sandwiches < children or sandwiches <= 0:
                            continue
                        candidates.append(
                            {
                                "children": children,
                                "trays": trays,
                                "sandwiches": sandwiches,
                            }
                        )
                return _pick_best_candidate(
                    candidates,
                    {
                        "children": ref.scale_value(
                            "children", target_object_count
                        ),
                        "trays": ref.scale_value("trays", target_object_count),
                        "sandwiches": ref.scale_value(
                            "sandwiches", target_object_count
                        ),
                    },
                )
            case "ferry-ipc":
                candidates = [
                    {"cars": cars, "locations": locations}
                    for locations in range(2, target_object_count)
                    for cars in [target_object_count - locations]
                    if cars >= 1
                ]
                return _pick_best_candidate(
                    candidates,
                    {
                        "cars": ref.scale_value("cars", target_object_count),
                        "locations": ref.scale_value(
                            "locations", target_object_count
                        ),
                    },
                )
            case "floortile-ipc":
                candidates = []
                robot_cap = max(int(ref.params.get("robots", 1)) + 1, 4)
                for rows in range(1, target_object_count + 1):
                    for columns in range(1, target_object_count + 1):
                        base = (rows + 1) * columns + 2
                        if base >= target_object_count:
                            break
                        robots = target_object_count - base
                        if robots < 1 or robots > min(columns, robot_cap):
                            continue
                        candidates.append(
                            {
                                "rows": rows,
                                "columns": columns,
                                "robots": robots,
                            }
                        )
                return _pick_best_candidate(
                    candidates,
                    {
                        "rows": ref.scale_value("rows", target_object_count),
                        "columns": ref.scale_value(
                            "columns", target_object_count
                        ),
                        "robots": ref.scale_value(
                            "robots", target_object_count
                        ),
                    },
                )
            case "miconic-ipc":
                candidates = [
                    {
                        "floors": target_object_count - passengers,
                        "passengers": passengers,
                    }
                    for passengers in range(1, target_object_count - 1)
                    if target_object_count - passengers >= 2
                ]
                return _pick_best_candidate(
                    candidates,
                    {
                        "floors": ref.scale_value(
                            "floors", target_object_count
                        ),
                        "passengers": ref.scale_value(
                            "passengers", target_object_count
                        ),
                    },
                )
            case "spanner-ipc":
                candidates = []
                for nuts in range(1, target_object_count + 1):
                    for spanners in range(nuts, target_object_count + 1):
                        locations = target_object_count - spanners - nuts - 3
                        if locations < 1:
                            continue
                        candidates.append(
                            {
                                "spanners": spanners,
                                "nuts": nuts,
                                "locations": locations,
                            }
                        )
                return _pick_best_candidate(
                    candidates,
                    {
                        "spanners": ref.scale_value(
                            "spanners", target_object_count
                        ),
                        "nuts": ref.scale_value("nuts", target_object_count),
                        "locations": ref.scale_value(
                            "locations", target_object_count
                        ),
                    },
                )
        return None

    def _generate_problem_text(
        self,
        target_object_count: int,
        params: dict[str, int | float],
        seed: int,
        problem_name: str,
    ) -> str:
        match self.domain_name:
            case "blocksworld-ipc":
                return _generate_blocksworld_problem(
                    problem_name=problem_name,
                    blocks=int(params["blocks"]),
                    towers=int(params["towers"]),
                    seed=seed,
                )
            case "childsnack-ipc":
                return self._generate_childsnack_problem(
                    problem_name, params, seed
                )
            case "ferry-ipc":
                return self._generate_ferry_problem(problem_name, params, seed)
            case "floortile-ipc":
                return self._generate_floortile_problem(
                    problem_name, params, seed
                )
            case "miconic-ipc":
                return self._generate_miconic_problem(
                    problem_name, params, seed
                )
            case "spanner-ipc":
                return self._generate_spanner_problem(
                    problem_name, params, seed
                )
        raise RuntimeError(
            f"No generator implementation available for '{self.domain_name}'."
        )

    def _generate_childsnack_problem(
        self,
        problem_name: str,
        params: Mapping[str, int | float],
        seed: int,
    ) -> str:
        reference = self._choose_reference(seed)
        children = int(params["children"])
        sandwiches = int(params["sandwiches"])
        trays = int(params["trays"])
        allergic_children = min(
            children,
            max(
                0,
                int(
                    round(
                        children
                        * (
                            int(reference.params.get("allergic_children", 0))
                            / max(
                                1,
                                int(reference.params.get("children", children)),
                            )
                        )
                    )
                ),
            ),
        )
        gluten_factor = allergic_children / max(children, 1)
        const_ratio = sandwiches / max(children, 1)
        script = self.root / "childsnack" / "child-snack-generator.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "pool",
                str(seed),
                str(children),
                str(trays),
                f"{gluten_factor:.6f}",
                f"{const_ratio:.6f}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        text = result.stdout
        text = _replace_problem_name(text, problem_name)
        text = text.replace("(:domain child-snack)", "(:domain childsnack)")
        header = [
            f";; children={children}, allergic_children={allergic_children}, trays={trays}, sandwiches={sandwiches}, seed={seed}",
            "",
        ]
        return "".join(f"{line}\n" for line in header) + text.strip() + "\n"

    def _generate_ferry_problem(
        self,
        problem_name: str,
        params: Mapping[str, int | float],
        seed: int,
    ) -> str:
        cars = int(params["cars"])
        locations = int(params["locations"])
        binary = self.root / "ferry" / "ferry"
        result = subprocess.run(
            [str(binary), "-l", str(locations), "-c", str(cars)],
            check=True,
            capture_output=True,
            text=True,
        )
        rename = {f"l{idx}": f"loc{idx + 1}" for idx in range(locations)} | {
            f"c{idx}": f"car{idx + 1}" for idx in range(cars)
        }
        init_match = re.search(r"\(:init(.*?)\)\s*\(:goal", result.stdout, re.S)
        goal_match = re.search(r"\(:goal(.*?)\)\s*\)\s*$", result.stdout, re.S)
        if init_match is None or goal_match is None:
            raise RuntimeError("Unexpected ferry generator output.")
        init_lines = []
        for raw_line in init_match.group(1).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("(location ") or line.startswith("(car "):
                continue
            if line.startswith("(not-eq "):
                continue
            init_lines.append(_apply_renames(line, rename))
        goal_lines = [
            _apply_renames(line.strip(), rename)
            for line in goal_match.group(1).splitlines()
            if line.strip() and line.strip() != "(and" and line.strip() != ")"
        ]
        body = [
            f";; cars={cars}, locations={locations}, seed={seed}",
            "",
            f"(define (problem {problem_name})",
            " (:domain ferry)",
            " (:objects ",
            f"    {' '.join(f'car{i}' for i in range(1, cars + 1))} - car",
            f"    {' '.join(f'loc{i}' for i in range(1, locations + 1))} - location",
            " )",
            " (:init ",
            *[f"    {line}" for line in init_lines],
            " )",
            " (:goal  (and",
            *[f"   {line}" for line in goal_lines],
            "  ))",
            ")",
        ]
        return "\n".join(body) + "\n"

    def _generate_floortile_problem(
        self,
        problem_name: str,
        params: Mapping[str, int | float],
        seed: int,
    ) -> str:
        rows = int(params["rows"])
        columns = int(params["columns"])
        robots = int(params["robots"])
        script = self.root / "floortile" / "floortile-generator.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                problem_name,
                str(rows),
                str(columns),
                str(robots),
                "time",
                str(seed),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        text = result.stdout
        text = _replace_domain_name(text, "floortile")
        text = re.sub(r"tile_(\d+)-(\d+)", r"tile_\1_\2", text)
        text = re.sub(
            r"^\s*\(:metric minimize \(total-time\)\)\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )
        header = [
            f";; rows={rows}, columns={columns}, robots={robots}, seed={seed}",
            "",
        ]
        return "".join(f"{line}\n" for line in header) + text.strip() + "\n"

    def _generate_miconic_problem(
        self,
        problem_name: str,
        params: Mapping[str, int | float],
        seed: int,
    ) -> str:
        floors = int(params["floors"])
        passengers = int(params["passengers"])
        binary = self.root / "miconic" / "miconic"
        result = subprocess.run(
            [
                str(binary),
                "-f",
                str(floors),
                "-p",
                str(passengers),
                "-r",
                str(seed),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        text = result.stdout
        text = re.sub(r"\bp(\d+)\b", lambda m: f"p{int(m.group(1)) + 1}", text)
        text = re.sub(r"\bf(\d+)\b", lambda m: f"f{int(m.group(1)) + 1}", text)
        text = _replace_problem_name(text, problem_name)
        header = [
            f";; passengers={passengers}, floors={floors}, seed={seed}",
            "",
        ]
        return "".join(f"{line}\n" for line in header) + text.strip() + "\n"

    def _generate_spanner_problem(
        self,
        problem_name: str,
        params: Mapping[str, int | float],
        seed: int,
    ) -> str:
        spanners = int(params["spanners"])
        nuts = int(params["nuts"])
        locations = int(params["locations"])
        script = self.root / "spanner" / "spanner-generator.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(spanners),
                str(nuts),
                str(locations),
                "--seed",
                str(seed),
                "--problem-name",
                problem_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        text = result.stdout.replace("useable", "usable")
        header = [
            f";; spanners={spanners}, nuts={nuts}, locations={locations}, seed={seed}",
            "",
        ]
        return "".join(f"{line}\n" for line in header) + text.strip() + "\n"


@dataclass(frozen=True)
class ReferenceSpec:
    object_count: int
    params: dict[str, int]
    goal_counts: dict[str, int]

    @classmethod
    def from_record(
        cls, domain_name: str, record: ProblemRecord
    ) -> "ReferenceSpec":
        params = parse_header_parameters(record.path)
        goal_counts = parse_goal_counts(record.path)
        if domain_name == "blocksworld-ipc" and "blocks" not in params:
            params["blocks"] = record.object_count
        return cls(
            object_count=record.object_count,
            params=params,
            goal_counts=goal_counts,
        )

    def scale_value(self, key: str, target_object_count: int) -> int:
        value = int(self.params.get(key, 1))
        return max(
            1,
            round(value * target_object_count / max(self.object_count, 1)),
        )


def parse_header_parameters(problem_path: Path) -> dict[str, int]:
    lines = problem_path.read_text().splitlines()[:5]
    params: dict[str, int] = {}
    for line in lines:
        for key, value in re.findall(r"([A-Za-z_]+)=([0-9]+)", line):
            params[key] = int(value)
    return params


def parse_goal_counts(problem_path: Path) -> dict[str, int]:
    text = problem_path.read_text()
    match = re.search(r"\(:goal\s*\(and(.*?)\)\s*\)\s*\)", text, re.S)
    if match is None:
        return {}
    counts: dict[str, int] = {}
    for predicate in re.findall(r"\(([^()\s]+)", match.group(1)):
        counts[predicate] = counts.get(predicate, 0) + 1
    return counts


def canonical_problem_text(text: str) -> str:
    normalized_lines = []
    for line in text.splitlines():
        stripped = line.partition(";")[0].strip()
        if stripped:
            normalized_lines.append(stripped)
    return " ".join(normalized_lines)


def _group_by_object_count(
    records: Sequence[ProblemRecord],
) -> list[list[ProblemRecord]]:
    ordered = sorted(records, key=_sorted_record_key)
    groups: list[list[ProblemRecord]] = []
    for _, group in groupby(ordered, key=lambda record: record.object_count):
        groups.append(list(group))
    return groups


def _sorted_record_key(record: ProblemRecord) -> tuple[int, int, str]:
    if hasattr(record, "object_count") and hasattr(record, "goal_size"):
        return (int(record.object_count), int(record.goal_size), str(record.path))
    problem = record.problem
    return (
        len(problem.get_objects()),
        len(problem.get_goal_condition()),
        str(record.path),
    )


def _validation_pack_key(record: ProblemRecord) -> tuple[int, int, str]:
    if hasattr(record, "goal_size") and hasattr(record, "object_count"):
        return (int(record.goal_size), int(record.object_count), str(record.path))
    problem = record.problem
    return (
        len(problem.get_goal_condition()),
        len(problem.get_objects()),
        str(record.path),
    )


def _pick_best_candidate(
    candidates: Sequence[dict[str, int | float]],
    targets: Mapping[str, int | float],
) -> dict[str, int | float] | None:
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda candidate: sum(
            abs(float(candidate.get(key, 0)) - float(value))
            for key, value in targets.items()
        ),
    )
    return dict(best)


def _replace_problem_name(text: str, name: str) -> str:
    return re.sub(
        r"\(define\s+\(problem\s+[^)\s]+\)",
        f"(define (problem {name})",
        text,
        count=1,
    )


def _replace_domain_name(text: str, domain_name: str) -> str:
    return re.sub(
        r"\(:domain\s+[^)\s]+\)",
        f"(:domain {domain_name})",
        text,
        count=1,
    )


def _apply_renames(text: str, rename_map: Mapping[str, str]) -> str:
    ordered = sorted(
        rename_map.items(), key=lambda item: len(item[0]), reverse=True
    )
    for source, target in ordered:
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


def _derived_seed(
    base_seed: int, target_object_count: int, batch_index: int, attempt: int
) -> int:
    modulus = 2_147_483_647
    value = (
        base_seed * 1_000_003
        + target_object_count * 97
        + batch_index * 53
        + attempt * 11
    )
    return value % modulus or 1


def _generate_blocksworld_problem(
    problem_name: str, blocks: int, towers: int, seed: int
) -> str:
    rng = random.Random(seed)
    block_names = [f"b{index}" for index in range(1, blocks + 1)]
    init_tower_count = rng.randint(
        1, max(1, min(blocks, int(math.sqrt(blocks))))
    )
    init_towers = _random_towers(block_names, init_tower_count, rng)
    goal_towers = _random_towers(block_names, max(1, min(towers, blocks)), rng)
    init_atoms = _render_blocksworld_state(init_towers)
    goal_atoms = _render_blocksworld_state(goal_towers)
    goal_lines = [
        atom
        for atom in goal_atoms
        if atom.startswith("(clear ")
        or atom.startswith("(on ")
        or atom.startswith("(on-table ")
    ]
    header = [
        f";; blocks={blocks}, towers={len(goal_towers)}, seed={seed}",
        "",
    ]
    body = [
        f"(define (problem {problem_name})",
        " (:domain blocksworld)",
        f" (:objects {' '.join(block_names)} - object)",
        " (:init ",
        *[f"    {atom}" for atom in init_atoms],
        " )",
        " (:goal  (and ",
        *[f"    {atom}" for atom in goal_lines],
        "  ))",
        ")",
    ]
    return "".join(f"{line}\n" for line in header + body)


def _random_towers(
    blocks: Sequence[str], tower_count: int, rng: random.Random
) -> list[list[str]]:
    shuffled = list(blocks)
    rng.shuffle(shuffled)
    tower_count = max(1, min(tower_count, len(shuffled)))
    towers = [[block] for block in shuffled[:tower_count]]
    for block in shuffled[tower_count:]:
        towers[rng.randrange(len(towers))].append(block)
    for tower in towers:
        rng.shuffle(tower)
    return towers


def _render_blocksworld_state(towers: Sequence[Sequence[str]]) -> list[str]:
    atoms = ["(arm-empty)"]
    for tower in towers:
        if not tower:
            continue
        atoms.append(f"(on-table {tower[0]})")
        for index in range(1, len(tower)):
            atoms.append(f"(on {tower[index]} {tower[index - 1]})")
        atoms.append(f"(clear {tower[-1]})")
    return atoms
