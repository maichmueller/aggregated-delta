from __future__ import annotations

import gc
import multiprocessing as mp
from pathlib import Path

import pymimir as mm
import pymimir_rl as rl
import torch

from .. import train as base_train
from .model import StagedIWModel


def _evaluate_validation_problem_task(
    task: tuple[
        str,
        str,
        str,
        float,
        int,
        str,
        int | None,
        int | None,
    ],
) -> dict[str, object]:
    (
        problem_path_text,
        model_snapshot_path_text,
        reward_name,
        discount_factor,
        horizon,
        device_type,
        device_index,
        _,
    ) = task
    problem_path = Path(problem_path_text).resolve()
    model_snapshot_path = Path(model_snapshot_path_text)
    device = base_train._select_validation_worker_device(
        device_type, device_index
    )
    problem: mm.Problem | None = None
    model: StagedIWModel | None = None
    try:
        domain_path = base_train._resolve_problem_domain_path(problem_path)
        domain = mm.Domain(str(domain_path))
        problem = mm.Problem(domain, str(problem_path))
        reward_function = base_train._create_validation_worker_reward_function(
            reward_name,
            discount_factor,
        )
        model = StagedIWModel.load(domain, model_snapshot_path, device)
        greedy_policy = rl.GreedyPolicyTrajectorySampler(model, reward_function)
        with torch.no_grad():
            trajectory = greedy_policy.sample(
                [
                    (
                        problem.get_initial_state(),
                        problem.get_goal_condition(),
                    )
                ],
                horizon,
            )[0]
        return {
            "path": str(problem_path),
            "solved": trajectory.is_solution(),
            "length": len(trajectory),
            "td_error": base_train._compute_validation_td_error(trajectory),
            "discounted_return": base_train._compute_validation_discounted_return(
                trajectory, discount_factor
            ),
            "timed_out": False,
        }
    finally:
        if problem is not None:
            release_parallel_memory = getattr(
                problem, "release_parallel_memory", None
            )
            if callable(release_parallel_memory):
                try:
                    release_parallel_memory(True)
                except Exception:
                    pass
        base_train.clear_search_context_caches()
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        del model
        del problem
        gc.collect()


class StagedParallelFrontierAwarePolicyEvaluation(
    base_train.ParallelFrontierAwarePolicyEvaluation
):
    def evaluate(
        self, records: list[base_train.ProblemRecord]
    ) -> base_train.ValidationEvaluationResult:
        return super().evaluate(records)


class StagedDQNOptimizationContext(base_train.DQNOptimizationContext):
    def _get_parallel_validation_evaluator(
        self, args
    ) -> StagedParallelFrontierAwarePolicyEvaluation:
        assert self.validation_curriculum is not None
        if self.validation_parallel_evaluator is None:
            self.validation_parallel_evaluator = (
                StagedParallelFrontierAwarePolicyEvaluation(
                    self.evaluation_criteras,
                    args.validation_horizon,
                    self.reward_name,
                    self.discount_factor,
                    self.device,
                    self.validation_num_workers,
                    getattr(args, "validation_problem_timeout_seconds", None),
                    _evaluate_validation_problem_task,
                )
            )
        else:
            self.validation_parallel_evaluator.criterias = list(
                self.evaluation_criteras
            )
            self.validation_parallel_evaluator.horizon = args.validation_horizon
            self.validation_parallel_evaluator.reward_name = self.reward_name
            self.validation_parallel_evaluator.discount_factor = (
                self.discount_factor
            )
            self.validation_parallel_evaluator.validation_problem_timeout_seconds = getattr(
                args, "validation_problem_timeout_seconds", None
            )
            self.validation_parallel_evaluator.worker_task = (
                _evaluate_validation_problem_task
            )
        return self.validation_parallel_evaluator

    def _get_serial_validation_evaluator(
        self,
        args,
        model_snapshot_path=None,
    ):
        timeout_seconds = getattr(
            args, "validation_problem_timeout_seconds", None
        )
        if timeout_seconds is None:
            return super()._get_serial_validation_evaluator(
                args, model_snapshot_path
            )
        if model_snapshot_path is None:
            raise RuntimeError(
                "Timeout-aware serial validation requires a model snapshot path."
            )
        return base_train.TimeoutAwareFrontierAwarePolicyEvaluation(
            self.evaluation_criteras,
            args.validation_horizon,
            self.reward_name,
            self.discount_factor,
            self.device,
            model_snapshot_path,
            timeout_seconds,
            worker_task=_evaluate_validation_problem_task,
        )
