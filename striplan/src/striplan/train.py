import argparse
import gc
import math
import multiprocessing as mp
import json
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import time
from datetime import timedelta
from urllib.parse import quote
import pymimir as mm
import pymimir_rl as rl
import pymimir_rgnn as rgnn
import sys
import torch
import wandb
from torch.nn.functional import huber_loss

from .iw_depth import cli_requested_iw_depth_override, resolve_iw_depth_config
from .lgan import cli_requested_lgan_override, resolve_lgan_config
from .rw_return import (
    RW_RETURN_EXACT_BLOCKED,
    RW_RETURN_EXACT_FULL,
    RW_RETURN_NONE,
    cli_requested_rw_return_override,
    resolve_rw_return_config,
)
from ._native_backend import assert_native_backend_ready
from .iw import IWGroundAction
from .model import (
    IWModel,
    clear_search_context_caches,
    get_collection_action_counts,
    reset_collection_action_counts,
)
from .search_context import (
    reset_problem_search_contexts as _reset_problem_search_contexts,
)
from .regret import RegretCriteria
from .utils import create_device
from .validation_curriculum import (
    CurriculumExpansion,
    FrontierAwarePolicyEvaluation,
    ProblemRecord,
    ValidationEvaluationResult,
    ValidationCurriculumManager,
    compare_evaluations,
    detect_pddl_generators_root,
    _validation_pack_key,
)

from loguru import logger
from pathlib import Path


# Configure logger
logger.remove()
logger.add(sys.stdout, format="[{level}; {elapsed}] {message}")
# Suppress noisy beta warning from PyTorch index_reduce used by pymimir_rgnn
warnings.filterwarnings(
    "ignore",
    message=r".*index_reduce\(\) is in beta.*",
    category=UserWarning,
    module=r"pymimir_rgnn\.aggregations",
)


def compute_depth_order_aux_loss(
    depth_scores: torch.Tensor,
    depth_node_sizes: list[int] | torch.Tensor,
    margin: float = 0.0,
    inverse_gap_weighting: bool = True,
) -> torch.Tensor:
    if isinstance(depth_node_sizes, torch.Tensor):
        size_list = [int(size) for size in depth_node_sizes.tolist()]
    else:
        size_list = [int(size) for size in depth_node_sizes]
    zero = depth_scores.sum() * 0.0
    if not size_list:
        return zero
    offset = 0
    sample_losses: list[torch.Tensor] = []
    for size in size_list:
        sample_scores = depth_scores[offset : offset + size]
        offset += size
        if size < 2:
            continue
        source_positions, target_positions = torch.triu_indices(
            size,
            size,
            offset=1,
            device=depth_scores.device,
        )
        pair_losses = torch.nn.functional.softplus(
            margin
            + sample_scores.index_select(0, source_positions)
            - sample_scores.index_select(0, target_positions)
        )
        if inverse_gap_weighting:
            weights = 1.0 / (target_positions - source_positions).to(
                pair_losses.dtype
            )
        else:
            weights = torch.ones_like(pair_losses)
        sample_losses.append((weights * pair_losses).sum() / weights.sum())
    if not sample_losses:
        return zero
    return torch.stack(sample_losses).mean()


class DepthOrderAuxDQNOptimization(rl.DQNOptimization):
    def __init__(
        self,
        model: IWModel,
        model_optimizer: torch.optim.Optimizer,
        model_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        target_model: IWModel,
        discount_factor: float,
        mellowmax_factor: float,
        use_bounds_loss: bool,
        depth_order_aux_weight: float,
    ) -> None:
        super().__init__(
            model,
            model_optimizer,
            model_lr_scheduler,
            target_model,
            discount_factor,
            mellowmax_factor,
            use_bounds_loss,
        )
        self.depth_order_aux_weight = depth_order_aux_weight
        self.last_depth_order_aux_loss = 0.0
        self.last_total_loss = 0.0

    def __call__(
        self, transitions: list[rl.Transition], weights: torch.Tensor
    ) -> torch.Tensor:
        dead_end_value = -10000
        device = next(self.model.parameters()).device
        target_q_values = self._compute_targets(transitions, dead_end_value, device)
        state_goals = [
            (transition.current_state, transition.goal_condition)
            for transition in transitions
        ]
        if self.use_bounds_loss:
            lower_bounds, upper_bounds = self.get_value_bounds(
                transitions, device
            )
        self.model.train()
        batch_outputs = self.model.forward_with_aux(state_goals)
        batch_q_values = batch_outputs["action_outputs"]
        selected_q_values = torch.stack(
            [
                q_values[actions.index(transition.selected_action)]
                for (q_values, actions), transition in zip(
                    batch_q_values, transitions
                )
            ]
        )
        losses = huber_loss(
            selected_q_values,
            target_q_values,
            delta=1.0,
            reduction="none",
        )
        if self.use_bounds_loss:
            bounds_errors = selected_q_values - selected_q_values.clamp(
                lower_bounds,
                upper_bounds,
            ).detach()
            bounds_losses = huber_loss(
                bounds_errors,
                torch.zeros_like(bounds_errors),
                delta=1.0,
                reduction="none",
            )
            losses += bounds_losses
        losses *= weights.to(losses.device)
        depth_scores = batch_outputs["depth_scores"]
        depth_node_sizes = batch_outputs["depth_node_sizes"]
        aux_loss = (
            compute_depth_order_aux_loss(depth_scores, depth_node_sizes)
            if depth_scores is not None and depth_node_sizes is not None
            else losses.sum() * 0.0
        )
        total_loss = losses.mean() + self.depth_order_aux_weight * aux_loss
        self.model_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()
        self.model_lr_scheduler.step()
        self.last_depth_order_aux_loss = float(aux_loss.detach().item())
        self.last_total_loss = float(total_loss.detach().item())
        return losses.detach()


class ActionCountRewardFunction(rl.RewardFunction):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        return (
            -len(action.get_underlying_actions())
            if isinstance(action, IWGroundAction)
            else -1
        )


class DiscountedRewardFunction(rl.RewardFunction):
    def __init__(self, discount_factor: float) -> None:
        super().__init__()
        self.discount_factor = discount_factor

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        n = (
            len(action.get_underlying_actions())
            if isinstance(action, IWGroundAction)
            else 1
        )
        return -1 * (2 - self.discount_factor ** (n - 1))


class SymmetricDifferenceRewardFunction(rl.RewardFunction):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        atoms = set(current_state.get_atoms(ignore_static=True))
        atoms.symmetric_difference_update(
            successor_state.get_atoms(ignore_static=True)
        )
        return -len(atoms)

    def get_value_bounds(
        self,
        immediate_reward: float,
        future_rewards: float,
        part_of_solution: bool,
    ) -> tuple[float, float]:
        return (
            (immediate_reward + future_rewards, 0.0)
            if part_of_solution
            else (float("-inf"), float("inf"))
        )


def _resolve_iw_depth_cli_args(args: argparse.Namespace) -> None:
    args.iw_depth_override_requested = cli_requested_iw_depth_override(args)
    config = resolve_iw_depth_config(
        iw_depth_feature=args.iw_depth_feature,
        iw_depth_mode=args.iw_depth_mode,
        iw_depth_precedes_mode=args.iw_depth_precedes_mode,
        iw_depth_precedes_scan=args.iw_depth_precedes_scan,
    )
    args.iw_depth_feature = config.uses_init_scalar
    args.iw_depth_mode = config.mode
    args.iw_depth_precedes_mode = config.precedes_mode
    args.iw_depth_precedes_scan = config.precedes_scan


def _resolve_rw_return_cli_args(args: argparse.Namespace) -> None:
    args.rw_return_override_requested = cli_requested_rw_return_override(args)
    args.rw_return_config = resolve_rw_return_config(
        rw_return_feature_mode=args.rw_return_feature_mode,
        rw_walk_mode=args.rw_walk_mode,
        rw_return_k=args.rw_return_k,
        rw_return_include_depth_nodes=args.rw_return_include_depth_nodes,
        rw_return_lazy=args.rw_return_lazy,
        rw_return_lazy_beta=args.rw_return_lazy_beta,
        rw_return_block_size=args.rw_return_block_size,
        rw_return_force_block_threshold=args.rw_return_force_block_threshold,
    )
    args.rw_return_feature_mode = args.rw_return_config.mode
    args.rw_walk_mode = args.rw_return_config.walk_mode
    args.rw_return_k = args.rw_return_config.k_max
    args.rw_return_include_depth_nodes = (
        args.rw_return_config.include_depth_nodes
    )
    args.rw_return_lazy = args.rw_return_config.lazy_walk
    args.rw_return_lazy_beta = args.rw_return_config.lazy_beta
    args.rw_return_block_size = args.rw_return_config.block_size
    args.rw_return_force_block_threshold = (
        args.rw_return_config.force_block_threshold
    )


def _resolve_lgan_cli_args(args: argparse.Namespace) -> None:
    args.lgan_override_requested = cli_requested_lgan_override(args)
    args.lgan_config = resolve_lgan_config(
        include_lgan_edges=args.include_lgan_edges,
        lgan_anchor_sources=args.lgan_anchor_sources,
        lgan_tn_edge_pos=args.lgan_tn_edge_pos,
        lgan_nn_edge_pos=args.lgan_nn_edge_pos,
        lgan_rr_edge_pos=args.lgan_rr_edge_pos,
    )
    args.include_lgan_edges = args.lgan_config.include_lgan_edges
    args.lgan_anchor_sources = list(args.lgan_config.lgan_anchor_sources)
    args.lgan_tn_edge_pos = args.lgan_config.lgan_tn_edge_pos
    args.lgan_nn_edge_pos = args.lgan_config.lgan_nn_edge_pos
    args.lgan_rr_edge_pos = args.lgan_config.lgan_rr_edge_pos


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Settings for training with DQN."
    )
    parser.add_argument(
        "--encoding",
        default="concurrent_internal",
        type=str,
        choices=[
            "concurrent_internal",
            "concurrent_internal_tree",
            "concurrent_internal_tree_rooted",
            "concurrent_internal_comparison_tree",
            "concurrent_internal_action_tree",
            "concurrent_internal_actions",
            "concurrent_internal_actions_source_action",
            "concurrent_internal_actions_action_successor",
            "concurrent_internal_actions_source_action_successor",
            "concurrent_internal_actions_source_action_rooted",
            "concurrent_internal_actions_action_successor_rooted",
            "concurrent_internal_actions_source_action_successor_rooted",
            "concurrent_internal_joint_actions",
            "concurrent_internal_action_hybrid_tree",
            "sequential_internal",
            "sequential_delta",
            "sequential_delta_rooted",
            "sequential_aux",
            "sequential_comparison",
            "sequential_external",
            "ground_actions",
        ],
        help="Type of encoding strategy to use.",
    )
    parser.add_argument(
        "--train",
        required=True,
        type=Path,
        help="Path to directory with training instances (supports {domain} placeholder).",
    )
    parser.add_argument(
        "--validation",
        default=None,
        type=Path,
        help="Path to directory with validation instances (supports {domain} placeholder).",
    )
    parser.add_argument(
        "--test",
        default=None,
        type=Path,
        help="Path to directory with test instances used by test_curriculum (supports {domain} placeholder).",
    )
    parser.add_argument(
        "--validation_strategy",
        default="static",
        choices=["static", "test_curriculum", "generator_curriculum"],
        help="Validation scheme to use during training.",
    )
    parser.add_argument(
        "--validation_frontier_size",
        default=10,
        type=int,
        help="Number of validation problems to add when the frontier expands.",
    )
    parser.add_argument(
        "--validation_frontier_threshold",
        default=0.9,
        type=float,
        help="Solved-ratio threshold required to expand the active validation frontier.",
    )
    parser.add_argument(
        "--validation_num_workers",
        "--validation_workers",
        dest="validation_num_workers",
        default=None,
        type=int,
        help=(
            "Maximum number of concurrent validation worker processes. "
            "Use 0 for serial validation in the parent process. "
            "Defaults to one worker per active validation problem."
        ),
    )
    parser.add_argument(
        "--validation_problem_timeout",
        default="00:10:00",
        type=str,
        help=(
            "Optional per-problem validation timeout in HH:MM:SS. "
            "Applies individually to each validation/test problem."
        ),
    )
    parser.add_argument(
        "--pddl_generators_root",
        default=None,
        type=Path,
        help="Root directory of the pddl-generators checkout used by generator_curriculum. If omitted, auto-detects $PDDL_GENERATORS_ROOT, ./pddl-generators, or ../pddl-generators.",
    )
    parser.add_argument(
        "--validation_cache_dir",
        default=None,
        type=Path,
        help="Directory used to persist dynamic validation state and generated PDDL problems.",
    )
    parser.add_argument(
        "--hindsight",
        required=True,
        type=str,
        choices=["lifted", "propositional", "state", "partial_state", "none"],
        help="Type of hindsight to use.",
    )
    parser.add_argument(
        "--aggregation",
        default="smax",
        type=str,
        help='Aggregation function used by the policy model ("add", "mean", "smax", "hmax").',
    )
    parser.add_argument(
        "--embedding_size",
        default=32,
        type=int,
        help="Dimension of the embedding vector for each object used by the policy model.",
    )
    parser.add_argument(
        "--binary_embeddings",
        action="store_true",
        help="Use binary embeddings instead of real-valued embeddings.",
    )
    parser.add_argument(
        "--layers", default=10, type=int, help="Number of layers in the model."
    )
    parser.add_argument(
        "--bt_initial",
        default=1.0,
        type=float,
        help="The initial Boltzmann temperature.",
    )
    parser.add_argument(
        "--bt_final",
        default=0.1,
        type=float,
        help="The final Boltzmann temperature.",
    )
    parser.add_argument(
        "--bt_steps",
        default=1000,
        type=int,
        help="The number of episodes to reach the final Boltzmann temperature.",
    )
    parser.add_argument(
        "--lr_initial", default=0.001, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--lr_final", default=0.00001, type=float, help="Final learning rate."
    )
    parser.add_argument(
        "--lr_steps",
        default=300,
        type=float,
        help="Steps to episodes the final learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--discount_factor", default=0.999, type=float, help="Discount factor."
    )
    parser.add_argument(
        "--mellowmax_factor", default=10.0, type=float, help="Mellowmax factor."
    )
    parser.add_argument(
        "--train_horizon",
        default=20,
        type=int,
        help="Maximum rollout length for the training set.",
    )
    parser.add_argument(
        "--validation_horizon",
        default=40,
        type=int,
        help="Maximum rollout length for the validation set.",
    )
    parser.add_argument(
        "--val_freq",
        "--validation_freq",
        default=10,
        type=int,
        help="Run validation every N episodes (0 to disable).",
    )
    parser.add_argument(
        "--parallel_memory_cache_reset_episodes",
        default=50,
        type=int,
        help=(
            "Reserved training-memory knob. Full training search-context "
            "resets are disabled because replay and IW caches can retain "
            "stale state wrappers. "
            "Use values less than or equal to 0 to disable."
        ),
    )
    parser.add_argument(
        "--max_new_trajectories",
        default=100,
        type=int,
        help="Max number of new trajectories to derive.",
    )
    parser.add_argument(
        "--lifted_her_max_subgoals_per_size",
        default=2,
        type=int,
        help="Maximum number of lifted HER subgoals to keep per subgoal size.",
    )
    parser.add_argument(
        "--lifted_her_max_subgoal_size",
        default=10,
        type=int,
        help="Maximum size of a lifted HER subgoal.",
    )
    parser.add_argument(
        "--reward",
        "--reward_function",
        dest="reward",
        default="constant",
        choices=[
            "constant",
            "action_count",
            "discounted_jumps",
            "symmetric_difference",
        ],
        help="Reward signal to use (constant, action_count, symmetric_difference).",
    )
    parser.add_argument(
        "--min_buffer_size",
        default=10,
        type=int,
        help="Minimum size of the experience buffer to update model.",
    )
    parser.add_argument(
        "--max_buffer_size",
        default=100,
        type=int,
        help="Maximum size of the experience buffer.",
    )
    parser.add_argument(
        "--num_rollouts",
        default=4,
        type=int,
        help="Number of trajectories to compute in parallel.",
    )
    parser.add_argument(
        "--train_steps",
        default=32,
        type=int,
        help="Number of training steps per iteration.",
    )
    parser.add_argument(
        "--seed",
        default=np.random.randint(0, int(1e6)),
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU to be used."
    )
    parser.add_argument(
        "--device",
        default=None,
        type=int,
        help="Preferred CUDA device ID (if multiple GPUs are available).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information."
    )
    parser.add_argument(
        "--logfile",
        default=None,
        type=Path,
        help="If set, write logs to the specified file (supports {domain} placeholder).",
    )
    parser.add_argument(
        "--num_episodes",
        default=1000,
        type=int,
        help="Number of episodes to train.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="If set, save the models to the specified directory (supports {domain} placeholder).",
    )
    parser.add_argument(
        "--runs_root",
        default=Path("experiments"),
        type=Path,
        help="Base directory used when auto-generating run folders.",
    )
    parser.add_argument(
        "--run_group",
        default="lookahead",
        type=str,
        help="Subdirectory under runs_root used to group runs.",
    )
    parser.add_argument(
        "--run_prefix",
        default="run",
        type=str,
        help="Prefix used when auto-generating run folder names.",
    )
    parser.add_argument(
        "--experiment_name",
        "--wandb_name",
        dest="experiment_name",
        default=None,
        type=str,
        help="Experiment run name for Weights & Biases tracking.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str,
        help="Weights & Biases project name (defaults to WANDB_PROJECT or current directory).",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        type=str,
        help="Weights & Biases entity/team (defaults to WANDB_ENTITY or your W&B default).",
    )
    parser.add_argument(
        "--wandb_run_info",
        default=None,
        type=Path,
        help="Optional file path to store Weights & Biases run metadata for reuse (supports {domain} placeholder).",
    )
    parser.add_argument(
        "--domain",
        default=None,
        type=str,
        help="Optional domain identifier used to expand {domain} placeholders in other arguments.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging for this run.",
    )
    parser.add_argument(
        "--ground",
        action="store_true",
        help="Ground the instances before training.",
    )
    parser.add_argument(
        "--optimal_model",
        type=Path,
        help="Path to the optimal model, to use for calculating regret.",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        type=Path,
        help=(
            "Path to a checkpoint file to resume training from. "
            "The episode number is inferred from the filename (e.g., best_123-...)."
        ),
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="If set, will run the models in flat mode, which means that no IW is performed and only the applicable actions are added in the GNN",
    )
    parser.add_argument(
        "--projective_iw",
        action="store_true",
        help="Use projective IW(1) when generating IW macro-actions.",
    )
    parser.add_argument(
        "--typed_projection",
        action="store_true",
        help="Use typed projection inside projective IW(1).",
    )
    parser.add_argument(
        "--abstracted_iw",
        action="store_true",
        help="Use Abstracted IW when generating IW macro-actions.",
    )
    parser.add_argument(
        "--abstracted_iw_width",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Width k for Abstracted IW(k).",
    )
    parser.add_argument(
        "--base_abstracted_iw",
        action="store_true",
        help=(
            "Use Base-Abstracted IW: abstract all non-preserved object slots "
            "to one universal type."
        ),
    )
    parser.add_argument(
        "--projective_iw_keep_depth_one_novel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep depth-1 novel states when using projective IW(1).",
    )
    parser.add_argument(
        "--keep_goal_nonunary_atoms",
        action="store_true",
        help="Keep non-unary goal atoms in projective IW.",
    )
    parser.add_argument(
        "--iw_layer_ordering_strategy",
        choices=(
            "in_order",
            "reverse_order",
            "randomized",
            "goal_count_more",
            "goal_count_less",
        ),
        default=None,
        help="Optional layer ordering strategy for layer-based IW beam selection.",
    )
    parser.add_argument(
        "--iw_max_next_layer_states",
        type=int,
        default=None,
        help="Optional cap on novelty-checked next-layer states in IW.",
    )
    parser.add_argument(
        "--iw_beam_width",
        type=int,
        default=None,
        help="Optional beam width for layer-based IW.",
    )
    parser.add_argument(
        "--iw_beam_novelty_mode",
        choices=("all_tested", "survivors_only"),
        default=None,
        help="Novelty-table update mode for IW beam search.",
    )
    parser.add_argument(
        "--iw_max_depth",
        type=int,
        default=None,
        help="Optional maximum IW search depth. Use -1 for no depth limit.",
    )
    parser.add_argument(
        "--iw1_precheck_add_effect_novelty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pre-check successor add effects before the full IW(1) novelty test. "
            "Applies to both standard and projective IW."
        ),
    )
    parser.add_argument(
        "--iw1_atom_first_mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable atom-first successor ordering inside IW(1). "
            "Applies to both standard and projective IW."
        ),
    )
    parser.add_argument(
        "--iw1_atom_first_ratio",
        type=float,
        default=None,
        help=(
            "Atom-first scheduling ratio for IW(1). The engine default is 1.0."
        ),
    )
    parser.add_argument(
        "--iw1_incremental_first_applicability",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable IW(1) incremental first-applicability discovery. "
            "Applies to both standard and projective IW."
        ),
    )
    parser.add_argument(
        "--iw1_incremental_first_applicability_debug_crosscheck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable exact debug cross-checking for IW(1) incremental "
            "first-applicability."
        ),
    )
    parser.add_argument(
        "--iw_relaxed_survivors_only_beam",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable pymimir's relaxed survivors-only beam mode. "
            "This only applies with survivors_only beam novelty."
        ),
    )
    parser.add_argument(
        "--iw_randomize_equal_score_ties",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Randomize equal-score tie ordering in IW beam selection.",
    )
    parser.add_argument(
        "--iw_equal_score_tie_seed",
        type=int,
        default=None,
        help="Seed used when equal-score IW ties are randomized.",
    )
    parser.add_argument(
        "--iw_num_threads",
        type=int,
        default=None,
        help=(
            "Thread count for IW parallel beam evaluation. "
            "Values greater than 1 activate the parallel beam path when "
            "beam search is enabled."
        ),
    )
    parser.add_argument(
        "--iw_chunk_size",
        type=int,
        default=None,
        help=(
            "Chunk size for IW parallel beam batching. "
            "Values less than or equal to 0 keep the engine default."
        ),
    )
    parser.add_argument(
        "--iw_depth_feature",
        action="store_true",
        help="Optional depth signal for IW tree action nodes.",
    )
    parser.add_argument(
        "--iw_depth_mode",
        choices=("none", "init_scalar", "depth_nodes"),
        default=None,
        help="IW depth encoding mode to use during training.",
    )
    parser.add_argument(
        "--iw_depth_precedes_mode",
        choices=("mlp", "factorized"),
        default=None,
        help="Ordered depth-node relation family to use for depth-node IW.",
    )
    parser.add_argument(
        "--iw_depth_precedes_scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable scan-based factorized ordering for depth-node IW.",
    )
    parser.add_argument(
        "--iw_depth_order_aux_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable auxiliary ranking supervision on depth-node embeddings.",
    )
    parser.add_argument(
        "--iw_depth_order_aux_weight",
        type=float,
        default=0.0,
        help="Weight for the optional depth-order auxiliary loss.",
    )
    parser.add_argument(
        "--rw_return_feature_mode",
        choices=(RW_RETURN_NONE, RW_RETURN_EXACT_FULL, RW_RETURN_EXACT_BLOCKED),
        default=None,
        help="Enable exact k-return structural features for supported IW tree encodings.",
    )
    parser.add_argument(
        "--rw_walk_mode",
        choices=("standard", "non_returning"),
        default=None,
        help="Walk semantics for exact rw-return features.",
    )
    parser.add_argument(
        "--rw_return_k",
        type=int,
        default=None,
        help="Maximum return-probability walk length K for rw-return features.",
    )
    parser.add_argument(
        "--rw_return_include_depth_nodes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include depth nodes in the rw-return kept-node set.",
    )
    parser.add_argument(
        "--rw_return_lazy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a lazy walk operator for rw-return features.",
    )
    parser.add_argument(
        "--rw_return_lazy_beta",
        type=float,
        default=None,
        help="Lazy-walk beta in [0, 1) for rw-return features.",
    )
    parser.add_argument(
        "--rw_return_block_size",
        type=int,
        default=None,
        help="Block size for exact blocked rw-return propagation.",
    )
    parser.add_argument(
        "--rw_return_force_block_threshold",
        type=int,
        default=None,
        help=(
            "Fallback exact_full to exact_blocked when the kept-node count exceeds this threshold. "
            "Use 0 to disable forced fallback."
        ),
    )
    parser.add_argument(
        "--include_lgan_edges",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable LGAN carrier generation for supported tree encodings.",
    )
    parser.add_argument(
        "--lgan_anchor_sources",
        nargs="+",
        default=None,
        help="Anchor-source groups used when LGAN edges are enabled.",
    )
    parser.add_argument(
        "--lgan_tn_edge_pos",
        default=None,
        help="Relation name reserved for LGAN TN edges.",
    )
    parser.add_argument(
        "--lgan_nn_edge_pos",
        default=None,
        help="Relation name reserved for LGAN NN edges.",
    )
    parser.add_argument(
        "--lgan_rr_edge_pos",
        default=None,
        help="Relation name reserved for LGAN RR edges.",
    )
    args = parser.parse_args()
    if args.typed_projection and not (
        args.projective_iw or args.abstracted_iw
    ):
        parser.error("--typed_projection requires --projective_iw or --abstracted_iw.")
    _resolve_iw_depth_cli_args(args)
    _resolve_rw_return_cli_args(args)
    _resolve_lgan_cli_args(args)
    _apply_domain_placeholders(args)
    _validate_cli_arguments(parser, args)
    return args


def _apply_domain_placeholders(args: argparse.Namespace) -> None:
    domain_value = getattr(args, "domain", None)
    if not domain_value:
        return

    for key, value in list(vars(args).items()):
        if isinstance(value, Path):
            replaced = Path(str(value).replace("{domain}", domain_value))
            setattr(args, key, replaced)
        elif isinstance(value, str):
            setattr(args, key, value.replace("{domain}", domain_value))


def _validate_cli_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if (
        args.projective_iw_keep_depth_one_novel is not None
        and not args.projective_iw
    ):
        parser.error(
            "--projective_iw_keep_depth_one_novel requires --projective_iw."
        )
    if args.keep_goal_nonunary_atoms and not args.projective_iw:
        parser.error("--keep_goal_nonunary_atoms requires --projective_iw.")
    if args.base_abstracted_iw and not (
        args.abstracted_iw or args.projective_iw
    ):
        parser.error("--base_abstracted_iw requires --abstracted_iw or --projective_iw.")
    if (
        args.iw1_atom_first_ratio is not None
        and args.iw1_atom_first_ratio <= 0.0
    ):
        parser.error("--iw1_atom_first_ratio must be > 0.")
    if args.validation_strategy in {"static", "generator_curriculum"}:
        if args.validation is None:
            parser.error(
                f"--validation is required for {args.validation_strategy}."
            )
    elif args.validation is None and args.test is None:
        parser.error(
            "test_curriculum requires either --test or --validation "
            "(to infer a sibling test/ directory)."
        )
    if args.validation_frontier_size < 1:
        parser.error("--validation_frontier_size must be at least 1.")
    if not 0.0 <= args.validation_frontier_threshold <= 1.0:
        parser.error("--validation_frontier_threshold must be in [0, 1].")
    if (
        args.validation_num_workers is not None
        and args.validation_num_workers < 0
    ):
        parser.error("--validation_num_workers must be at least 0.")
    try:
        args.validation_problem_timeout_seconds = (
            _parse_validation_problem_timeout_to_seconds(
                args.validation_problem_timeout
            )
            if args.validation_problem_timeout is not None
            else None
        )
    except ValueError as exc:
        parser.error(
            f"--validation_problem_timeout invalid: {exc}"
        )
    if (
        args.iw_depth_mode == "depth_nodes"
        and args.iw_depth_precedes_mode == "mlp"
        and args.iw_depth_precedes_scan
    ):
        parser.error(
            "--iw_depth_precedes_scan is incompatible with "
            "--iw_depth_precedes_mode=mlp."
        )
    if args.iw_depth_order_aux_weight < 0.0:
        parser.error("--iw_depth_order_aux_weight must be >= 0.")
    if args.iw_depth_order_aux_loss and args.iw_depth_mode != "depth_nodes":
        parser.error(
            "--iw_depth_order_aux_loss requires --iw_depth_mode=depth_nodes."
        )


def _infer_dataset_domain_name(args: argparse.Namespace) -> str:
    if args.domain:
        return args.domain
    candidates = [args.train, args.validation, args.test]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.name in {"train", "validation", "test"}:
            return path.parent.name
        if path.parent.name in {"train", "validation", "test"}:
            return path.parent.parent.name
        if path.is_dir():
            return path.name
    return "default"


def _default_test_path(validation_path: Path | None) -> Path | None:
    if validation_path is None:
        return None
    validation_path = Path(validation_path)
    if validation_path.is_dir():
        return validation_path.parent / "test"
    return validation_path.parent.parent / "test"


def _resolve_dynamic_validation_paths(
    args: argparse.Namespace, output_dir: Path
) -> None:
    if args.validation_cache_dir is None:
        args.validation_cache_dir = output_dir / "validation_dynamic"
    else:
        args.validation_cache_dir = Path(args.validation_cache_dir)
    if args.test is None and args.validation_strategy != "static":
        args.test = _default_test_path(args.validation)
    if (
        args.validation_strategy == "generator_curriculum"
        and args.pddl_generators_root is None
    ):
        args.pddl_generators_root = detect_pddl_generators_root()


def _determine_wandb_run_info_path(args: argparse.Namespace) -> Path | None:
    explicit_path = getattr(args, "wandb_run_info", None)
    if explicit_path:
        info_path = Path(explicit_path)
    elif args.output is not None:
        info_path = Path(args.output) / "wandb_run.json"
    else:
        return None

    return info_path


def _persist_wandb_run_info(
    run: wandb.sdk.wandb_run.Run | None, args: argparse.Namespace
) -> None:
    if run is None:
        return

    info_path = _determine_wandb_run_info_path(args)
    if info_path is None:
        return

    info_path = info_path if info_path.suffix else info_path / "wandb_run.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)

    entity = getattr(run, "entity", None)
    project = getattr(run, "project", None)
    browser_url = None
    if entity and project and run.id:
        browser_url = (
            f"https://wandb.ai/{quote(str(entity), safe='')}/"
            f"{quote(str(project), safe='')}/runs/{quote(str(run.id), safe='')}"
        )

    data = {
        "id": run.id,
        "project": project,
        "entity": entity,
        "name": run.name,
        "url": browser_url or run.url,
        "sdk_url": run.url,
    }

    info_path.write_text(json.dumps(data, indent=2))


def _parse_instances(
    input_: Path,
    *,
    skip_dead_starts: bool = False,
) -> tuple[mm.Domain, list[mm.Problem]]:
    if input_.is_file():
        domain_path = input_.parent / "domain.pddl"
        problem_paths = [str(input_)]
    else:
        domain_path = input_ / "domain.pddl"
        if not domain_path.exists():
            domain_path = input_.parent / "domain.pddl"
        if not domain_path.exists():
            raise RuntimeError(
                f"Cannot find domain file in {input_} or its parent directory."
            )
        domain_path = str(domain_path)
        problem_paths = [
            str(file)
            for file in input_.glob("*.pddl")
            if file.name != "domain.pddl"
        ]
        problem_paths.sort()
    domain = mm.Domain(domain_path)
    problems = [
        mm.Problem(domain, problem_path) for problem_path in problem_paths
    ]
    unsolved_problems = [
        problem
        for problem in problems
        if not problem.get_goal_condition().holds(problem.get_initial_state())
    ]
    if skip_dead_starts:
        dead_start_problems = [
            problem
            for problem in unsolved_problems
            if len(problem.get_initial_state().generate_applicable_actions()) == 0
        ]
        if dead_start_problems:
            logger.warning(
                "Skipping {} dead-start training instances with no applicable actions.",
                len(dead_start_problems),
            )
            unsolved_problems = [
                problem
                for problem in unsolved_problems
                if len(problem.get_initial_state().generate_applicable_actions())
                > 0
            ]
    return domain, unsolved_problems


def _get_optimal_returns(
    problems: list[mm.Problem],
    optimal_model_path: Path,
    device: torch.device,
    discount_factor: float,
    horizon: int,
    reward_function: rl.RewardFunction,
) -> list[float]:
    logger.info("Calculating optimal returns for regret computation.")
    logger.info(f"Loading optimal model from {optimal_model_path}.")
    model = IWModel.load(problems[0].get_domain(), optimal_model_path, device)

    returns = []
    with torch.no_grad():
        greedy_policy = rl.GreedyPolicyTrajectorySampler(model, reward_function)
        initial_state_goals = [
            (problem.get_initial_state(), problem.get_goal_condition())
            for problem in problems
        ]
        trajectories = greedy_policy.sample(initial_state_goals, horizon)
        for trajectory in trajectories:
            total_return = 0.0
            for i, transition in enumerate(trajectory.transitions):
                total_return += (
                    discount_factor**i
                ) * transition.immediate_reward
            returns.append(total_return)
    return returns


def _resolve_aggregation_function(aggregation_name: str):
    if aggregation_name == "smax":
        return rgnn.SmoothMaximumAggregation()
    if aggregation_name == "hmax":
        return rgnn.HardMaximumAggregation()
    if aggregation_name == "mean":
        return rgnn.MeanAggregation()
    if aggregation_name == "add":
        return rgnn.SumAggregation()
    raise RuntimeError(f"Unknown aggregation function: {aggregation_name}.")


def _build_model_components(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
) -> tuple[rgnn.HyperparameterConfig, tuple, rgnn.ModuleConfig]:
    aggregation_function = _resolve_aggregation_function(aggregation_name)
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    input_spec = (
        rgnn.StateEncoder(),
        rgnn.TransitionEffectsEncoder(),
        rgnn.GoalEncoder(),
    )
    module_config = rgnn.ModuleConfig(
        aggregation_function=aggregation_function,
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    return hparam_config, input_spec, module_config


def _create_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
) -> rgnn.RelationalGraphNeuralNetwork:
    hparam_config, input_spec, module_config = _build_model_components(
        domain, embedding_size, num_layers, aggregation_name, binary_embeddings
    )
    output_spec = [("action_scalar", rgnn.ActionScalarDecoder(hparam_config))]
    return rgnn.RelationalGraphNeuralNetwork(
        hparam_config, module_config, input_spec, output_spec
    )  # type: ignore


def _create_reward_function(args) -> rl.RewardFunction:
    match args.reward:
        case "constant":
            return rl.ConstantRewardFunction(-1)
        case "action_count":
            return ActionCountRewardFunction()
        case "symmetric_difference":
            return SymmetricDifferenceRewardFunction()
        case "discounted_jumps":
            return DiscountedRewardFunction(args.discount_factor)
    raise RuntimeError(f"Unknown reward function: {args.reward}.")


def _build_validation_criterias(
    problems: list[mm.Problem],
    args: argparse.Namespace,
    device: torch.device,
    reward_function: rl.RewardFunction,
) -> list[rl.EvaluationCriteria]:
    criterias: list[rl.EvaluationCriteria] = [
        rl.CoverageCriteria(),
        rl.LengthCriteria(),
        rl.TDErrorCriteria(),
    ]
    if args.optimal_model and problems:
        optimal_returns = _get_optimal_returns(
            problems,
            args.optimal_model,
            device,
            args.discount_factor,
            args.validation_horizon,
            reward_function,
        )
        criterias.append(
            RegretCriteria(
                problems,
                optimal_returns,
                args.discount_factor,
            )
        )
    return criterias


@dataclass
class ValidationRunSummary:
    best: bool
    evaluation: list[float | int]
    total_validation: int
    validation_total: int
    validation_solved_count: int
    test_total: int
    test_solved_count: int
    test_evaluated: bool
    solved_by_path: dict[str, bool]
    timed_out_by_path: dict[str, bool] = field(default_factory=dict)
    timed_out_count: int = 0


def _parse_validation_problem_timeout_to_seconds(
    value: str,
) -> int | None:
    text = value.strip()
    if not text:
        return None
    match = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", text)
    if match is None:
        raise ValueError(
            "validation timeout must use HH:MM:SS format"
        )
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(
            "validation timeout must use HH:MM:SS format"
        )
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if total_seconds < 1:
        raise ValueError("validation timeout must be at least one second")
    return total_seconds


def _resolve_problem_domain_path(problem_path: Path) -> Path:
    domain_path = problem_path.parent / "domain.pddl"
    if not domain_path.exists():
        domain_path = problem_path.parent.parent / "domain.pddl"
    if not domain_path.exists():
        raise RuntimeError(
            f"Cannot find domain file for problem path {problem_path}."
        )
    return domain_path


def _select_validation_worker_device(
    device_type: str, device_index: int | None
) -> torch.device:
    if device_type != "cuda":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for validation worker, but torch.cuda.is_available() is False."
        )
    index = 0 if device_index is None else device_index
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def _initialize_validation_worker_device(
    device_type: str, device_index: int | None
) -> None:
    _select_validation_worker_device(device_type, device_index)


def _flush_validation_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        gc.collect()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _create_validation_worker_reward_function(
    reward_name: str, discount_factor: float
) -> rl.RewardFunction:
    return _create_reward_function(
        SimpleNamespace(
            reward=reward_name,
            discount_factor=discount_factor,
        )
    )


def _compute_validation_td_error(trajectory: rl.Trajectory) -> float:
    if len(trajectory) == 0:
        return 0.0
    total_error = 0.0
    for idx in range(len(trajectory) - 1):
        curr = trajectory[idx]
        succ = trajectory[idx + 1]
        q_curr = curr.predicted_q_value
        q_succ = succ.predicted_q_value + curr.immediate_reward
        total_error += abs(q_succ - q_curr)
    total_error += abs(trajectory[-1].predicted_q_value)
    return total_error


def _compute_validation_discounted_return(
    trajectory: rl.Trajectory, discount_factor: float
) -> float:
    total_return = 0.0
    for idx, transition in enumerate(trajectory.transitions):
        total_return += (discount_factor**idx) * transition.immediate_reward
    return total_return


def _timed_out_validation_problem_result(
    problem_path_text: str,
) -> dict[str, object]:
    return {
        "path": str(Path(problem_path_text).resolve()),
        "solved": False,
        "length": 0,
        "td_error": 0.0,
        "discounted_return": 0.0,
        "timed_out": True,
    }


ValidationProblemTask = tuple[
    str,
    str,
    str,
    float,
    int,
    str,
    int | None,
    int | None,
]


def _validation_problem_subprocess_entry(
    send_conn,
    worker_task: Callable[[ValidationProblemTask], dict[str, object]],
    task: ValidationProblemTask,
) -> None:
    try:
        send_conn.send({"ok": True, "result": worker_task(task)})
    except BaseException as exc:
        send_conn.send({"ok": False, "error": repr(exc)})
    finally:
        send_conn.close()


def _terminate_validation_problem_process(process: mp.Process) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join()


def _evaluate_validation_problem_task(
    task: ValidationProblemTask,
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
    device = _select_validation_worker_device(device_type, device_index)
    problem: mm.Problem | None = None
    model: IWModel | None = None
    try:
        domain_path = _resolve_problem_domain_path(problem_path)
        domain = mm.Domain(str(domain_path))
        problem = mm.Problem(domain, str(problem_path))
        reward_function = _create_validation_worker_reward_function(
            reward_name,
            discount_factor,
        )
        model = IWModel.load(domain, model_snapshot_path, device)
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
            "td_error": _compute_validation_td_error(trajectory),
            "discounted_return": _compute_validation_discounted_return(
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
        clear_search_context_caches()
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        del model
        del problem
        gc.collect()


def _aggregate_validation_worker_results(
    records: list[ProblemRecord],
    worker_results: list[dict[str, object]],
    criterias: list[rl.EvaluationCriteria],
    total_validation: int | None = None,
) -> ValidationEvaluationResult:
    solved_by_path: dict[str, bool] = {}
    timed_out_by_path: dict[str, bool] = {}
    solved_flags: list[bool] = []
    lengths: list[int] = []
    td_errors: list[float] = []
    discounted_returns: list[float] = []
    for record, worker_result in zip(records, worker_results, strict=True):
        record_path = str(record.path.resolve())
        worker_path = str(worker_result["path"])
        assert worker_path == record_path, (
            f"Worker returned path {worker_path}, expected {record_path}."
        )
        solved = bool(worker_result["solved"])
        timed_out = bool(worker_result.get("timed_out", False))
        solved_by_path[record_path] = solved
        timed_out_by_path[record_path] = timed_out
        solved_flags.append(solved)
        lengths.append(int(worker_result["length"]))
        td_errors.append(float(worker_result["td_error"]))
        discounted_returns.append(float(worker_result["discounted_return"]))

    evaluation: list[float | int] = []
    for criteria in criterias:
        if isinstance(criteria, rl.CoverageCriteria):
            evaluation.append(sum(1 for solved in solved_flags if solved))
        elif isinstance(criteria, rl.LengthCriteria):
            evaluation.append(
                sum(
                    length if (not criteria.only_solutions or solved) else 0
                    for length, solved in zip(
                        lengths, solved_flags, strict=True
                    )
                )
            )
        elif isinstance(criteria, rl.TDErrorCriteria):
            total_error = math.fsum(
                td_error if (not criteria.only_solutions or solved) else 0.0
                for td_error, solved in zip(
                    td_errors, solved_flags, strict=True
                )
            )
            evaluation.append(round(total_error))
        elif isinstance(criteria, RegretCriteria):
            assert len(criteria.optimal_returns) == len(discounted_returns)
            total_regret = math.fsum(
                optimal_return - discounted_return
                for optimal_return, discounted_return in zip(
                    criteria.optimal_returns, discounted_returns, strict=True
                )
            )
            evaluation.append(round(total_regret))
        else:
            raise RuntimeError(
                "Parallel validation evaluator only supports the built-in "
                "coverage, length, TD error, and regret criteria."
            )
    return ValidationEvaluationResult(
        evaluation=evaluation,
        solved_by_path=solved_by_path,
        total_validation=len(records)
        if total_validation is None
        else total_validation,
        validation_total=len(records)
        if total_validation is None
        else total_validation,
        validation_solved_count=sum(1 for solved in solved_flags if solved),
        test_total=0,
        test_solved_count=0,
        test_evaluated=False,
        timed_out_by_path=timed_out_by_path,
        timed_out_count=sum(1 for timed_out in timed_out_by_path.values() if timed_out),
    )


@dataclass
class _ValidationTaskProcessState:
    index: int
    task: ValidationProblemTask
    process: Any
    recv_conn: Any
    started_at: float


def _run_validation_tasks_in_subprocesses(
    tasks: list[ValidationProblemTask],
    worker_count: int,
    worker_task: Callable[[ValidationProblemTask], dict[str, object]],
    timeout_seconds: int | None,
) -> list[dict[str, object]]:
    if worker_count < 1:
        raise ValueError("Validation subprocess runner requires at least one worker.")
    if not tasks:
        return []

    ctx = mp.get_context("spawn")
    results: list[dict[str, object] | None] = [None] * len(tasks)
    active: list[_ValidationTaskProcessState] = []
    next_index = 0

    def _start_task(task_index: int) -> None:
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_validation_problem_subprocess_entry,
            args=(send_conn, worker_task, tasks[task_index]),
        )
        process.start()
        send_conn.close()
        active.append(
            _ValidationTaskProcessState(
                index=task_index,
                task=tasks[task_index],
                process=process,
                recv_conn=recv_conn,
                started_at=time.monotonic(),
            )
        )

    try:
        while next_index < len(tasks) or active:
            while next_index < len(tasks) and len(active) < worker_count:
                _start_task(next_index)
                next_index += 1

            progress = False
            for state in list(active):
                state.process.join(timeout=0)
                if state.process.exitcode is not None:
                    payload = state.recv_conn.recv() if state.recv_conn.poll() else None
                    state.recv_conn.close()
                    active.remove(state)
                    progress = True
                    if payload is None:
                        raise RuntimeError(
                            "Validation worker exited without returning a result."
                        )
                    if not bool(payload.get("ok")):
                        raise RuntimeError(
                            f"Validation worker failed: {payload.get('error')}"
                        )
                    result = payload.get("result")
                    if not isinstance(result, dict):
                        raise RuntimeError(
                            "Validation worker returned an invalid result."
                        )
                    results[state.index] = result
                    continue

                if (
                    timeout_seconds is not None
                    and time.monotonic() - state.started_at >= timeout_seconds
                ):
                    _terminate_validation_problem_process(state.process)
                    state.recv_conn.close()
                    active.remove(state)
                    results[state.index] = _timed_out_validation_problem_result(
                        str(state.task[0])
                    )
                    progress = True

            if not progress and active:
                time.sleep(0.01)
    finally:
        for state in active:
            if state.process.is_alive():
                _terminate_validation_problem_process(state.process)
            state.recv_conn.close()

    return [
        result
        for result in results
        if result is not None
    ]


class TimeoutAwareFrontierAwarePolicyEvaluation:
    def __init__(
        self,
        criterias: list[rl.EvaluationCriteria],
        horizon: int,
        reward_name: str,
        discount_factor: float,
        device: torch.device,
        model_snapshot_path: Path,
        validation_problem_timeout_seconds: int,
        worker_task: Callable[[tuple[object, ...]], dict[str, object]] = _evaluate_validation_problem_task,
        pack_size: int = 3,
    ) -> None:
        self.criterias = list(criterias)
        self.horizon = horizon
        self.reward_name = reward_name
        self.discount_factor = discount_factor
        self.device_type = device.type
        self.device_index = device.index
        self.model_snapshot_path = model_snapshot_path
        self.validation_problem_timeout_seconds = (
            validation_problem_timeout_seconds
        )
        self.worker_task = worker_task
        self.pack_size = pack_size

    def evaluate(
        self, records: list[ProblemRecord]
    ) -> ValidationEvaluationResult:
        if not records:
            return ValidationEvaluationResult([], {}, 0, 0, 0, 0, 0, False)
        if self.pack_size < 1:
            raise ValueError(
                "TimeoutAwareFrontierAwarePolicyEvaluation requires pack_size >= 1."
            )

        ordered_records = sorted(records, key=_validation_pack_key)
        evaluated_records: list[ProblemRecord] = []
        evaluated_worker_results: list[dict[str, object]] = []
        for idx in range(0, len(ordered_records), self.pack_size):
            pack = ordered_records[idx : idx + self.pack_size]
            tasks = [
                (
                    str(record.path.resolve()),
                    str(self.model_snapshot_path),
                    self.reward_name,
                    self.discount_factor,
                    self.horizon,
                    self.device_type,
                    self.device_index,
                    self.validation_problem_timeout_seconds,
                )
                for record in pack
            ]
            worker_results = _run_validation_tasks_in_subprocesses(
                tasks,
                worker_count=1,
                worker_task=self.worker_task,
                timeout_seconds=self.validation_problem_timeout_seconds,
            )
            evaluated_records.extend(pack)
            evaluated_worker_results.extend(worker_results)
            if any(
                not bool(worker_result["solved"])
                for worker_result in worker_results
            ):
                break
        return _aggregate_validation_worker_results(
            evaluated_records,
            evaluated_worker_results,
            self.criterias,
            total_validation=len(records),
        )


class ParallelFrontierAwarePolicyEvaluation:
    def __init__(
        self,
        criterias: list[rl.EvaluationCriteria],
        horizon: int,
        reward_name: str,
        discount_factor: float,
        device: torch.device,
        num_workers: int | None = None,
        validation_problem_timeout_seconds: int | None = None,
        worker_task: Callable[[tuple[object, ...]], dict[str, object]] = _evaluate_validation_problem_task,
        pack_size: int = 3,
    ) -> None:
        self.criterias = list(criterias)
        self.horizon = horizon
        self.reward_name = reward_name
        self.discount_factor = discount_factor
        self.device_type = device.type
        self.device_index = device.index
        self.num_workers = num_workers
        self.validation_problem_timeout_seconds = (
            validation_problem_timeout_seconds
        )
        self.worker_task = worker_task
        self.pack_size = pack_size
        self.model_snapshot_path: Path | None = None
        self._pool = None
        self._pool_worker_count: int | None = None

    def set_model_snapshot_path(self, model_snapshot_path: Path) -> None:
        self.model_snapshot_path = model_snapshot_path

    def close(self) -> None:
        pool = self._pool
        if pool is None:
            return
        self._pool = None
        self._pool_worker_count = None
        try:
            pool.close()
        finally:
            pool.join()

    def _resolve_worker_count(self, record_count: int) -> int:
        if self.num_workers is None:
            return record_count
        if self.num_workers < 0:
            raise ValueError(
                "ParallelFrontierAwarePolicyEvaluation requires a non-negative worker count."
            )
        return self.num_workers

    def _get_pool(self, worker_count: int):
        if worker_count < 1:
            raise ValueError(
                "ParallelFrontierAwarePolicyEvaluation requires at least one worker."
            )
        if self._pool is not None and self._pool_worker_count == worker_count:
            return self._pool
        self.close()
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=worker_count,
            initializer=_initialize_validation_worker_device,
            initargs=(self.device_type, self.device_index),
        )
        self._pool_worker_count = worker_count
        return self._pool

    def evaluate(
        self, records: list[ProblemRecord]
    ) -> ValidationEvaluationResult:
        if not records:
            return ValidationEvaluationResult([], {}, 0, 0, 0, 0, 0, False)
        if self.model_snapshot_path is None:
            raise RuntimeError(
                "ParallelFrontierAwarePolicyEvaluation requires a model snapshot path before evaluate()."
            )
        if self.pack_size < 1:
            raise ValueError(
                "ParallelFrontierAwarePolicyEvaluation requires pack_size >= 1."
            )
        worker_count = self._resolve_worker_count(len(records))
        ordered_records = sorted(records, key=_validation_pack_key)
        evaluated_records: list[ProblemRecord] = []
        evaluated_worker_results: list[dict[str, object]] = []
        pool = None
        if self.validation_problem_timeout_seconds is None:
            pool = self._get_pool(worker_count)
        for idx in range(0, len(ordered_records), self.pack_size):
            pack = ordered_records[idx : idx + self.pack_size]
            tasks = [
                (
                    str(record.path.resolve()),
                    str(self.model_snapshot_path),
                    self.reward_name,
                    self.discount_factor,
                    self.horizon,
                    self.device_type,
                    self.device_index,
                    self.validation_problem_timeout_seconds,
                )
                for record in pack
            ]
            if pool is None:
                worker_results = _run_validation_tasks_in_subprocesses(
                    tasks,
                    worker_count=min(worker_count, len(tasks)),
                    worker_task=self.worker_task,
                    timeout_seconds=self.validation_problem_timeout_seconds,
                )
            else:
                worker_results = list(
                    pool.imap(self.worker_task, tasks, chunksize=1)
                )
            evaluated_records.extend(pack)
            evaluated_worker_results.extend(worker_results)
            if any(
                not bool(worker_result["solved"])
                for worker_result in worker_results
            ):
                break
        return _aggregate_validation_worker_results(
            evaluated_records,
            evaluated_worker_results,
            self.criterias,
            total_validation=len(records),
        )


class DQNOptimizationContext:
    def __init__(
        self,
        train_problems: list[mm.Problem],
        validation_problems: list[mm.Problem],
        model: IWModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        args: argparse.Namespace,
        validation_curriculum: ValidationCurriculumManager | None = None,
        experiment_run=None,
    ) -> None:
        self.episode = 1
        self.device = device
        self.train_problems = train_problems
        self.validation_problems = validation_problems
        self.validation_curriculum = validation_curriculum
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.experiment_run = experiment_run
        self.dynamic_best_evaluation: list[float | int] | None = None
        self.dynamic_best_total_validation: int = 0
        self.reward_name = args.reward
        self.discount_factor = args.discount_factor
        self.validation_num_workers = args.validation_num_workers
        self.validation_parallel_evaluator: (
            ParallelFrontierAwarePolicyEvaluation | None
        ) = None
        if args.iw_depth_order_aux_loss:
            self.loss_function = DepthOrderAuxDQNOptimization(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.model,
                args.discount_factor,
                args.mellowmax_factor,
                True,
                args.iw_depth_order_aux_weight,
            )
        else:
            self.loss_function = rl.DQNOptimization(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.model,
                args.discount_factor,
                args.mellowmax_factor,
                True,
            )
        self.reward_function = _create_reward_function(args)
        self.original_replay_buffer = rl.PrioritizedReplayBuffer(
            args.max_buffer_size
        )
        self.hindsight_replay_buffer = rl.PrioritizedReplayBuffer(
            args.max_buffer_size
        )
        self.trajectory_sampler = rl.BoltzmannTrajectorySampler(
            self.model, self.reward_function, args.bt_initial
        )
        self.problem_sampler = rl.UniformProblemSampler()
        self.initial_state_sampler = rl.OriginalInitialStateSampler()
        self.goal_sampler = rl.OriginalGoalConditionSampler()
        if args.hindsight == "lifted":
            self.trajectory_refiner = (
                rl.LiftedHindsightTrajectoryRefiner(
                    self.train_problems,
                    args.max_new_trajectories,
                )
            )
        elif args.hindsight == "propositional":
            self.trajectory_refiner = (
                rl.PropositionalHindsightTrajectoryRefiner(
                    self.train_problems,
                    args.max_new_trajectories,
                )
            )
        elif args.hindsight == "state":
            self.trajectory_refiner = rl.StateHindsightTrajectoryRefiner(
                args.max_new_trajectories
            )
        elif args.hindsight == "partial_state":
            self.trajectory_refiner = rl.PartialStateHindsightTrajectoryRefiner(
                args.max_new_trajectories
            )
        elif args.hindsight.lower() == "none":
            self.trajectory_refiner = None
        else:
            raise RuntimeError(f"Unexpected hindsight type: {args.hindsight}")
        self.algorithm = rl.OffPolicyAlgorithm(
            self.train_problems,
            self.loss_function,
            self.reward_function,
            self.original_replay_buffer,
            self.hindsight_replay_buffer,
            self.trajectory_sampler,
            args.train_horizon,
            args.num_rollouts,
            args.batch_size,
            args.train_steps,
            self.problem_sampler,
            self.initial_state_sampler,
            self.goal_sampler,
            self.trajectory_refiner,
        )
        self.evaluation_trajectory_sampler = rl.GreedyPolicyTrajectorySampler(
            self.model, self.reward_function
        )
        # Track best-so-far evaluation tuple and episode for logging/comparison
        self.best_eval_values: list[float] | list[int] | list[object] = []
        self.best_eval_episode: int = 0
        self.best_eval_total_validation: int = 0
        if self.validation_curriculum is None:
            self.evaluation_criteras = _build_validation_criterias(
                self.validation_problems, args, device, self.reward_function
            )
            self.evaluator = rl.SequentialPolicyEvaluation(
                self.validation_problems,
                self.evaluation_criteras,
                self.evaluation_trajectory_sampler,
                args.validation_horizon,
                3,
            )
        else:
            self.evaluation_criteras = []
            self.evaluator = None
            self._rebuild_dynamic_validation_runtime(args)

    def _rebuild_dynamic_validation_runtime(
        self, args: argparse.Namespace, *, reset_best: bool = False
    ) -> None:
        assert self.validation_curriculum is not None
        self.validation_problems = [
            record.problem
            for record in self.validation_curriculum.active_records()
        ]
        self.evaluation_criteras = _build_validation_criterias(
            self.validation_problems, args, self.device, self.reward_function
        )
        if reset_best:
            self.dynamic_best_evaluation = None
            self.dynamic_best_total_validation = 0
            self.best_eval_values = []
            self.best_eval_episode = 0
            self.best_eval_total_validation = 0
        if self.validation_parallel_evaluator is not None:
            self.validation_parallel_evaluator.criterias = list(
                self.evaluation_criteras
            )

    def _get_parallel_validation_evaluator(
        self, args: argparse.Namespace
    ) -> ParallelFrontierAwarePolicyEvaluation:
        assert self.validation_curriculum is not None
        if self.validation_parallel_evaluator is None:
            self.validation_parallel_evaluator = (
                ParallelFrontierAwarePolicyEvaluation(
                    self.evaluation_criteras,
                    args.validation_horizon,
                    self.reward_name,
                    self.discount_factor,
                    self.device,
                    self.validation_num_workers,
                    getattr(args, "validation_problem_timeout_seconds", None),
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
        return self.validation_parallel_evaluator

    def _get_serial_validation_evaluator(
        self,
        args: argparse.Namespace,
        model_snapshot_path: Path | None = None,
    ) -> FrontierAwarePolicyEvaluation | TimeoutAwareFrontierAwarePolicyEvaluation:
        timeout_seconds = getattr(
            args, "validation_problem_timeout_seconds", None
        )
        if timeout_seconds is None:
            return FrontierAwarePolicyEvaluation(
                self.evaluation_criteras,
                self.evaluation_trajectory_sampler,
                args.validation_horizon,
            )
        if model_snapshot_path is None:
            raise RuntimeError(
                "Timeout-aware serial validation requires a model snapshot path."
            )
        return TimeoutAwareFrontierAwarePolicyEvaluation(
            self.evaluation_criteras,
            args.validation_horizon,
            self.reward_name,
            self.discount_factor,
            self.device,
            model_snapshot_path,
            timeout_seconds,
        )

    def evaluate_validation(
        self, args: argparse.Namespace
    ) -> ValidationRunSummary:
        _flush_validation_cuda_cache(self.device)
        dynamic_best_evaluation = getattr(
            self, "dynamic_best_evaluation", None
        )
        dynamic_best_total_validation = getattr(
            self, "dynamic_best_total_validation", 0
        )
        if self.validation_curriculum is None:
            assert self.evaluator is not None
            best, evaluation = self.evaluator.evaluate()
            return ValidationRunSummary(
                best=best,
                evaluation=list(evaluation),
                total_validation=len(self.validation_problems),
                validation_total=len(self.validation_problems),
                validation_solved_count=int(evaluation[0]) if evaluation else 0,
                test_total=0,
                test_solved_count=0,
                test_evaluated=False,
                solved_by_path={},
            )

        if self.validation_num_workers == 0:
            timeout_seconds = getattr(
                args, "validation_problem_timeout_seconds", None
            )
            if timeout_seconds is None:
                serial_evaluator = self._get_serial_validation_evaluator(args)
                result = self.validation_curriculum.evaluate_with_frontier(
                    serial_evaluator
                )
            else:
                with tempfile.TemporaryDirectory(
                    prefix="validation_snapshot_"
                ) as tmp_dir:
                    snapshot_path = Path(tmp_dir) / "model.pt"
                    self.model.save(snapshot_path)
                    serial_evaluator = self._get_serial_validation_evaluator(
                        args,
                        snapshot_path,
                    )
                    result = self.validation_curriculum.evaluate_with_frontier(
                        serial_evaluator
                    )
            best = _is_better_validation_result(
                self.evaluation_criteras,
                result.evaluation,
                result.total_validation,
                dynamic_best_evaluation,
                dynamic_best_total_validation,
                args.validation_horizon,
            )
            if best:
                self.dynamic_best_evaluation = list(result.evaluation)
                self.dynamic_best_total_validation = result.total_validation
            return ValidationRunSummary(
                best=best,
                evaluation=list(result.evaluation),
                total_validation=result.total_validation,
                validation_total=result.validation_total,
                validation_solved_count=result.validation_solved_count,
                test_total=result.test_total,
                test_solved_count=result.test_solved_count,
                test_evaluated=result.test_evaluated,
                solved_by_path=result.solved_by_path,
                timed_out_by_path=result.timed_out_by_path,
                timed_out_count=result.timed_out_count,
            )

        with tempfile.TemporaryDirectory(
            prefix="validation_snapshot_"
        ) as tmp_dir:
            snapshot_path = Path(tmp_dir) / "model.pt"
            self.model.save(snapshot_path)
            evaluator = self._get_parallel_validation_evaluator(args)
            evaluator.set_model_snapshot_path(snapshot_path)
            result = self.validation_curriculum.evaluate_with_frontier(
                evaluator
            )
        best = _is_better_validation_result(
            self.evaluation_criteras,
            result.evaluation,
            result.total_validation,
            dynamic_best_evaluation,
            dynamic_best_total_validation,
            args.validation_horizon,
        )
        if best:
            self.dynamic_best_evaluation = list(result.evaluation)
            self.dynamic_best_total_validation = result.total_validation
        return ValidationRunSummary(
            best=best,
            evaluation=list(result.evaluation),
            total_validation=result.total_validation,
            validation_total=result.validation_total,
            validation_solved_count=result.validation_solved_count,
            test_total=result.test_total,
            test_solved_count=result.test_solved_count,
            test_evaluated=result.test_evaluated,
            solved_by_path=result.solved_by_path,
            timed_out_by_path=result.timed_out_by_path,
            timed_out_count=result.timed_out_count,
        )

    def maybe_expand_validation(
        self, args: argparse.Namespace, solved_by_path: dict[str, bool]
    ) -> CurriculumExpansion | None:
        if self.validation_curriculum is None:
            return None
        expansion = self.validation_curriculum.maybe_expand(solved_by_path)
        if expansion.expanded:
            self._rebuild_dynamic_validation_runtime(args, reset_best=False)
        return expansion

    def reset_validation_search_contexts(self) -> int:
        reset_count = _reset_problem_search_contexts(self.validation_problems)
        if reset_count > 0:
            clear_search_context_caches()
        return reset_count

    def close(self) -> None:
        if self.validation_parallel_evaluator is not None:
            self.validation_parallel_evaluator.close()
            self.validation_parallel_evaluator = None


def _register_printing_hooks(
    context: DQNOptimizationContext, args: argparse.Namespace
) -> None:
    def avg_num_objects(ps: list[mm.Problem]) -> float:
        if len(ps) > 0:
            return sum(len(p.get_objects()) for p in ps) / len(ps)
        else:
            return float("nan")

    def avg_goal_size(ts: list[rl.Trajectory]) -> float:
        if len(ts) > 0:
            return sum(
                len(t[0].goal_condition) for t in ts if len(t) > 0
            ) / len(ts)
        else:
            return float("nan")

    def avg_trajectory_length(ts: list[rl.Trajectory]) -> float:
        if len(ts) > 0:
            return sum(len(t) for t in ts if len(t) > 0) / len(ts)
        else:
            return float("nan")

    sampled_rollout_count = 0
    sampled_solution_rollout_count = 0
    sampled_dead_end_rollout_count = 0
    sampled_solution_transition_count = 0
    sampled_dead_end_transition_count = 0

    def _remember_sampled_rollouts(ts: list[rl.Trajectory]) -> None:
        nonlocal sampled_rollout_count
        nonlocal sampled_solution_rollout_count
        nonlocal sampled_dead_end_rollout_count
        nonlocal sampled_solution_transition_count
        nonlocal sampled_dead_end_transition_count
        sampled_rollout_count = len(ts)
        sampled_solution_rollout_count = sum(
            1 for trajectory in ts if trajectory.is_solution()
        )
        sampled_dead_end_rollout_count = sum(
            1 for trajectory in ts if trajectory.is_unsolvable()
        )
        sampled_solution_transition_count = sum(
            len(trajectory) for trajectory in ts if trajectory.is_solution()
        )
        sampled_dead_end_transition_count = sum(
            len(trajectory) for trajectory in ts if trajectory.is_unsolvable()
        )

    def _reset_sampled_rollouts() -> None:
        _remember_sampled_rollouts([])

    def _log_sampled_trajectories(ts: list[rl.Trajectory]) -> None:
        _remember_sampled_rollouts(ts)
        logger.info(
            f"[{context.episode}] > Sampled Trajectories; {avg_goal_size(ts):.1f} mean goal size; {avg_trajectory_length(ts):.1f} mean trajectory length; {sum([t.is_solution() for t in ts])} / {len(ts)} solutions."
        )

    def _log_refined_trajectories(ts: list[rl.Trajectory]) -> None:
        logger.info(
            f"[{context.episode}] > Refined Trajectories; {avg_goal_size(ts):.1f} mean goal size; {avg_trajectory_length(ts):.1f} mean trajectory length."
        )

    def _log_collected_experience() -> None:
        novel_state_count, ground_action_count = get_collection_action_counts()
        rollout_count = sampled_rollout_count
        rollout_mean = (
            novel_state_count / rollout_count if rollout_count > 0 else 0.0
        )
        ground_action_mean = (
            ground_action_count / rollout_count if rollout_count > 0 else 0.0
        )
        if args.flat:
            logger.info(
                f"[{context.episode}] Collected Experience; {rollout_count} rollouts; {ground_action_count} ground actions total; {ground_action_mean:.2f} ground actions/rollout."
            )
            return
        logger.info(
            f"[{context.episode}] Collected Experience; {rollout_count} rollouts; {novel_state_count} novel IW states total; {rollout_mean:.2f} novel IW states/rollout."
        )
        logger.info(
            f"[{context.episode}] Original Replay Additions [solved | deadend]: {sampled_solution_rollout_count} | {sampled_dead_end_rollout_count} rollouts / {sampled_solution_transition_count} | {sampled_dead_end_transition_count} transitions."
        )

    context.algorithm.register_on_pre_collect_experience(
        reset_collection_action_counts
    )
    context.algorithm.register_on_pre_collect_experience(
        _reset_sampled_rollouts
    )
    context.algorithm.register_on_pre_collect_experience(
        lambda: logger.info("----------")
    )
    context.algorithm.register_on_pre_collect_experience(
        lambda: logger.info(f"[{context.episode}] Collecting Experience.")
    )
    context.algorithm.register_on_sample_problems(
        lambda ps: logger.info(
            f"[{context.episode}] > Sampled Problems; {avg_num_objects(ps):.1f} mean object count."
        )
    )
    context.algorithm.register_on_sample_initial_states(
        lambda x: logger.info(f"[{context.episode}] > Sampled Initial States.")
    )
    context.algorithm.register_on_sample_goal_conditions(
        lambda x: logger.info(f"[{context.episode}] > Sampled Goals.")
    )
    context.algorithm.register_on_sample_trajectories(_log_sampled_trajectories)
    context.algorithm.register_on_refine_trajectories(_log_refined_trajectories)
    context.algorithm.register_on_post_collect_experience(
        _log_collected_experience
    )
    context.algorithm.register_on_pre_optimize_model(
        lambda: logger.info(f"[{context.episode}] Optimizing Model.")
    )

    if args.verbose:
        context.algorithm.register_on_train_step(
            lambda ts, loss: logger.info(
                f"[{context.episode}] > Train step: {loss.mean().item():.5f} mean loss."
            )
        )
    else:
        # Create a closure to aggregate the losses over all train steps.
        agg_loss_sum = 0.0
        agg_loss_num = 0
        agg_total_loss_sum = 0.0
        agg_total_loss_num = 0
        agg_aux_loss_sum = 0.0
        agg_aux_loss_num = 0

        def add_loss(_: list[rl.Transition], losses: torch.Tensor):
            nonlocal agg_loss_sum
            nonlocal agg_loss_num
            nonlocal agg_total_loss_sum
            nonlocal agg_total_loss_num
            nonlocal agg_aux_loss_sum
            nonlocal agg_aux_loss_num
            agg_loss_sum += losses.sum().item()
            agg_loss_num += losses.numel()
            if isinstance(context.loss_function, DepthOrderAuxDQNOptimization):
                agg_total_loss_sum += context.loss_function.last_total_loss
                agg_total_loss_num += 1
                agg_aux_loss_sum += (
                    context.loss_function.last_depth_order_aux_loss
                )
                agg_aux_loss_num += 1

        def print_and_reset_aggregate_loss():
            nonlocal agg_loss_sum
            nonlocal agg_loss_num
            nonlocal agg_total_loss_sum
            nonlocal agg_total_loss_num
            nonlocal agg_aux_loss_sum
            nonlocal agg_aux_loss_num
            if agg_loss_num > 0:
                loss = agg_loss_sum / agg_loss_num
                logger.info(
                    f"[{context.episode}] > Train step: {loss:.5f} mean loss."
                )
                metrics = {
                    "train_loss": loss,
                }
                if agg_total_loss_num > 0:
                    metrics["train_total_loss"] = (
                        agg_total_loss_sum / agg_total_loss_num
                    )
                if agg_aux_loss_num > 0:
                    metrics["train_depth_order_aux_loss"] = (
                        agg_aux_loss_sum / agg_aux_loss_num
                    )
                if (
                    "train_total_loss" in metrics
                    or "train_depth_order_aux_loss" in metrics
                ):
                    logger.info(
                        "[{}] > Train metrics: train_loss={:.5f}; train_total_loss={:.5f}; train_depth_order_aux_loss={:.5f}.",
                        context.episode,
                        metrics["train_loss"],
                        metrics.get("train_total_loss", metrics["train_loss"]),
                        metrics.get("train_depth_order_aux_loss", 0.0),
                    )
                if context.experiment_run:
                    context.experiment_run.log(metrics)
            agg_loss_sum = 0.0
            agg_loss_num = 0
            agg_total_loss_sum = 0.0
            agg_total_loss_num = 0
            agg_aux_loss_sum = 0.0
            agg_aux_loss_num = 0

        context.algorithm.register_on_train_step(add_loss)
        context.algorithm.register_on_post_optimize_model(
            print_and_reset_aggregate_loss
        )

    context.algorithm.register_on_post_optimize_model(
        lambda: logger.info(f"[{context.episode}] Optimized Model.")
    )
    context.algorithm.register_on_post_optimize_model(
        lambda: sys.stdout.flush()
    )


def _prepare_experiment_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def _format_coverage_value(solved: object, total: int) -> str:
    if total <= 0:
        return f"coverage={solved}"
    try:
        solved_count = int(solved)
    except (TypeError, ValueError):
        return f"coverage={solved}"
    return f"coverage={solved_count}/{total} ({solved_count / total:.1%})"


def _compute_total_length_all(
    total_length: object,
    solved: object,
    total: int,
    validation_horizon: int,
) -> int | float | None:
    try:
        solved_count = int(solved)
        base_length = float(total_length)
    except (TypeError, ValueError):
        return None
    unsolved_count = max(total - solved_count, 0)
    augmented_length = base_length + unsolved_count * (validation_horizon + 1)
    return (
        int(augmented_length)
        if augmented_length.is_integer()
        else augmented_length
    )


def _compare_coverage_across_validation_sizes(
    current_solved: object,
    current_total: int,
    best_solved: object,
    best_total: int,
    coverage_criteria: rl.EvaluationCriteria,
) -> int | None:
    try:
        current_solved_count = int(current_solved)
        best_solved_count = int(best_solved)
    except (TypeError, ValueError):
        return None

    if current_total > 0 and best_total > 0 and current_total != best_total:
        lhs = current_solved_count * best_total
        rhs = best_solved_count * current_total
        if lhs != rhs:
            return (lhs > rhs) - (lhs < rhs)
        if current_solved_count != best_solved_count:
            return (current_solved_count > best_solved_count) - (
                current_solved_count < best_solved_count
            )
        return 0

    return int(
        coverage_criteria.compare(current_solved_count, best_solved_count)
    )


def _is_better_validation_result(
    criterias: Sequence[rl.EvaluationCriteria],
    current: Sequence[float | int],
    current_total_validation: int,
    best: Sequence[float | int] | None,
    best_total_validation: int,
    validation_horizon: int,
) -> bool:
    if best is None:
        return True
    if not current:
        return False
    if not best:
        return True
    if not criterias:
        return False

    coverage_comparison = _compare_coverage_across_validation_sizes(
        current[0],
        current_total_validation,
        best[0],
        best_total_validation,
        criterias[0],
    )
    if coverage_comparison is None:
        return compare_evaluations(criterias, current, best)
    if coverage_comparison < 0:
        return False
    if coverage_comparison > 0:
        return True

    if len(current) > 1 and len(best) > 1:
        current_total_length_all = _compute_total_length_all(
            current[1],
            current[0],
            current_total_validation,
            validation_horizon,
        )
        best_total_length_all = _compute_total_length_all(
            best[1],
            best[0],
            best_total_validation,
            validation_horizon,
        )
        if (
            current_total_length_all is not None
            and best_total_length_all is not None
        ):
            if current_total_length_all < best_total_length_all:
                return True
            if current_total_length_all > best_total_length_all:
                return False
        else:
            comparison = int(criterias[1].compare(current[1], best[1]))
            if comparison < 0:
                return False
            if comparison > 0:
                return True

    max_index = min(len(criterias), len(current), len(best))
    for index in range(2, max_index):
        comparison = int(criterias[index].compare(current[index], best[index]))
        if comparison < 0:
            return False
        if comparison > 0:
            return True
    return False


def _prepare_run_paths(
    args: argparse.Namespace, domain_label: str
) -> tuple[Path, Path]:
    if args.output is not None:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = Path(args.runs_root)
        if args.run_group:
            base_dir = base_dir / args.run_group
        domain_dir = base_dir / domain_label
        domain_dir.mkdir(parents=True, exist_ok=True)
        counter = 1
        prefix = args.run_prefix or "run"
        while True:
            candidate = domain_dir / f"{prefix}{counter:03d}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                output_dir = candidate
                break
            except FileExistsError:
                counter += 1
    if args.logfile is not None:
        logfile = Path(args.logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
    else:
        logfile = output_dir / "run.log"
    return output_dir, logfile


def _extract_episode_from_checkpoint_name(path: Path) -> int | None:
    """
    Try to extract the episode number from a checkpoint filename.
    Assumes validation-saved convention like: best_<EPISODE>-<metrics>.pt

    Returns the parsed episode as int or None if it cannot be parsed.
    """
    try:
        name = path.name
        # Typical: best_123-....pt
        m = re.match(r"^best_(\d+)-", name)
        if m:
            return int(m.group(1))
        # Fallback: first integer in the stem, if any
        m2 = re.search(r"(\d+)", path.stem)
        if m2:
            return int(m2.group(1))
    except Exception:
        pass
    return None


def _init_experiment_run(args: argparse.Namespace, domain_name: str | None):
    if args.no_wandb:
        return None

    project_name = (
        args.wandb_project or os.getenv("WANDB_PROJECT") or Path.cwd().name
    )
    entity_name = args.wandb_entity or os.getenv("WANDB_ENTITY")
    default_name = (
        _compose_default_experiment_name(args.domain, args) or project_name
    )
    experiment_name = args.experiment_name or default_name

    config = _prepare_experiment_config(args)
    init_kwargs = {
        "name": experiment_name,
        "project": project_name,
        "config": config,
    }
    if args.output is not None:
        init_kwargs["dir"] = str(Path(args.output))

    if entity_name:
        init_kwargs["entity"] = entity_name

    run = wandb.init(**init_kwargs)
    if run is not None and hasattr(run, "config"):
        run.config.update(config, allow_val_change=True)
        _persist_wandb_run_info(run, args)
        entity = getattr(run, "entity", None)
        project = getattr(run, "project", None)
        if entity and project and run.id:
            browser_url = (
                f"https://wandb.ai/{quote(str(entity), safe='')}/"
                f"{quote(str(project), safe='')}/runs/{quote(str(run.id), safe='')}"
            )
            logger.info(f"Weights & Biases run: {browser_url}")
    return run


def _compose_default_experiment_name(
    domain_name: str | None, args: argparse.Namespace
) -> str:
    parts: list[str] = []

    if isinstance(domain_name, str) and domain_name.strip():
        parts.append(domain_name.strip())

    DEFAULT_VALUES: dict[str, object] = {
        "hindsight": "lifted",
        "aggregation": "smax",
        "reward": "constant",
        "encoding": "concurrent_internal",
        "iw_depth_feature": False,
        "binary_embeddings": False,
        "embedding_size": 32,
        "layers": 10,
    }
    if args.encoding != DEFAULT_VALUES["encoding"]:
        parts.append(args.encoding)
    if args.hindsight != DEFAULT_VALUES["hindsight"]:
        parts.append(args.hindsight)

    if args.aggregation != DEFAULT_VALUES["aggregation"]:
        parts.append(args.aggregation)

    if args.reward != DEFAULT_VALUES["reward"]:
        parts.append(args.reward)

    if args.iw_depth_mode != "none":
        parts.append("iw_depth")

    if args.binary_embeddings != DEFAULT_VALUES["binary_embeddings"]:
        parts.append("binary")

    if args.embedding_size != DEFAULT_VALUES["embedding_size"]:
        parts.append(f"emb{args.embedding_size}")

    if args.layers != DEFAULT_VALUES["layers"]:
        parts.append(f"layers{args.layers}")

    if args.iw_num_threads is not None:
        parts.append(f"iwthreads{args.iw_num_threads}")
    if args.iw1_precheck_add_effect_novelty:
        parts.append("iwaddfx")
    if args.iw1_atom_first_mode:
        parts.append("iwatomfirst")
    if args.iw1_atom_first_ratio is not None:
        parts.append(f"iwatomratio{args.iw1_atom_first_ratio:g}")
    if args.iw1_incremental_first_applicability:
        parts.append("iwincrfirst")
    if args.iw1_incremental_first_applicability_debug_crosscheck:
        parts.append("iwincrxcheck")

    parts.append(f"seed{args.seed}")

    sanitized = [
        re.sub(r"[^A-Za-z0-9._-]+", "-", part) for part in parts if part
    ]
    return "-".join(filter(None, sanitized))


def _train(context: DQNOptimizationContext, args: argparse.Namespace):
    _register_printing_hooks(context, args)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    if len(context.validation_problems) == 0:
        logger.error("No validation problems found; skipping evaluation.")
    logger.info(f"Output folder: {output_folder.resolve()}.")
    while context.episode <= args.num_episodes:
        # Reset per-episode GPU peak memory statistics if running on CUDA
        if (
            hasattr(context, "device")
            and isinstance(context.device, torch.device)
            and context.device.type == "cuda"
        ):
            try:
                torch.cuda.reset_peak_memory_stats(context.device)
            except Exception:
                pass
        # Run RL algorithm.
        context.algorithm.fit()
        # Update Boltzmann temperature.
        bt_ratio = min(1.0, context.episode / args.bt_steps)
        bt_temperature = (
            bt_ratio * args.bt_final + (1.0 - bt_ratio) * args.bt_initial
        )
        context.trajectory_sampler.set_temperature(bt_temperature)
        # Collect GPU memory usage (current and peak) for this episode
        gpu_mem_metrics: dict[str, int] | None = None
        if (
            hasattr(context, "device")
            and isinstance(context.device, torch.device)
            and context.device.type == "cuda"
        ):
            try:
                alloc_cur = torch.cuda.memory_allocated(context.device)
                alloc_peak = torch.cuda.max_memory_allocated(context.device)
                reserv_cur = torch.cuda.memory_reserved(context.device)
                reserv_peak = torch.cuda.max_memory_reserved(context.device)
                to_mib = lambda b: int(b / (1024 * 1024))
                gpu_mem_metrics = {
                    "gpu_mem/allocated_mb": to_mib(alloc_cur),
                    "gpu_mem/allocated_peak_mb": to_mib(alloc_peak),
                    "gpu_mem/reserved_mb": to_mib(reserv_cur),
                    "gpu_mem/reserved_peak_mb": to_mib(reserv_peak),
                }
                logger.info(
                    "[GPU] mem allocated: {}MB (peak {}MB); reserved: {}MB (peak {}MB)",
                    gpu_mem_metrics["gpu_mem/allocated_mb"],
                    gpu_mem_metrics["gpu_mem/allocated_peak_mb"],
                    gpu_mem_metrics["gpu_mem/reserved_mb"],
                    gpu_mem_metrics["gpu_mem/reserved_peak_mb"],
                )
            except Exception:
                pass
        # Evaluate every now and then.
        if args.val_freq > 0 and context.episode % args.val_freq == 0:
            validation_summary = context.evaluate_validation(args)
            best = validation_summary.best
            evaluation = validation_summary.evaluation
            total_validation = validation_summary.total_validation
            # Log detailed evaluation metrics in a labeled form
            try:
                cov = evaluation[0]
                length = evaluation[1] if len(evaluation) > 1 else None
                td_err = evaluation[2] if len(evaluation) > 2 else None
                regret_val = evaluation[3] if len(evaluation) > 3 else None
            except Exception:
                cov, length, td_err, regret_val = None, None, None, None
            total_length_all = _compute_total_length_all(
                length,
                cov,
                total_validation,
                args.validation_horizon,
            )

            msg = f"[{context.episode}] Best: {best}, Evaluation: {evaluation}"
            logger.info(msg)
            validation_coverage = (
                validation_summary.validation_solved_count
                / validation_summary.validation_total
                if validation_summary.validation_total > 0
                else 0.0
            )
            test_coverage = (
                validation_summary.test_solved_count
                / validation_summary.test_total
                if validation_summary.test_total > 0
                else 0.0
            )
            # Also print labeled metrics for quick comparison
            label_parts: list[str] = []
            if cov is not None:
                label_parts.append(
                    _format_coverage_value(cov, total_validation)
                )
            if length is not None:
                label_parts.append(f"length={length}")
            if total_length_all is not None:
                label_parts.append(f"total_length_all={total_length_all}")
            if td_err is not None:
                label_parts.append(f"td_error={td_err}")
            if regret_val is not None:
                label_parts.append(f"regret={regret_val}")
            label_parts.append(
                f"validation={validation_summary.validation_solved_count}/"
                f"{validation_summary.validation_total}"
                f" ({validation_coverage:.1%})"
            )
            if validation_summary.timed_out_count > 0:
                label_parts.append(
                    f"timeouts={validation_summary.timed_out_count}"
                )
            if validation_summary.test_total > 0:
                test_status = (
                    "evaluated" if validation_summary.test_evaluated else "gated"
                )
                label_parts.append(
                    f"test={validation_summary.test_solved_count}/"
                    f"{validation_summary.test_total}"
                    f" ({test_coverage:.1%}, {test_status})"
                )
            if label_parts:
                logger.info(
                    f"[{context.episode}] Eval metrics: "
                    + ", ".join(label_parts)
                )

            # Update best-so-far trackers and report them
            if best:
                context.best_eval_values = list(evaluation)
                context.best_eval_episode = context.episode
                context.best_eval_total_validation = total_validation
            if context.best_eval_episode > 0 and context.best_eval_values:
                be = context.best_eval_values
                best_total_validation = context.best_eval_total_validation
                try:
                    b_cov = be[0] if len(be) > 0 else None
                    b_len = be[1] if len(be) > 1 else None
                    b_td = be[2] if len(be) > 2 else None
                    b_reg = be[3] if len(be) > 3 else None
                except Exception:
                    b_cov, b_len, b_td, b_reg = None, None, None, None
                b_total_length_all = _compute_total_length_all(
                    b_len,
                    b_cov,
                    best_total_validation,
                    args.validation_horizon,
                )
                best_parts: list[str] = []
                if b_cov is not None:
                    best_parts.append(
                        _format_coverage_value(b_cov, best_total_validation)
                    )
                if b_len is not None:
                    best_parts.append(f"length={b_len}")
                if b_total_length_all is not None:
                    best_parts.append(f"total_length_all={b_total_length_all}")
                if b_td is not None:
                    best_parts.append(f"td_error={b_td}")
                if b_reg is not None:
                    best_parts.append(f"regret={b_reg}")
                best_label = (
                    ", ".join(best_parts) if best_parts else f"values={be}"
                )
                logger.info(
                    f"[{context.episode}] Best-so-far (ep {context.best_eval_episode}): "
                    + best_label
                )
            if context.experiment_run:
                total_coverage = (
                    evaluation[0] / total_validation
                    if total_validation > 0
                    else 0.0
                )
                logging_data = {
                    "episode": context.episode,
                    "validation/total_solved": evaluation[0],
                    "validation/total_length": evaluation[1],
                    "validation/total_td_error": evaluation[2],
                    "validation/total_coverage": total_coverage,
                    "validation/timed_out": validation_summary.timed_out_count,
                    "validation/validation_solved": (
                        validation_summary.validation_solved_count
                    ),
                    "validation/validation_coverage": validation_coverage,
                    "validation/test_solved": (
                        validation_summary.test_solved_count
                    ),
                    "validation/test_coverage": test_coverage,
                }
                if total_length_all is not None:
                    logging_data["validation/total_length_all"] = (
                        total_length_all
                    )
                if args.optimal_model:
                    logging_data["regret"] = evaluation[3]
                if gpu_mem_metrics is not None:
                    logging_data.update(gpu_mem_metrics)
                context.experiment_run.log(logging_data)
            if best:
                best_name = f"best_{context.episode}-{'-'.join([str(x) for x in evaluation])}.pt"
                context.model.save(output_folder / best_name)
            expansion = context.maybe_expand_validation(
                args, validation_summary.solved_by_path
            )
            if expansion is not None and expansion.expanded:
                logger.info(
                    "[{}] {}; validation pool {} -> {} problems.",
                    context.episode,
                    expansion.message,
                    expansion.previous_total,
                    expansion.new_total,
                )
            context.model.save(output_folder / "latest.pt")
            validation_reset_count = context.reset_validation_search_contexts()
            if validation_reset_count > 0:
                logger.info(
                    "[{}] Rebuilt search contexts for {} validation problem(s).",
                    context.episode,
                    validation_reset_count,
                )
        # Increment episode.
        logger.info(
            f"Episode: {context.episode} / {args.num_episodes} finished."
        )
        # Also log per-episode GPU memory to wandb if available (even when not validating)
        if context.experiment_run and gpu_mem_metrics is not None:
            try:
                payload = {"episode": context.episode}
                payload.update(gpu_mem_metrics)
                context.experiment_run.log(payload)
            except Exception:
                pass
        context.episode += 1
        sys.stdout.flush()


def _main(args: argparse.Namespace) -> None:
    domain_label = _infer_dataset_domain_name(args)
    auto_output = args.output is None
    auto_log = args.logfile is None
    output_dir, logfile = _prepare_run_paths(args, domain_label)
    if auto_output:
        args.output = output_dir
    else:
        output_dir = Path(args.output)
    if auto_log:
        args.logfile = logfile
    _resolve_dynamic_validation_paths(args, output_dir)
    logfile_path = Path(args.logfile) if args.logfile is not None else logfile
    if args.logfile is not None:
        logger.add(
            str(logfile_path),
            rotation=None,
            format="[{level}; {time:YYYY-MM-DD at HH:mm:ss}; {elapsed}] {message}",
        )
    logger.info("Arguments:")
    for name, value in args.__dict__.items():
        logger.info(f"> {name}: {str(value)}")
    logger.info(f"Torch: {torch.__version__}")
    device = create_device(device=args.device, force_cpu=args.cpu)
    logger.info(f"Using device: {device}.")
    domain, train_problems = _parse_instances(
        args.train, skip_dead_starts=True
    )
    domain_name = domain.get_name()
    logger.info(f"Parsed {len(train_problems)} training instances.")
    validation_curriculum: ValidationCurriculumManager | None = None
    if args.validation_strategy == "static":
        assert args.validation is not None
        _, validation_problems = _parse_instances(args.validation)
        logger.info(f"Parsed {len(validation_problems)} validation instances.")
    else:
        if args.test is None:
            raise RuntimeError(
                "Dynamic validation requires --test or a validation path from "
                "which a sibling test/ directory can be inferred."
            )
        validation_curriculum = ValidationCurriculumManager(
            requested_strategy=args.validation_strategy,
            domain_name=domain_label,
            domain=domain,
            validation_path=args.validation,
            test_path=args.test,
            cache_dir=Path(args.validation_cache_dir),
            frontier_size=args.validation_frontier_size,
            frontier_threshold=args.validation_frontier_threshold,
            seed=args.seed,
            pddl_generators_root=args.pddl_generators_root,
        )
        validation_problems = [
            record.problem for record in validation_curriculum.active_records()
        ]
        logger.info(
            "Initialized {} validation with {} active problems.",
            validation_curriculum.effective_strategy,
            len(validation_problems),
        )
        if validation_curriculum.fallback_reason:
            logger.warning(validation_curriculum.fallback_reason)
    experiment_run = _init_experiment_run(args, domain_name)
    # Create or load model depending on resume flag
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        logger.info(f"Loading model from checkpoint: {ckpt_path}")
        model = IWModel.load(
            domain,
            ckpt_path,
            device,
            projective_iw=True if args.projective_iw else None,
            typed_projection=True if args.typed_projection else None,
            abstracted_iw=True if args.abstracted_iw else None,
            abstracted_iw_width=args.abstracted_iw_width,
            base_abstracted_iw=(
                True
                if args.base_abstracted_iw
                else (not args.typed_projection if args.projective_iw else None)
            ),
            preserve_goal_atoms=True if args.abstracted_iw else None,
            keep_depth_one_novel=args.projective_iw_keep_depth_one_novel,
            keep_goal_nonunary_atoms=(
                True if args.keep_goal_nonunary_atoms else None
            ),
            iw_layer_ordering_strategy=args.iw_layer_ordering_strategy,
            iw_max_next_layer_states=args.iw_max_next_layer_states,
            iw_beam_width=args.iw_beam_width,
            iw_beam_novelty_mode=args.iw_beam_novelty_mode,
            iw_max_depth=args.iw_max_depth,
            iw1_precheck_add_effect_novelty=(
                args.iw1_precheck_add_effect_novelty
            ),
            iw1_atom_first_mode=args.iw1_atom_first_mode,
            iw1_atom_first_ratio=args.iw1_atom_first_ratio,
            iw1_incremental_first_applicability=(
                args.iw1_incremental_first_applicability
            ),
            iw1_incremental_first_applicability_debug_crosscheck=(
                args.iw1_incremental_first_applicability_debug_crosscheck
            ),
            iw_relaxed_survivors_only_beam=(
                args.iw_relaxed_survivors_only_beam
            ),
            iw_randomize_equal_score_ties=args.iw_randomize_equal_score_ties,
            iw_equal_score_tie_seed=args.iw_equal_score_tie_seed,
            iw_num_threads=args.iw_num_threads,
            iw_chunk_size=args.iw_chunk_size,
        )
        if (
            model.iw_depth_feature != args.iw_depth_feature
            or model.iw_depth_mode != args.iw_depth_mode
            or model.iw_depth_precedes_mode != args.iw_depth_precedes_mode
            or model.iw_depth_precedes_scan != args.iw_depth_precedes_scan
        ):
            logger.info(
                "Checkpoint IW depth config overrides requested values: "
                f"checkpoint={model.iw_depth_config}, requested="
                f"(mode={args.iw_depth_mode}, precedes_mode={args.iw_depth_precedes_mode}, precedes_scan={args.iw_depth_precedes_scan})."
            )
            args.iw_depth_feature = model.iw_depth_feature
            args.iw_depth_mode = model.iw_depth_mode
            args.iw_depth_precedes_mode = model.iw_depth_precedes_mode
            args.iw_depth_precedes_scan = model.iw_depth_precedes_scan
        if args.rw_return_override_requested and model.rw_return_config != args.rw_return_config:
            logger.info(
                "Checkpoint RW-return config overrides requested values: "
                f"checkpoint={model.rw_return_config}, requested={args.rw_return_config}."
            )
            args.rw_return_config = model.rw_return_config
            args.rw_return_feature_mode = model.rw_return_feature_mode
            args.rw_return_k = model.rw_return_k
            args.rw_return_include_depth_nodes = (
                model.rw_return_include_depth_nodes
            )
            args.rw_return_lazy = model.rw_return_lazy
            args.rw_return_lazy_beta = model.rw_return_lazy_beta
            args.rw_return_block_size = model.rw_return_block_size
            args.rw_return_force_block_threshold = (
                model.rw_return_force_block_threshold
            )
        if args.lgan_override_requested and model.lgan_config != args.lgan_config:
            logger.info(
                "Checkpoint LGAN config overrides requested values: "
                f"checkpoint={model.lgan_config}, requested={args.lgan_config}."
            )
            args.lgan_config = model.lgan_config
            args.include_lgan_edges = model.include_lgan_edges
            args.lgan_anchor_sources = list(model.lgan_anchor_sources)
            args.lgan_tn_edge_pos = model.lgan_tn_edge_pos
            args.lgan_nn_edge_pos = model.lgan_nn_edge_pos
            args.lgan_rr_edge_pos = model.lgan_rr_edge_pos
        if (
            model.iw_depth_order_aux_loss != args.iw_depth_order_aux_loss
            or model.iw_depth_order_aux_weight
            != args.iw_depth_order_aux_weight
        ):
            logger.info(
                "Checkpoint depth-order aux config overrides requested values: "
                f"loss={model.iw_depth_order_aux_loss}, weight={model.iw_depth_order_aux_weight}."
            )
            args.iw_depth_order_aux_loss = model.iw_depth_order_aux_loss
            args.iw_depth_order_aux_weight = model.iw_depth_order_aux_weight
    else:
        logger.info("Creating model...")
        model = IWModel.new(
            domain,
            args.embedding_size,
            args.layers,
            args.aggregation,
            args.binary_embeddings,
            args.encoding,
            flat=args.flat,
            projective_iw=args.projective_iw,
            typed_projection=args.typed_projection,
            abstracted_iw=args.abstracted_iw or args.projective_iw,
            abstracted_iw_width=args.abstracted_iw_width,
            base_abstracted_iw=(
                args.base_abstracted_iw
                or (args.projective_iw and not args.typed_projection)
            ),
            preserve_goal_atoms=(
                args.keep_goal_nonunary_atoms if args.projective_iw else True
            ),
            keep_depth_one_novel=args.projective_iw_keep_depth_one_novel,
            keep_goal_nonunary_atoms=args.keep_goal_nonunary_atoms,
            iw_layer_ordering_strategy=args.iw_layer_ordering_strategy,
            iw_max_next_layer_states=args.iw_max_next_layer_states,
            iw_beam_width=args.iw_beam_width,
            iw_beam_novelty_mode=args.iw_beam_novelty_mode,
            iw_max_depth=args.iw_max_depth,
            iw1_precheck_add_effect_novelty=(
                False
                if args.iw1_precheck_add_effect_novelty is None
                else args.iw1_precheck_add_effect_novelty
            ),
            iw1_atom_first_mode=(
                False
                if args.iw1_atom_first_mode is None
                else args.iw1_atom_first_mode
            ),
            iw1_atom_first_ratio=(
                1.0
                if args.iw1_atom_first_ratio is None
                else args.iw1_atom_first_ratio
            ),
            iw1_incremental_first_applicability=(
                False
                if args.iw1_incremental_first_applicability is None
                else args.iw1_incremental_first_applicability
            ),
            iw1_incremental_first_applicability_debug_crosscheck=(
                False
                if args.iw1_incremental_first_applicability_debug_crosscheck
                is None
                else args.iw1_incremental_first_applicability_debug_crosscheck
            ),
            iw_relaxed_survivors_only_beam=(
                args.iw_relaxed_survivors_only_beam
            ),
            iw_randomize_equal_score_ties=args.iw_randomize_equal_score_ties,
            iw_equal_score_tie_seed=args.iw_equal_score_tie_seed,
            iw_num_threads=args.iw_num_threads,
            iw_chunk_size=args.iw_chunk_size,
            iw_depth_feature=args.iw_depth_feature,
            iw_depth_mode=args.iw_depth_mode,
            iw_depth_precedes_mode=args.iw_depth_precedes_mode,
            iw_depth_precedes_scan=args.iw_depth_precedes_scan,
            lgan_config=args.lgan_config,
            rw_return_config=args.rw_return_config,
            iw_depth_order_aux_loss=args.iw_depth_order_aux_loss,
            iw_depth_order_aux_weight=args.iw_depth_order_aux_weight,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_initial)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.lr_steps * args.train_steps, args.lr_final
    )
    logger.info("Training model...")
    context = DQNOptimizationContext(
        train_problems,
        validation_problems,
        model,
        optimizer,
        lr_scheduler,
        device,
        args,
        validation_curriculum=validation_curriculum,
        experiment_run=experiment_run,
    )
    try:
        # If resuming from a checkpoint, set start episode and report remaining
        if args.resume_from is not None:
            last_ep = _extract_episode_from_checkpoint_name(
                Path(args.resume_from)
            )
            if last_ep is not None:
                context.episode = max(1, last_ep + 1)
                remaining = max(0, args.num_episodes - last_ep)
                logger.info(
                    f"Continuing from checkpoint '{Path(args.resume_from).name}' starting at episode {context.episode}."
                )
                logger.info(
                    f"Will run {remaining} remaining episode(s) to complete {args.num_episodes} total."
                )
                if remaining <= 0:
                    logger.info(
                        "No remaining episodes to run based on --num_episodes and checkpoint name. Exiting."
                    )
                    return
            else:
                logger.warning(
                    "Could not infer episode from checkpoint filename; starting from episode 1."
                )
        start_time = time.perf_counter()
        _train(context, args)
        elapsed = time.perf_counter() - start_time
        try:
            human = str(timedelta(seconds=int(elapsed)))
        except Exception:
            human = f"{elapsed:.2f}s"
        logger.info(f"Total training runtime: {human} ({elapsed:.2f}s)")
        if experiment_run is not None:
            try:
                experiment_run.log({"total_training_seconds": elapsed})
            except Exception:
                pass
    finally:
        context.close()


def main() -> None:
    assert_native_backend_ready("train startup")
    _main(_parse_arguments())


if __name__ == "__main__":
    main()
