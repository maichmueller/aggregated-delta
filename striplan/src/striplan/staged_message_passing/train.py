from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch

from .. import train as base_train
from .model import SUPPORTED_ENCODINGS, StagedIWModel
from .validation import StagedDQNOptimizationContext
from ..utils import create_device


def _parse_arguments() -> argparse.Namespace:
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument(
        "--world_layers",
        type=int,
        default=None,
        help="Number of world-only message-passing layers.",
    )
    stage_parser.add_argument(
        "--tree_layers",
        type=int,
        default=None,
        help="Number of tree-structure message-passing layers.",
    )
    stage_parser.add_argument(
        "--joint_layers",
        type=int,
        default=None,
        help="Number of final joint message-passing layers.",
    )
    stage_parser.add_argument(
        "--residual_fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse base/world/tree embeddings before the joint stage.",
    )
    stage_args, remaining = stage_parser.parse_known_args()
    previous_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        try:
            args = base_train._parse_arguments()
        except SystemExit as exc:
            if exc.code == 0 and any(
                flag in remaining for flag in ("-h", "--help")
            ):
                print("\nStaged message-passing options:")
                print(stage_parser.format_help().strip())
            raise
    finally:
        sys.argv = previous_argv
    merged = vars(args)
    merged.update(vars(stage_args))
    return argparse.Namespace(**merged)


def _default_stage_layout(total_layers: int) -> tuple[int, int, int]:
    if total_layers <= 1:
        return 0, 0, 1
    if total_layers == 2:
        return 1, 0, 1
    world_layers = max(1, total_layers // 4)
    tree_layers = max(1, total_layers // 4)
    joint_layers = total_layers - world_layers - tree_layers
    if joint_layers < 1:
        if tree_layers > world_layers and tree_layers > 0:
            tree_layers -= 1
        elif world_layers > 0:
            world_layers -= 1
        joint_layers = total_layers - world_layers - tree_layers
    return world_layers, tree_layers, joint_layers


def _resolve_stage_layout(args: argparse.Namespace) -> None:
    stage_values = (
        args.world_layers,
        args.tree_layers,
        args.joint_layers,
    )
    if any(value is not None for value in stage_values) and any(
        value is None for value in stage_values
    ):
        raise RuntimeError(
            "Specify either all of --world_layers/--tree_layers/--joint_layers or none of them."
        )
    if all(value is None for value in stage_values):
        world_layers, tree_layers, joint_layers = _default_stage_layout(
            int(args.layers)
        )
    else:
        world_layers = int(args.world_layers)
        tree_layers = int(args.tree_layers)
        joint_layers = int(args.joint_layers)
    if joint_layers < 1:
        raise RuntimeError("The staged joint tower must have at least one layer.")
    if world_layers < 0 or tree_layers < 0:
        raise RuntimeError("Stage layer counts must be non-negative.")
    args.world_layers = world_layers
    args.tree_layers = tree_layers
    args.joint_layers = joint_layers
    args.layers = world_layers + tree_layers + joint_layers


def _validate_staged_args(args: argparse.Namespace) -> None:
    if args.flat:
        raise RuntimeError(
            "Staged message passing does not support --flat."
        )
    if args.encoding not in SUPPORTED_ENCODINGS:
        raise RuntimeError(
            f"--encoding must be one of {SUPPORTED_ENCODINGS!r} for staged training."
        )


def _main(args: argparse.Namespace) -> None:
    _resolve_stage_layout(args)
    _validate_staged_args(args)
    domain_label = base_train._infer_dataset_domain_name(args)
    auto_output = args.output is None
    auto_log = args.logfile is None
    output_dir, logfile = base_train._prepare_run_paths(args, domain_label)
    if auto_output:
        args.output = output_dir
    else:
        output_dir = Path(args.output)
    if auto_log:
        args.logfile = logfile
    base_train._resolve_dynamic_validation_paths(args, output_dir)
    logfile_path = Path(args.logfile) if args.logfile is not None else logfile
    if args.logfile is not None:
        base_train.logger.add(
            str(logfile_path),
            rotation=None,
            format="[{level}; {time:YYYY-MM-DD at HH:mm:ss}; {elapsed}] {message}",
        )
    base_train.logger.info("Arguments:")
    for name, value in args.__dict__.items():
        base_train.logger.info(f"> {name}: {str(value)}")
    device = create_device(device=args.device, force_cpu=args.cpu)
    base_train.logger.info(f"Torch: {torch.__version__}")
    base_train.logger.info(f"Using device: {device}.")
    domain, train_problems = base_train._parse_instances(
        args.train, skip_dead_starts=True
    )
    domain_name = domain.get_name()
    base_train.logger.info(f"Parsed {len(train_problems)} training instances.")
    validation_curriculum = None
    if args.validation_strategy == "static":
        assert args.validation is not None
        _, validation_problems = base_train._parse_instances(args.validation)
        base_train.logger.info(
            f"Parsed {len(validation_problems)} validation instances."
        )
    else:
        if args.test is None:
            raise RuntimeError(
                "Dynamic validation requires --test or a validation path from which a sibling test/ directory can be inferred."
            )
        validation_curriculum = base_train.ValidationCurriculumManager(
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
            record.problem
            for record in validation_curriculum.active_records()
        ]
        base_train.logger.info(
            "Initialized {} validation with {} active problems.",
            validation_curriculum.effective_strategy,
            len(validation_problems),
        )
        if validation_curriculum.fallback_reason:
            base_train.logger.warning(validation_curriculum.fallback_reason)
    experiment_run = base_train._init_experiment_run(args, domain_name)
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        base_train.logger.info(f"Loading model from checkpoint: {ckpt_path}")
        model = StagedIWModel.load(
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
        if model.iw_depth_feature != args.iw_depth_feature:
            base_train.logger.info(
                f"Checkpoint iw_depth_feature={model.iw_depth_feature} overrides requested value {args.iw_depth_feature}."
            )
            args.iw_depth_feature = model.iw_depth_feature
        args.world_layers = model.stage_layout.world_layers
        args.tree_layers = model.stage_layout.tree_layers
        args.joint_layers = model.stage_layout.joint_layers
        args.residual_fusion = model.stage_layout.residual_fusion
        args.layers = model.stage_layout.total_layers
    else:
        base_train.logger.info("Creating staged model...")
        model = StagedIWModel.new(
            domain,
            args.embedding_size,
            args.aggregation,
            args.binary_embeddings,
            args.encoding,
            world_layers=args.world_layers,
            tree_layers=args.tree_layers,
            joint_layers=args.joint_layers,
            residual_fusion=args.residual_fusion,
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
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_initial)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.lr_steps * args.train_steps, args.lr_final
    )
    base_train.logger.info(
        "Stage layout: world={} tree={} joint={} residual_fusion={}.",
        args.world_layers,
        args.tree_layers,
        args.joint_layers,
        args.residual_fusion,
    )
    base_train.logger.info("Training model...")
    context = StagedDQNOptimizationContext(
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
        if args.resume_from is not None:
            last_ep = base_train._extract_episode_from_checkpoint_name(
                Path(args.resume_from)
            )
            if last_ep is not None:
                context.episode = max(1, last_ep + 1)
                remaining = max(0, args.num_episodes - last_ep)
                base_train.logger.info(
                    f"Continuing from checkpoint '{Path(args.resume_from).name}' starting at episode {context.episode}."
                )
                base_train.logger.info(
                    f"Will run {remaining} remaining episode(s) to complete {args.num_episodes} total."
                )
                if remaining <= 0:
                    base_train.logger.info(
                        "No remaining episodes to run based on --num_episodes and checkpoint name. Exiting."
                    )
                    return
            else:
                base_train.logger.warning(
                    "Could not infer episode from checkpoint filename; starting from episode 1."
                )
        start_time = time.perf_counter()
        base_train._train(context, args)
        elapsed = time.perf_counter() - start_time
        human = str(timedelta(seconds=int(elapsed)))
        base_train.logger.info(
            f"Total training runtime: {human} ({elapsed:.2f}s)"
        )
        if experiment_run is not None:
            try:
                experiment_run.log({"total_training_seconds": elapsed})
            except Exception:
                pass
    finally:
        context.close()


if __name__ == "__main__":
    _main(_parse_arguments())
