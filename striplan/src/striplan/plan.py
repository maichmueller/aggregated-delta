import argparse
import atexit
import gc
import itertools
import json
import multiprocessing as mp
import os
import re
import signal
import time
import warnings
import pymimir as mm
import pymimir_rl as rl
import torch
import wandb

from .iw import IWGroundAction
from .iw_depth import cli_requested_iw_depth_override, resolve_iw_depth_config
from .lgan import cli_requested_lgan_override, resolve_lgan_config
from .rw_return import (
    RW_RETURN_EXACT_BLOCKED,
    RW_RETURN_EXACT_FULL,
    RW_RETURN_NONE,
    cli_requested_rw_return_override,
    resolve_rw_return_config,
)
from pathlib import Path

from .model import IWModel, clear_search_context_caches
from .search_context import reset_problem_search_context
from .utils import create_device

from statistics import mean
from typing import Any, Optional

from tqdm import tqdm


PLAN_HORIZON = 1000
SUMMARY_PLAN_LINE_THRESHOLD = 200
SUMMARY_SNIPPET_LINES = 20


# Ensure we never use 'fork' with CUDA. 'spawn' gives each worker a fresh
# interpreter so CUDA/cuBLAS can initialize cleanly.
mp.set_start_method("spawn", force=True)

_WORKER_CONTEXT: dict[str, Any] | None = None
_CURRENT_PROBLEM_SUMMARY_PATH: Optional[Path] = None
_CURRENT_PROBLEM_STR: Optional[str] = None
_CURRENT_TRACE_PATH: Optional[Path] = None
_INTERRUPT_HANDLED = False
_LIVE_TRACE_TO_STDOUT = True

# Suppress noisy beta warning from PyTorch index_reduce used by pymimir_rgnn
warnings.filterwarnings(
    "ignore",
    message=r".*index_reduce\(\) is in beta.*",
    category=UserWarning,
    module=r"pymimir_rgnn\.aggregations",
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


def _write_minimal_summary(
    summary_path: Path, problem_str: str, note: str
) -> None:
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"Problem: {problem_str}",
            "Solved: no",
            note,
            "",
        ]
        summary_path.write_text("\n".join(lines))
    except Exception:
        pass


def _write_trace_header(
    trace_path: Path, problem_str: str, search_strategy: str, horizon: int
) -> None:
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"Problem: {problem_str}",
            "Artifact-Type: search-trace",
            "Status: running",
            (
                "Warning: this is a live attempted trace, not a validated final solution. "
                "Use the .plan file as the authoritative final artifact."
            ),
            f"Search: {search_strategy}",
            f"Horizon: {horizon}",
            "",
        ]
        with trace_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        pass


def _append_trace_line(trace_path: Optional[Path], line: str = "") -> None:
    if trace_path is None:
        return
    try:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        pass


def _emit_live_trace(
    trace_path: Optional[Path], trace_label: str, line: str = ""
) -> None:
    _append_trace_line(trace_path, line)
    if not line or not _LIVE_TRACE_TO_STDOUT:
        return
    print(f"[trace {trace_label}] {line}", flush=True)


def _on_interrupt(signum, frame):  # type: ignore[no-untyped-def]
    global _CURRENT_PROBLEM_SUMMARY_PATH, _CURRENT_PROBLEM_STR, _CURRENT_TRACE_PATH
    global _INTERRUPT_HANDLED

    if _INTERRUPT_HANDLED:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return

    _INTERRUPT_HANDLED = True

    if _CURRENT_PROBLEM_SUMMARY_PATH and _CURRENT_PROBLEM_STR:
        _write_minimal_summary(
            _CURRENT_PROBLEM_SUMMARY_PATH,
            _CURRENT_PROBLEM_STR,
            f"Note: terminated by signal {signum}; run did not conclude.",
        )
    if _CURRENT_TRACE_PATH is not None:
        trace_label = (
            Path(_CURRENT_PROBLEM_STR).stem if _CURRENT_PROBLEM_STR else "trace"
        )
        _emit_live_trace(
            _CURRENT_TRACE_PATH,
            trace_label,
            f"Final-Status: interrupted by signal {signum}",
        )
        _emit_live_trace(
            _CURRENT_TRACE_PATH,
            trace_label,
            "Trace-Complete: no",
        )

    raise KeyboardInterrupt


def _on_exit():
    # If exiting without having produced a full summary for the current problem,
    # leave a minimal artifact indicating interruption.
    global _CURRENT_PROBLEM_SUMMARY_PATH, _CURRENT_PROBLEM_STR, _CURRENT_TRACE_PATH
    global _INTERRUPT_HANDLED
    if _INTERRUPT_HANDLED:
        return
    if _CURRENT_PROBLEM_SUMMARY_PATH and _CURRENT_PROBLEM_STR:
        _write_minimal_summary(
            _CURRENT_PROBLEM_SUMMARY_PATH,
            _CURRENT_PROBLEM_STR,
            "Note: process exited before conclusion (atexit).",
        )
    if _CURRENT_TRACE_PATH is not None:
        trace_label = (
            Path(_CURRENT_PROBLEM_STR).stem if _CURRENT_PROBLEM_STR else "trace"
        )
        _emit_live_trace(
            _CURRENT_TRACE_PATH,
            trace_label,
            "Final-Status: process exited before conclusion",
        )
        _emit_live_trace(_CURRENT_TRACE_PATH, trace_label, "Trace-Complete: no")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Settings for testing")
    parser.add_argument(
        "--domain", required=True, type=Path, help="Path to the domain file"
    )
    parser.add_argument(
        "--problem",
        required=True,
        type=Path,
        help="Path to a problem file or a directory of problem files",
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="Path to a pre-trained model"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU to be used."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Preferred CUDA device ID (if multiple GPUs are available).",
    )
    parser.add_argument(
        "--device-strategy",
        "--device_strategy",
        dest="device_strategy",
        choices=("sticky", "cycle"),
        default="sticky",
        help=(
            "How workers select GPUs when multiple are visible: 'sticky' pins each worker to a single device "
            "for its lifetime (balanced via shared queue); 'cycle' rotates device per problem (may open contexts on multiple GPUs)."
        ),
    )
    parser.add_argument(
        "--plan_output_dir",
        type=Path,
        default=None,
        help="Directory to store the generated plan for each problem (one file per problem).",
    )
    parser.add_argument(
        "--flat_plans",
        action="store_true",
        help="If set, write plan files directly into --plan_output_dir without model subfolders, using .plan extension.",
    )
    parser.add_argument(
        "--summary_output_dir",
        type=Path,
        default=None,
        help="Directory to store the per-problem summary results (one file per problem).",
    )
    parser.add_argument(
        "--wandb_run_info",
        type=Path,
        default=None,
        help="Path to Weights & Biases run metadata (defaults to <output>/wandb_run.json if present).",
    )
    parser.add_argument(
        "--num_workers",
        "--nr_workers",
        "--nr_worker",
        dest="num_workers",
        type=int,
        default=0,
        help="Number of worker processes to use (-1 uses all available cores).",
    )
    parser.add_argument(
        "--plan_horizon",
        type=int,
        default=PLAN_HORIZON,
        help=(
            "Maximum number of choices to consider per problem. For greedy search "
            "this is the trajectory horizon; for A* it is max_num_states."
        ),
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help=(
            "Re-run problems recorded as unsolved, but skip ones that previously exceeded the planning horizon."
        ),
    )
    parser.add_argument(
        "--retry_all_failed",
        action="store_true",
        help=(
            "Re-run all problems recorded as unsolved, including those that exceeded the planning horizon."
        ),
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help=(
            "Resume an interrupted run by continuing any problems that "
            "were not processed previously."
        ),
    )
    parser.add_argument(
        "--verbosity",
        choices=("full", "summary", "quiet"),
        default="summary",
        help=(
            "Control per-problem console output: 'full' prints entire plans, "
            "'summary' truncates long plans (default), 'quiet' suppresses plan text."
        ),
    )
    parser.add_argument(
        "--search",
        choices=("greedy", "astar"),
        default="greedy",
        help=(
            "Search strategy to use: 'greedy' uses the policy sampler (default), "
            "'astar' runs A* with the model Q-values as heuristic."
        ),
    )
    parser.add_argument(
        "--closed-set-greedy",
        action="store_true",
        help=(
            "When using greedy search, avoid selecting actions whose successor "
            "state has already been seen in the current rollout."
        ),
    )
    parser.add_argument(
        "--search-context-reset-interval",
        "--search_context_reset_interval",
        dest="search_context_reset_interval",
        type=int,
        default=0,
        help=(
            "For greedy planning only, rebuild the current problem search "
            "context and clear cached search state after every N macro "
            "choices (0 disables)."
        ),
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="If set, will run the models in flat mode, which means that no IW is performed and only the applicable actions are added in the GNN",
    )
    parser.add_argument(
        "--iw1_precheck_add_effect_novelty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override checkpoint IW setting for add-effect novelty prechecks "
            "during planning."
        ),
    )
    parser.add_argument(
        "--iw1_atom_first_mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override checkpoint IW setting for atom-first successor ordering "
            "during planning."
        ),
    )
    parser.add_argument(
        "--iw1_atom_first_ratio",
        type=float,
        default=None,
        help="Override checkpoint IW atom-first ratio during planning.",
    )
    parser.add_argument(
        "--iw1_incremental_first_applicability",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override checkpoint IW setting for incremental first-applicability "
            "during planning."
        ),
    )
    parser.add_argument(
        "--iw1_incremental_first_applicability_debug_crosscheck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override checkpoint IW setting for incremental first-applicability "
            "debug cross-checking during planning."
        ),
    )
    parser.add_argument(
        "--iw_depth_feature",
        action="store_true",
        help=(
            "Expect the checkpoint to use the legacy scalar IW depth feature."
        ),
    )
    parser.add_argument(
        "--iw_depth_mode",
        choices=("none", "init_scalar", "depth_nodes"),
        default=None,
        help="Expected IW depth mode stored in the checkpoint.",
    )
    parser.add_argument(
        "--iw_depth_precedes_mode",
        choices=("mlp", "factorized"),
        default=None,
        help=(
            "Expected ordered depth-node relation family stored in the checkpoint."
        ),
    )
    parser.add_argument(
        "--iw_depth_precedes_scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Expected scan setting for factorized checkpoint depth relations.",
    )
    parser.add_argument(
        "--rw_return_feature_mode",
        choices=(RW_RETURN_NONE, RW_RETURN_EXACT_FULL, RW_RETURN_EXACT_BLOCKED),
        default=None,
        help="Expected RW-return feature mode stored in the checkpoint.",
    )
    parser.add_argument(
        "--rw_walk_mode",
        choices=("standard", "non_returning"),
        default=None,
        help="Expected RW-return walk semantics stored in the checkpoint.",
    )
    parser.add_argument(
        "--rw_return_k",
        type=int,
        default=None,
        help="Expected RW-return walk length K stored in the checkpoint.",
    )
    parser.add_argument(
        "--rw_return_include_depth_nodes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Expected RW-return depth-node inclusion stored in the checkpoint.",
    )
    parser.add_argument(
        "--rw_return_lazy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Expected lazy-walk toggle stored in the checkpoint.",
    )
    parser.add_argument(
        "--rw_return_lazy_beta",
        type=float,
        default=None,
        help="Expected lazy-walk beta stored in the checkpoint.",
    )
    parser.add_argument(
        "--rw_return_block_size",
        type=int,
        default=None,
        help="Expected blocked-propagation block size stored in the checkpoint.",
    )
    parser.add_argument(
        "--rw_return_force_block_threshold",
        type=int,
        default=None,
        help="Expected forced exact_full-to-blocked threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--include_lgan_edges",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Expected LGAN enablement stored in the checkpoint.",
    )
    parser.add_argument(
        "--lgan_anchor_sources",
        nargs="+",
        default=None,
        help="Expected LGAN anchor sources stored in the checkpoint.",
    )
    parser.add_argument(
        "--lgan_tn_edge_pos",
        default=None,
        help="Expected LGAN TN relation name stored in the checkpoint.",
    )
    parser.add_argument(
        "--lgan_nn_edge_pos",
        default=None,
        help="Expected LGAN NN relation name stored in the checkpoint.",
    )
    parser.add_argument(
        "--lgan_rr_edge_pos",
        default=None,
        help="Expected LGAN RR relation name stored in the checkpoint.",
    )
    args = parser.parse_args()
    _resolve_iw_depth_cli_args(args)
    _resolve_rw_return_cli_args(args)
    _resolve_lgan_cli_args(args)
    if (
        args.iw1_atom_first_ratio is not None
        and args.iw1_atom_first_ratio <= 0.0
    ):
        parser.error("--iw1_atom_first_ratio must be > 0.")
    if (
        args.iw_depth_mode == "depth_nodes"
        and args.iw_depth_precedes_mode == "mlp"
        and args.iw_depth_precedes_scan
    ):
        parser.error(
            "--iw_depth_precedes_scan is incompatible with "
            "--iw_depth_precedes_mode=mlp."
        )
    if args.search_context_reset_interval < 0:
        parser.error("--search-context-reset-interval must be >= 0.")
    if args.search_context_reset_interval > 0 and args.search != "greedy":
        parser.error(
            "--search-context-reset-interval is only supported with "
            "--search greedy."
        )
    return args


# Helper to collect problem files from file or directory


def _collect_problem_paths(path: Path) -> list[Path]:
    """Return a sorted list of problem files to solve.

    If `path` is a directory, prefer common PDDL extensions; if none are found,
    fall back to all regular files in the directory. If `path` is a file,
    just return it.
    """
    if path.is_dir():
        # Prefer common PDDL extensions
        candidates = sorted(path.glob("*.pddl"))
        if not candidates:
            candidates = sorted(p for p in path.iterdir() if p.is_file())
        return candidates
    else:
        return [path]


def _parse_summary_metrics_line(
    line: str,
) -> dict[str, Optional[float | int]]:
    """Parse a metrics line from a previous summary into numeric values."""
    metrics: dict[str, Optional[float | int]] = {}
    # Expected format:
    # subgoals=..., primitive_steps=..., avg_Q=..., avg_prob=..., return=...
    for chunk in line.split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not value or value.lower() == "n/a":
            parsed_value: Optional[float | int] = None
        else:
            parsed_value = None
            if key in {"subgoals", "primitive_steps"}:
                try:
                    parsed_value = int(value)
                except ValueError:
                    parsed_value = None
            else:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = None
        metrics[key] = parsed_value
    return metrics


def _load_previous_run_summary(
    summary_file: Path,
) -> dict[str, object]:
    """Load persisted summary information from a previous run (if available)."""
    data: dict[str, object] = {
        "summary_exists": False,
        "per_problem_reports": [],
        "all_stats": [],
        "pending_problems": [],
        "interrupted": False,
    }

    if not summary_file.exists():
        return data

    try:
        lines = summary_file.read_text().splitlines()
    except OSError:
        return data

    data["summary_exists"] = True

    section: Optional[str] = None
    current_problem: Optional[str] = None
    current_lines: list[str] = []
    per_problem_entries: list[tuple[str, list[str]]] = []

    for line in lines:
        if "Run interrupted by user" in line:
            data["interrupted"] = True

        if line.startswith("===== Per-Problem Summary"):
            section = "per_problem"
            current_problem = None
            current_lines = []
            continue
        if line.startswith("===== Aggregated Summary"):
            if current_problem is not None:
                per_problem_entries.append((current_problem, current_lines))
                current_problem = None
                current_lines = []
            section = "aggregated"
            continue
        if line.startswith("===== Unsolved Problems"):
            if current_problem is not None:
                per_problem_entries.append((current_problem, current_lines))
                current_problem = None
                current_lines = []
            section = "unsolved"
            continue

        if section == "per_problem":
            if line.startswith("Problem: "):
                if current_problem is not None:
                    per_problem_entries.append((current_problem, current_lines))
                current_problem = line[len("Problem: ") :].strip()
                current_lines = [line]
            elif current_problem is not None:
                if line.strip() or current_lines:
                    current_lines.append(line)

    if current_problem is not None:
        per_problem_entries.append((current_problem, current_lines))

    per_problem_reports: list[dict[str, object]] = []
    all_stats: list[dict[str, object]] = []
    pending_list: list[str] = []

    for problem_str, summary_lines in per_problem_entries:
        per_problem_reports.append(
            {"problem": problem_str, "summary_lines": summary_lines}
        )

        solved_line = next(
            (
                line
                for line in summary_lines
                if line.strip().lower().startswith("solved:")
            ),
            None,
        )
        solved_text = (
            solved_line.split(":", 1)[1].strip().lower() if solved_line else ""
        )
        pending = "interrupted before processing" in solved_text
        solved = solved_text.startswith("yes")

        stats_entry: dict[str, object] = {
            "problem": problem_str,
            "solved": solved,
        }

        metrics_line = next(
            (
                line
                for line in summary_lines
                if line.strip().startswith("subgoals=")
            ),
            None,
        )
        if solved and metrics_line:
            metrics = _parse_summary_metrics_line(metrics_line)
            if "subgoals" in metrics and metrics["subgoals"] is not None:
                stats_entry["n_subgoals"] = metrics["subgoals"]
            if (
                "primitive_steps" in metrics
                and metrics["primitive_steps"] is not None
            ):
                stats_entry["n_primitives"] = metrics["primitive_steps"]
            if "avg_Q" in metrics and metrics["avg_Q"] is not None:
                stats_entry["avg_q"] = metrics["avg_Q"]
            if "avg_prob" in metrics and metrics["avg_prob"] is not None:
                stats_entry["avg_prob"] = metrics["avg_prob"]
            if "return" in metrics and metrics["return"] is not None:
                stats_entry["total_return"] = metrics["return"]
        else:
            for line in summary_lines:
                if "Plan horizon exceeded" in line:
                    match = re.search(
                        r"attempted\s+(\d+)\s+choices\s+\(limit\s+(\d+)\)",
                        line,
                    )
                    if match:
                        stats_entry["num_choices"] = int(match.group(1))
                        stats_entry["horizon"] = int(match.group(2))
                    stats_entry["horizon_exceeded"] = True
                    break

        all_stats.append(stats_entry)

        if pending:
            pending_list.append(problem_str)

    if pending_list:
        data["interrupted"] = True

    data["per_problem_reports"] = per_problem_reports
    data["all_stats"] = all_stats
    data["pending_problems"] = pending_list
    return data


def _parse_gpu_id_list(value: str) -> list[int]:
    ids: list[int] = []
    for chunk in re.split(r"[;,\s]+", value.strip()):
        if not chunk:
            continue
        match = re.search(r"(\d+)", chunk)
        if match:
            ids.append(int(match.group(1)))
    return ids


def _select_device_ids(
    requested_device: int | None, force_cpu: bool
) -> list[int]:
    """Return visible CUDA device indices [0..N-1] honoring Slurm remapping.

    - If CUDA is unavailable or CPU forced, return [].
    - Use torch.cuda.device_count() to reflect CUDA_VISIBLE_DEVICES remapping.
    - Honor a requested device only if it is within the visible range.
    """
    if force_cpu or not torch.cuda.is_available():
        return []

    count = torch.cuda.device_count()
    if count <= 0:
        return []

    if requested_device is not None and 0 <= requested_device < count:
        return [requested_device]

    return list(range(count))


def _infer_worker_index() -> Optional[int]:
    """Try to infer a stable worker index from multiprocessing process name.

    Returns a 0-based index if available (e.g., 'SpawnPoolWorker-3' -> 2),
    otherwise None.
    """
    try:
        name = mp.current_process().name
        # Typical names: 'SpawnPoolWorker-1', 'ForkPoolWorker-5'
        m = re.search(r"(\d+)$", name)
        if m:
            idx = int(m.group(1))
            return max(0, idx - 1)
    except Exception:
        pass
    return None


def _build_device_cycle(
    device_id: int | None, force_cpu: bool, strategy: str = "sticky"
) -> itertools.cycle:
    """Build a fair round-robin cycle over visible devices.

    When no explicit device is requested, rotate the start based on Slurm
    task/rank identifiers or the process PID so concurrent workers do not all
    pick device 0 first.
    """
    device_ids = _select_device_ids(device_id, force_cpu)
    if not device_ids:
        return itertools.cycle(["cpu"])

    if device_id is None:
        n = len(device_ids)
        offset_val: Optional[int] = None
        # Prefer Slurm rank/task identifiers when available
        for key in (
            "SLURM_ARRAY_TASK_ID",
            "SLURM_PROCID",
            "PMI_RANK",
            "OMPI_COMM_WORLD_RANK",
            "SLURM_JOB_ID",
        ):
            v = os.environ.get(key)
            if v and v.isdigit():
                offset_val = int(v)
                break
        # Next, try to use the worker index provided by multiprocessing
        if offset_val is None:
            widx = _infer_worker_index()
            if widx is not None:
                offset_val = widx
        # Finally, fall back to PID
        if offset_val is None:
            try:
                offset_val = os.getpid()
            except Exception:
                offset_val = 0
        if n and offset_val:
            off = offset_val % n
            if strategy == "cycle":
                # Rotate the list but keep all devices
                if off:
                    device_ids = device_ids[off:] + device_ids[:off]
                return itertools.cycle(device_ids)
            else:  # sticky
                fixed = device_ids[off]
                return itertools.cycle([fixed])

    # If a specific device was requested, or no offset available
    if strategy == "cycle":
        return itertools.cycle(device_ids)
    else:
        # sticky: pick first visible or requested
        return itertools.cycle([device_ids[0]])


def _safe_mean(values: list[float]) -> Optional[float]:
    return mean(values) if values else None


def _format_optional_float(value: Optional[float]) -> str:
    return f"{value:.5f}" if value is not None else "n/a"


def _format_optional_int(value: Any) -> str:
    return str(value) if isinstance(value, int) else "n/a"


def _format_optional_ms(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def _format_elapsed_hms(value_ms: Any) -> str:
    if not isinstance(value_ms, (int, float)):
        return "n/a"
    total_seconds = max(0, int(value_ms // 1000.0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _format_search_stats_line(stats: dict[str, Any]) -> str:
    status = stats.get("status", "n/a")
    expanded = _format_optional_int(stats.get("expanded"))
    generated = _format_optional_int(stats.get("generated"))
    pruned = _format_optional_int(stats.get("pruned"))
    f_layers = _format_optional_int(stats.get("f_layers"))
    context_resets = _format_optional_int(stats.get("context_resets"))
    last_f = stats.get("last_f")
    last_f_str = _format_optional_float(
        last_f if isinstance(last_f, (int, float)) else None
    )
    time_ms = _format_optional_ms(stats.get("time_ms"))
    solution_cost = stats.get("solution_cost")
    solution_cost_str = _format_optional_float(
        solution_cost if isinstance(solution_cost, (int, float)) else None
    )
    return (
        "search_stats: "
        f"status={status}, "
        f"expanded={expanded}, "
        f"generated={generated}, "
        f"pruned={pruned}, "
        f"f_layers={f_layers}, "
        f"context_resets={context_resets}, "
        f"last_f={last_f_str}, "
        f"time_ms={time_ms}, "
        f"solution_cost={solution_cost_str}"
    )


def _release_planning_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _rebuild_rollout_state(
    problem: mm.Problem,
    primitive_actions: list[mm.GroundAction],
) -> tuple[mm.State, set[mm.State]]:
    rebuilt_state = problem.get_initial_state()
    rebuilt_seen_states: set[mm.State] = {rebuilt_state}
    for action in primitive_actions:
        rebuilt_state = action.apply(rebuilt_state)
        rebuilt_seen_states.add(rebuilt_state)
    return rebuilt_state, rebuilt_seen_states


def _format_astar_failure_reason(
    status: str, num_choices: Optional[int], horizon: int
) -> str:
    if status == "out_of_states":
        if isinstance(num_choices, int):
            return (
                f"Plan horizon exceeded: attempted {num_choices} choices "
                f"(limit {horizon})."
            )
        return f"Plan horizon exceeded (limit {horizon})."
    if status in {"timeout", "timed_out", "time_limit"}:
        return "Search timed out."
    if status in {"unsolved", "no_solution", "failed", "failure"}:
        return "Search exhausted without finding a solution."
    return f"Search terminated with status: {status or 'unknown'}."


def _format_greedy_closed_set_failure_reason(
    *,
    had_actions: bool,
    revisiting_candidates: int,
) -> str:
    if had_actions and revisiting_candidates > 0:
        return (
            "Closed-set greedy search exhausted: all candidate actions lead "
            "to previously seen states."
        )
    if had_actions:
        return "Greedy search exhausted without finding a solution."
    return "No actions available."


def _select_greedy_action(
    current_state: mm.State,
    goal_condition: mm.GroundConjunctiveCondition,
    model: IWModel,
    reward_function: rl.RewardFunction,
    seen_states: set[mm.State],
    *,
    hard_closed_set: bool,
) -> tuple[
    Optional[mm.GroundAction],
    Optional[float],
    Optional[float],
    Optional[mm.State],
    int,
    int,
    bool,
]:
    with torch.no_grad():
        q_values, candidate_actions = model.forward([(current_state, goal_condition)])[0]

    num_candidates = len(candidate_actions)

    if not candidate_actions:
        return None, None, None, None, num_candidates, 0, False

    order = (
        torch.argsort(q_values, descending=True).detach().cpu().tolist()
    )
    revisit_count = 0
    had_unseen_candidate = False

    if hard_closed_set:
        for candidate_index in order:
            candidate_action = candidate_actions[candidate_index]
            successor_state = candidate_action.apply(current_state)
            if successor_state in seen_states:
                revisit_count += 1
                continue
            had_unseen_candidate = True
            reward = reward_function(
                current_state,
                candidate_action,
                successor_state,
                goal_condition,
            )
            return (
                candidate_action,
                float(q_values[candidate_index].item()),
                float(reward),
                successor_state,
                num_candidates,
                revisit_count,
                had_unseen_candidate,
            )
        return (
            None,
            None,
            None,
            None,
            num_candidates,
            revisit_count,
            had_unseen_candidate,
        )

    penalized_values = q_values.detach().cpu().clone()
    successor_cache: dict[int, mm.State] = {}
    reward_cache: dict[int, float] = {}
    for candidate_index, candidate_action in enumerate(candidate_actions):
        successor_state = candidate_action.apply(current_state)
        successor_cache[candidate_index] = successor_state
        reward_cache[candidate_index] = float(
            reward_function(
                current_state,
                candidate_action,
                successor_state,
                goal_condition,
            )
        )
        if successor_state in seen_states:
            penalized_values[candidate_index] = -1000000.0
            revisit_count += 1
        else:
            had_unseen_candidate = True

    chosen_index = int(torch.argmax(penalized_values).item())
    chosen_action = candidate_actions[chosen_index]
    return (
        chosen_action,
        float(q_values[chosen_index].item()),
        reward_cache[chosen_index],
        successor_cache[chosen_index],
        num_candidates,
        revisit_count,
        had_unseen_candidate,
    )


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return None
    return None


def _resolve_numeric_attr(
    transitions: list[Any], attr_names: tuple[str, ...]
) -> Optional[str]:
    for name in attr_names:
        for transition in transitions:
            if not hasattr(transition, name):
                continue
            if _to_float(getattr(transition, name)) is not None:
                return name
    return None


def _resolve_wandb_info_path(
    args: argparse.Namespace,
    storage_output_dir: Path,
    model_plan_output_dir: Path | None,
    model_summary_output_dir: Path | None,
) -> Path | None:
    explicit = args.wandb_run_info
    if explicit is not None:
        return explicit if explicit.suffix else explicit / "wandb_run.json"

    candidate_dirs = [
        model_summary_output_dir,
        model_plan_output_dir,
        storage_output_dir,
    ]
    for directory in candidate_dirs:
        if directory is None:
            continue
        candidate = directory / "wandb_run.json"
        if candidate.exists():
            return candidate
    return None


def _init_wandb_run(info_path: Path | None) -> wandb.sdk.wandb_run.Run | None:
    if info_path is None or not info_path.exists():
        return None

    try:
        data = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError):
        print(f"Warning: unable to read wandb run info from {info_path}.")
        return None

    run_id = data.get("id") or data.get("run_id")
    project = data.get("project")
    entity = data.get("entity")
    name = data.get("name")

    if not run_id or not project:
        print(
            f"Warning: wandb run info at {info_path} missing required fields 'id' or 'project'."
        )
        return None

    init_kwargs: dict[str, Any] = {
        "project": project,
        "id": run_id,
        "resume": "allow",
        "reinit": True,
    }
    if entity:
        init_kwargs["entity"] = entity
    if name:
        init_kwargs["name"] = name

    run = wandb.init(**init_kwargs)
    return run


def _compute_stats(trajectory) -> dict:
    """Compute summary stats for a (solution) trajectory.

    Returns a dict with keys:
      - n_subgoals: int
      - n_primitives: int
      - avg_q: Optional[float]
      - avg_prob: Optional[float]
      - total_return: Optional[float]
    Any Optional[float] may be None if unavailable in the trajectory transitions.
    """
    # Count subgoals as number of transitions in the trajectory
    transitions = list(trajectory)
    n_subgoals = len(transitions)

    q_attr = _resolve_numeric_attr(transitions, ("predicted_q_value",))
    prob_attr = _resolve_numeric_attr(
        transitions,
        (
            "selected_action_probability",
            "action_probability",
            "probability",
            "predicted_probability",
            "pi",
        ),
    )
    reward_attr = _resolve_numeric_attr(
        transitions, ("observed_reward", "reward", "r")
    )

    q_vals: list[float] = []
    probs: list[float] = []
    rewards: list[float] = []

    n_primitives = 0

    for t in transitions:
        if q_attr is not None:
            q_val = _to_float(getattr(t, q_attr, None))
            if q_val is not None:
                q_vals.append(q_val)

        if prob_attr is not None:
            prob_val = _to_float(getattr(t, prob_attr, None))
            if prob_val is not None:
                probs.append(prob_val)

        if reward_attr is not None:
            reward_val = _to_float(getattr(t, reward_attr, None))
            if reward_val is not None:
                rewards.append(reward_val)

        selected_action = getattr(t, "selected_action", None)
        if isinstance(selected_action, IWGroundAction):
            n_primitives += len(selected_action.get_underlying_actions())
        elif isinstance(selected_action, mm.GroundAction):
            n_primitives += 1

    return {
        "n_subgoals": n_subgoals,
        "n_primitives": n_primitives,
        "avg_q": _safe_mean(q_vals),
        "avg_prob": _safe_mean(probs),
        "total_return": sum(rewards) if rewards else None,
    }


def _predict_max_q(
    model: IWModel,
    state: mm.State,
    goal: mm.GroundConjunctiveCondition,
) -> Optional[float]:
    with torch.no_grad():
        q_values, _ = model.forward([(state, goal)])[0]
    if not isinstance(q_values, torch.Tensor) or q_values.numel() == 0:
        return None
    return float(q_values.max().item())


def _predict_q_for_action(
    model: IWModel,
    state: mm.State,
    goal: mm.GroundConjunctiveCondition,
    action: mm.GroundAction,
) -> Optional[float]:
    with torch.no_grad():
        q_values, actions = model.forward([(state, goal)])[0]
    if not isinstance(q_values, torch.Tensor) or q_values.numel() == 0:
        return None
    for q_value, candidate in zip(q_values, actions):
        if candidate == action:
            return float(q_value.item())
    return float(q_values.max().item())


def _predict_q_for_action_with_choice_count(
    model: IWModel,
    state: mm.State,
    goal: mm.GroundConjunctiveCondition,
    action: mm.GroundAction,
) -> tuple[Optional[float], int]:
    with torch.no_grad():
        q_values, actions = model.forward([(state, goal)])[0]
    num_choices = len(actions)
    if not isinstance(q_values, torch.Tensor) or q_values.numel() == 0:
        return None, num_choices
    for q_value, candidate in zip(q_values, actions):
        if candidate == action:
            return float(q_value.item()), num_choices
    return float(q_values.max().item()), num_choices


class _ModelQHeuristic(mm.Heuristic):
    def __init__(
        self,
        model: IWModel,
        goal: mm.GroundConjunctiveCondition,
    ) -> None:
        super().__init__()
        self._model = model
        self._goal = goal
        self._cache: dict[mm.State, float] = {}

    def compute_value(
        self,
        state: mm.State,
        goal: Optional[mm.GroundConjunctiveCondition] = None,
    ) -> float:
        if state in self._cache:
            return self._cache[state]
        goal_condition = goal if goal is not None else self._goal
        if goal_condition.holds(state):
            self._cache[state] = 0.0
            return 0.0
        max_q = _predict_max_q(self._model, state, goal_condition)
        heuristic_value = float("inf") if max_q is None else -max_q
        self._cache[state] = heuristic_value
        return heuristic_value

    def get_preferred_actions(self) -> set[mm.GroundAction]:
        return set()


def _worker_init(
    domain_path: str,
    model_path: str,
    trace_output_dir: str | None,
    trace_stdout_enabled: bool,
    device_id: int,
    force_cpu: bool,
    horizon: int,
    device_strategy: str,
    search: str,
    closed_set_greedy: bool,
    search_context_reset_interval: int,
    flat: bool,
    iw1_precheck_add_effect_novelty: bool | None,
    iw1_atom_first_mode: bool | None,
    iw1_atom_first_ratio: float | None,
    iw1_incremental_first_applicability: bool | None,
    iw1_incremental_first_applicability_debug_crosscheck: bool | None,
    iw_depth_override_requested: bool,
    iw_depth_mode: str,
    iw_depth_precedes_mode: str,
    iw_depth_precedes_scan: bool,
    lgan_override_requested: bool,
    include_lgan_edges: bool,
    lgan_anchor_sources: list[str],
    lgan_tn_edge_pos: str,
    lgan_nn_edge_pos: str,
    lgan_rr_edge_pos: str,
    rw_return_override_requested: bool,
    rw_return_feature_mode: str,
    rw_return_k: int,
    rw_return_include_depth_nodes: bool,
    rw_return_lazy: bool,
    rw_return_lazy_beta: float,
    rw_return_block_size: int,
    rw_return_force_block_threshold: int,
) -> None:
    global _WORKER_CONTEXT, _LIVE_TRACE_TO_STDOUT

    mp.set_start_method("spawn", force=True)
    _LIVE_TRACE_TO_STDOUT = trace_stdout_enabled
    _WORKER_CONTEXT = {
        "domain_path": domain_path,
        "model_path": model_path,
        "trace_output_dir": trace_output_dir,
        "device_cycle": _build_device_cycle(
            device_id, force_cpu, device_strategy
        ),
        "force_cpu": force_cpu,
        "horizon": horizon,
        "search": search,
        "closed_set_greedy": closed_set_greedy,
        "search_context_reset_interval": search_context_reset_interval,
        "flat": flat,
        "iw1_precheck_add_effect_novelty": iw1_precheck_add_effect_novelty,
        "iw1_atom_first_mode": iw1_atom_first_mode,
        "iw1_atom_first_ratio": iw1_atom_first_ratio,
        "iw1_incremental_first_applicability": iw1_incremental_first_applicability,
        "iw1_incremental_first_applicability_debug_crosscheck": iw1_incremental_first_applicability_debug_crosscheck,
        "iw_depth_override_requested": iw_depth_override_requested,
        "iw_depth_mode": iw_depth_mode,
        "iw_depth_precedes_mode": iw_depth_precedes_mode,
        "iw_depth_precedes_scan": iw_depth_precedes_scan,
        "lgan_override_requested": lgan_override_requested,
        "include_lgan_edges": include_lgan_edges,
        "lgan_anchor_sources": lgan_anchor_sources,
        "lgan_tn_edge_pos": lgan_tn_edge_pos,
        "lgan_nn_edge_pos": lgan_nn_edge_pos,
        "lgan_rr_edge_pos": lgan_rr_edge_pos,
        "rw_return_override_requested": rw_return_override_requested,
        "rw_return_feature_mode": rw_return_feature_mode,
        "rw_return_k": rw_return_k,
        "rw_return_include_depth_nodes": rw_return_include_depth_nodes,
        "rw_return_lazy": rw_return_lazy,
        "rw_return_lazy_beta": rw_return_lazy_beta,
        "rw_return_block_size": rw_return_block_size,
        "rw_return_force_block_threshold": rw_return_force_block_threshold,
    }


def _solve_problem(job: tuple[int, str]) -> tuple[int, dict[str, Any]]:
    index, problem_path_str = job
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context has not been initialised.")

    global _CURRENT_PROBLEM_STR, _CURRENT_TRACE_PATH
    domain_path: str = _WORKER_CONTEXT["domain_path"]
    model_path: str = _WORKER_CONTEXT["model_path"]
    trace_output_dir = _WORKER_CONTEXT.get("trace_output_dir")
    device_cycle = _WORKER_CONTEXT["device_cycle"]
    force_cpu = _WORKER_CONTEXT["force_cpu"]
    horizon: int = _WORKER_CONTEXT["horizon"]
    flat: bool = _WORKER_CONTEXT["flat"]
    iw1_precheck_add_effect_novelty = _WORKER_CONTEXT.get(
        "iw1_precheck_add_effect_novelty"
    )
    iw1_atom_first_mode = _WORKER_CONTEXT.get("iw1_atom_first_mode")
    iw1_atom_first_ratio = _WORKER_CONTEXT.get("iw1_atom_first_ratio")
    iw1_incremental_first_applicability = _WORKER_CONTEXT.get(
        "iw1_incremental_first_applicability"
    )
    iw1_incremental_first_applicability_debug_crosscheck = (
        _WORKER_CONTEXT.get(
            "iw1_incremental_first_applicability_debug_crosscheck"
        )
    )
    iw_depth_override_requested = bool(
        _WORKER_CONTEXT.get("iw_depth_override_requested", False)
    )
    lgan_override_requested = bool(
        _WORKER_CONTEXT.get("lgan_override_requested", False)
    )
    rw_return_override_requested = bool(
        _WORKER_CONTEXT.get("rw_return_override_requested", False)
    )
    search_strategy = _WORKER_CONTEXT.get("search", "greedy")
    closed_set_greedy = bool(
        _WORKER_CONTEXT.get("closed_set_greedy", False)
    )
    search_context_reset_interval = int(
        _WORKER_CONTEXT.get("search_context_reset_interval", 0) or 0
    )

    problem_path = Path(problem_path_str)
    problem_abs = problem_path.resolve()
    problem_str = str(problem_abs)
    trace_label = problem_path.stem
    trace_path: Optional[Path] = None
    if trace_output_dir:
        trace_path = Path(trace_output_dir) / f"{problem_path.stem}.trace"
        _write_trace_header(trace_path, problem_str, search_strategy, horizon)
        _emit_live_trace(
            trace_path,
            trace_label,
            "Trace-Note: entries below are an in-progress search trace and may be incomplete.",
        )
        _CURRENT_TRACE_PATH = trace_path
    _CURRENT_PROBLEM_STR = problem_str

    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)

    device_token = next(device_cycle)
    if device_token == "cpu" or force_cpu:
        device = torch.device("cpu")
    else:
        device = create_device(device=int(device_token), force_cpu=False)
    model = IWModel.load(
        domain,
        Path(model_path),
        device,
        flat=flat,
        iw1_precheck_add_effect_novelty=iw1_precheck_add_effect_novelty,
        iw1_atom_first_mode=iw1_atom_first_mode,
        iw1_atom_first_ratio=iw1_atom_first_ratio,
        iw1_incremental_first_applicability=(
            iw1_incremental_first_applicability
        ),
        iw1_incremental_first_applicability_debug_crosscheck=(
            iw1_incremental_first_applicability_debug_crosscheck
        ),
    )
    if iw_depth_override_requested:
        requested_depth_config = resolve_iw_depth_config(
            iw_depth_mode=_WORKER_CONTEXT.get("iw_depth_mode"),
            iw_depth_precedes_mode=_WORKER_CONTEXT.get(
                "iw_depth_precedes_mode"
            ),
            iw_depth_precedes_scan=_WORKER_CONTEXT.get(
                "iw_depth_precedes_scan"
            ),
        )
        if requested_depth_config != model.iw_depth_config:
            raise RuntimeError(
                "Requested IW depth config does not match checkpoint: "
                f"requested={requested_depth_config}, "
                f"checkpoint={model.iw_depth_config}."
            )
    if lgan_override_requested:
        requested_lgan_config = resolve_lgan_config(
            include_lgan_edges=_WORKER_CONTEXT.get("include_lgan_edges"),
            lgan_anchor_sources=_WORKER_CONTEXT.get("lgan_anchor_sources"),
            lgan_tn_edge_pos=_WORKER_CONTEXT.get("lgan_tn_edge_pos"),
            lgan_nn_edge_pos=_WORKER_CONTEXT.get("lgan_nn_edge_pos"),
            lgan_rr_edge_pos=_WORKER_CONTEXT.get("lgan_rr_edge_pos"),
        )
        if requested_lgan_config != model.lgan_config:
            raise RuntimeError(
                "Requested LGAN config does not match checkpoint: "
                f"requested={requested_lgan_config}, checkpoint={model.lgan_config}."
            )
    if rw_return_override_requested:
        requested_rw_return_config = resolve_rw_return_config(
            rw_return_feature_mode=_WORKER_CONTEXT.get(
                "rw_return_feature_mode"
            ),
            rw_return_k=_WORKER_CONTEXT.get("rw_return_k"),
            rw_return_include_depth_nodes=_WORKER_CONTEXT.get(
                "rw_return_include_depth_nodes"
            ),
            rw_return_lazy=_WORKER_CONTEXT.get("rw_return_lazy"),
            rw_return_lazy_beta=_WORKER_CONTEXT.get(
                "rw_return_lazy_beta"
            ),
            rw_return_block_size=_WORKER_CONTEXT.get(
                "rw_return_block_size"
            ),
            rw_return_force_block_threshold=_WORKER_CONTEXT.get(
                "rw_return_force_block_threshold"
            ),
        )
        if requested_rw_return_config != model.rw_return_config:
            raise RuntimeError(
                "Requested RW-return config does not match checkpoint: "
                f"requested={requested_rw_return_config}, "
                f"checkpoint={model.rw_return_config}."
            )
    stats: Optional[dict[str, Any]] = None
    search_stats: Optional[dict[str, Any]] = None
    failure_reason: Optional[str] = None
    plan_steps: list[str] = []
    num_choices = 0
    horizon_exceeded = False
    solved = False
    context_reset_count = 0

    if search_strategy == "astar":
        model.eval()
        initial_state = problem.get_initial_state()
        goal_condition = problem.get_goal_condition()
        heuristic = _ModelQHeuristic(model, goal_condition)
        expanded_states = 0
        generated_states = 0
        pruned_states = 0
        finished_layers = 0
        last_f_value: Optional[float] = None

        def _on_expand_state(_: mm.State) -> None:
            nonlocal expanded_states
            expanded_states += 1

        def _on_expand_goal_state(_: mm.State) -> None:
            nonlocal expanded_states
            expanded_states += 1

        def _on_generate_state(
            _state: mm.State,
            _action: mm.GroundAction,
            _cost: float,
            _successor: mm.State,
        ) -> None:
            nonlocal generated_states
            generated_states += 1

        def _on_prune_state(_: mm.State) -> None:
            nonlocal pruned_states
            pruned_states += 1

        def _on_finish_f_layer(f_value: float) -> None:
            nonlocal finished_layers, last_f_value
            finished_layers += 1
            last_f_value = float(f_value)
            _emit_live_trace(
                trace_path,
                trace_label,
                (
                    f"Search-Progress: f_layer={finished_layers}, last_f={float(f_value):.5f}, "
                    f"expanded={expanded_states}, generated={generated_states}, pruned={pruned_states}"
                ),
            )

        start_time = time.perf_counter()
        _emit_live_trace(
            trace_path,
            trace_label,
            "Search-Progress: starting A* search.",
        )
        result = mm.astar_eager(
            problem,
            initial_state,
            heuristic,
            max_num_states=horizon,
            on_expand_state=_on_expand_state,
            on_expand_goal_state=_on_expand_goal_state,
            on_generate_state=_on_generate_state,
            on_prune_state=_on_prune_state,
            on_finish_f_layer=_on_finish_f_layer,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        status = str(result.status).lower()
        solved = status == "solved"
        horizon_exceeded = status == "out_of_states"
        num_choices = expanded_states if expanded_states else generated_states
        if not solved:
            failure_reason = _format_astar_failure_reason(
                status, num_choices, horizon
            )
        solution_actions = result.solution or []
        search_stats = {
            "status": status,
            "expanded": expanded_states,
            "generated": generated_states,
            "pruned": pruned_states,
            "f_layers": finished_layers,
            "context_resets": context_reset_count,
            "last_f": last_f_value,
            "time_ms": elapsed_ms,
            "solution_cost": result.solution_cost,
        }

        if solved:
            primitive_counter = 1
            q_values_for_stats: list[float] = []
            current_state = initial_state
            _emit_live_trace(
                trace_path, trace_label, "Solution-Trace: begin"
            )
            for macro_index, action in enumerate(solution_actions, start=1):
                q_value, iw_action_count = _predict_q_for_action_with_choice_count(
                    model, current_state, goal_condition, action
                )
                if q_value is None:
                    max_q = _predict_max_q(model, current_state, goal_condition)
                    q_value = max_q if max_q is not None else 0.0
                elapsed_ms_now = (time.perf_counter() - start_time) * 1000.0
                macro_line = (
                    f"---- [{_format_elapsed_hms(elapsed_ms_now)}] "
                    f"Macro {macro_index:>4}: === {q_value:.5f} === "
                    f"IW-actions: {iw_action_count}"
                )
                plan_steps.append(macro_line)
                _emit_live_trace(trace_path, trace_label, macro_line)
                if isinstance(action, IWGroundAction):
                    actions = action.get_underlying_actions()
                    for local_index, primitive_action in enumerate(
                        actions, start=1
                    ):
                        primitive_line = (
                            f"{primitive_counter:>4} "
                            f"[subgoal {macro_index:>3}, primitive {local_index:>3}]: "
                            f"{str(primitive_action)}"
                        )
                        plan_steps.append(primitive_line)
                        _emit_live_trace(
                            trace_path, trace_label, primitive_line
                        )
                        primitive_counter += 1
                else:
                    primitive_line = (
                        f"{primitive_counter:>4} "
                        f"[subgoal {macro_index:>3}, primitive {1:>3}]: "
                        f"{str(action)}"
                    )
                    plan_steps.append(primitive_line)
                    _emit_live_trace(trace_path, trace_label, primitive_line)
                    primitive_counter += 1
                q_values_for_stats.append(q_value)
                current_state = action.apply(current_state)

            stats = {
                "n_subgoals": len(solution_actions),
                "n_primitives": primitive_counter - 1,
                "avg_q": _safe_mean(q_values_for_stats),
                "avg_prob": None,
                "total_return": None,
            }
    else:
        initial_state = problem.get_initial_state()
        goal_condition = problem.get_goal_condition()
        reward_function = rl.ConstantRewardFunction()
        current_state = initial_state
        seen_states: set[mm.State] = {current_state}
        executed_primitive_actions: list[mm.GroundAction] = []
        primitive_counter = 1
        q_values_for_stats: list[float] = []
        rewards_for_stats: list[float] = []
        revisiting_candidates = 0
        start_time = time.perf_counter()
        _emit_live_trace(
            trace_path,
            trace_label,
            (
                "Search-Progress: starting greedy rollout"
                + (" with closed set." if closed_set_greedy else ".")
            ),
        )
        _emit_live_trace(trace_path, trace_label, "Attempted-Trace: begin")
        for macro_index in range(1, horizon + 1):
            if goal_condition.holds(current_state):
                solved = True
                break

            (
                chosen_action,
                chosen_q_value,
                reward_value,
                chosen_successor,
                iw_action_count,
                revisiting_this_step,
                had_unseen_candidate,
            ) = _select_greedy_action(
                current_state,
                goal_condition,
                model,
                reward_function,
                seen_states,
                hard_closed_set=closed_set_greedy,
            )
            revisiting_candidates += revisiting_this_step

            if chosen_action is None or chosen_q_value is None:
                if closed_set_greedy:
                    failure_reason = _format_greedy_closed_set_failure_reason(
                        had_actions=bool(
                            revisiting_this_step or had_unseen_candidate
                        ),
                        revisiting_candidates=revisiting_this_step,
                    )
                else:
                    failure_reason = "No actions available."
                break

            elapsed_ms_now = (time.perf_counter() - start_time) * 1000.0
            macro_line = (
                f"---- [{_format_elapsed_hms(elapsed_ms_now)}] "
                f"Macro {macro_index:>4}: === {chosen_q_value:.5f} === "
                f"IW-actions: {iw_action_count}"
            )
            plan_steps.append(macro_line)
            _emit_live_trace(trace_path, trace_label, macro_line)

            primitive_actions = (
                chosen_action.get_underlying_actions()
                if isinstance(chosen_action, IWGroundAction)
                else [chosen_action]
            )
            for local_index, action in enumerate(
                primitive_actions, start=1
            ):
                primitive_line = (
                    f"{primitive_counter:>4} "
                    f"[subgoal {macro_index:>3}, primitive {local_index:>3}]: "
                    f"{str(action)}"
                )
                plan_steps.append(primitive_line)
                _emit_live_trace(trace_path, trace_label, primitive_line)
                primitive_counter += 1

            executed_primitive_actions.extend(primitive_actions)
            q_values_for_stats.append(chosen_q_value)
            if reward_value is not None:
                rewards_for_stats.append(reward_value)
            num_choices = macro_index
            current_state = chosen_successor
            assert current_state is not None
            if not had_unseen_candidate and not closed_set_greedy:
                _emit_live_trace(
                    trace_path,
                    trace_label,
                    "Search-Progress: revisiting state because all candidates were already seen.",
                )
            seen_states.add(current_state)
            if (
                search_context_reset_interval > 0
                and macro_index % search_context_reset_interval == 0
            ):
                reset_count = reset_problem_search_context(problem)
                clear_search_context_caches()
                if reset_count > 0:
                    context_reset_count += reset_count
                    current_state, seen_states = _rebuild_rollout_state(
                        problem,
                        executed_primitive_actions,
                    )
                    _emit_live_trace(
                        trace_path,
                        trace_label,
                        (
                            "Search-Progress: rebuilt search context after "
                            f"{macro_index} greedy choice(s); "
                            f"total_resets={context_reset_count}."
                        ),
                    )
                    _release_planning_memory(device)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if not solved and goal_condition.holds(current_state):
            solved = True
        if not solved and num_choices >= horizon:
            horizon_exceeded = True
        if not solved and failure_reason is None and closed_set_greedy:
            failure_reason = _format_greedy_closed_set_failure_reason(
                had_actions=True,
                revisiting_candidates=revisiting_candidates,
            )
        search_stats = {
            "status": (
                "solved"
                if solved
                else (
                    "out_of_states"
                    if horizon_exceeded
                    else (
                        "closed_set_exhausted"
                        if closed_set_greedy
                        else "failed"
                    )
                )
            ),
            "expanded": num_choices,
            "generated": None,
            "pruned": revisiting_candidates if closed_set_greedy else None,
            "f_layers": None,
            "context_resets": context_reset_count,
            "last_f": None,
            "time_ms": elapsed_ms,
            "solution_cost": None,
        }
        if solved:
            stats = {
                "n_subgoals": num_choices,
                "n_primitives": primitive_counter - 1,
                "avg_q": _safe_mean(q_values_for_stats),
                "avg_prob": None,
                "total_return": (
                    sum(rewards_for_stats) if rewards_for_stats else None
                ),
            }
        _emit_live_trace(
            trace_path,
            trace_label,
            (
                "Search-Progress: greedy rollout finished with "
                f"{num_choices} transition(s)"
                + (
                    f", revisits_blocked={revisiting_candidates}."
                    if closed_set_greedy
                    else "."
                )
            ),
        )

    del model
    del problem
    del domain
    _release_planning_memory(device)

    _append_trace_line(trace_path)
    _emit_live_trace(
        trace_path,
        trace_label,
        f"Final-Status: {'solved' if solved else 'failed'}",
    )
    if isinstance(failure_reason, str) and failure_reason:
        _emit_live_trace(
            trace_path,
            trace_label,
            f"Failure-Reason: {failure_reason}",
        )
    if horizon_exceeded:
        _emit_live_trace(
            trace_path,
            trace_label,
            f"Horizon-Exceeded: attempted {num_choices} choices (limit {horizon}).",
        )
    _emit_live_trace(trace_path, trace_label, "Trace-Complete: yes")

    del problem_path
    del problem_abs
    _CURRENT_TRACE_PATH = None
    _CURRENT_PROBLEM_STR = None
    return (
        index,
        {
            "problem": problem_str,
            "solved": solved,
            "plan_steps": plan_steps,
            "stats": stats,
            "search_stats": search_stats,
            "failure_reason": failure_reason,
            "horizon": horizon,
            "horizon_exceeded": horizon_exceeded,
            "num_choices": num_choices,
        },
    )


def _main(args: argparse.Namespace) -> None:
    global _CURRENT_PROBLEM_SUMMARY_PATH, _CURRENT_PROBLEM_STR, _CURRENT_TRACE_PATH
    global _INTERRUPT_HANDLED

    _CURRENT_PROBLEM_SUMMARY_PATH = None
    _CURRENT_PROBLEM_STR = None
    _CURRENT_TRACE_PATH = None
    _INTERRUPT_HANDLED = False

    print(f"Torch: {torch.__version__}")
    initial_problem_paths = _collect_problem_paths(args.problem)

    model_output_segment = (
        args.model.stem if args.model.suffix else args.model.name
    )
    if not model_output_segment:
        model_output_segment = "model"

    plan_output_dir = args.plan_output_dir
    summary_output_dir = args.summary_output_dir
    model_plan_output_dir = None
    model_summary_output_dir = None
    model_trace_output_dir = None
    if plan_output_dir is not None:
        # Keep per-model subdirectory layout for plan files
        model_plan_output_dir = plan_output_dir / model_output_segment
        model_plan_output_dir.mkdir(parents=True, exist_ok=True)
    if summary_output_dir is not None:
        # Keep per-model subdirectory layout for summaries
        model_summary_output_dir = summary_output_dir / model_output_segment
        model_summary_output_dir.mkdir(parents=True, exist_ok=True)

    if model_summary_output_dir is not None:
        storage_output_dir = model_summary_output_dir
    elif model_plan_output_dir is not None:
        storage_output_dir = model_plan_output_dir
    else:
        storage_output_dir = Path.cwd() / f"{model_output_segment}_outputs"
        storage_output_dir.mkdir(parents=True, exist_ok=True)
    if model_plan_output_dir is not None:
        model_trace_output_dir = model_plan_output_dir
    else:
        model_trace_output_dir = storage_output_dir
    model_trace_output_dir.mkdir(parents=True, exist_ok=True)

    unsolved_file_path = storage_output_dir / "unsolved_problems.txt"
    summary_file_path = storage_output_dir / "planning_summary.txt"

    problem_paths: list[Path] = [
        p for p in initial_problem_paths if p.name != "domain.pddl"
    ]
    # Treat --retry_all_failed as an alias activating retry mode as well
    retry_mode = bool(
        getattr(args, "retry_failed", False)
        or getattr(args, "retry_all_failed", False)
    )
    if retry_mode and getattr(args, "continue_run", False):
        print("Options --retry_failed and --continue cannot be used together.")
        return

    preexisting_reports: list[dict[str, object]] = []
    preexisting_stats: list[dict[str, object]] = []
    resume_pending: list[str] = []
    resuming = False

    if retry_mode:
        if not unsolved_file_path.exists():
            print(
                f"No previous unsolved problem list found at {unsolved_file_path}; nothing to retry."
            )
            return

        retry_lines = [
            line.strip()
            for line in unsolved_file_path.read_text().splitlines()
            if line.strip()
        ]
        if not retry_lines:
            print("Unsolved problem list is empty; nothing to retry.")
            return

        retry_paths: list[Path] = []
        missing_paths: list[Path] = []
        for entry in retry_lines:
            candidate = Path(entry).expanduser()
            if candidate.exists():
                retry_paths.append(candidate)
            else:
                missing_paths.append(candidate)

        if missing_paths:
            print(
                "Warning: the following previously failed problems could not be found:"
            )
            for missing in missing_paths:
                print(f" - {missing}")

        if not retry_paths:
            print("No valid failed problems remain to retry; exiting.")
            return

        # If not explicitly asked to retry all failures, drop those that already hit the horizon
        if not getattr(args, "retry_all_failed", False):

            def _hit_horizon(stem: str) -> bool | None:
                # Prefer per-problem summary if available
                try:
                    if model_summary_output_dir is not None:
                        sf = model_summary_output_dir / f"{stem}_summary.txt"
                        if sf.exists():
                            txt = sf.read_text(errors="ignore")
                            if "Plan horizon exceeded" in txt:
                                return True
                            return False
                except Exception:
                    pass
                # Fallback to plan file if present
                try:
                    if model_plan_output_dir is not None:
                        pf = model_plan_output_dir / f"{stem}.plan"
                        if pf.exists():
                            txt = pf.read_text(errors="ignore")
                            if "Plan horizon exceeded" in txt:
                                return True
                            return False
                except Exception:
                    pass
                return None  # Unknown

            kept: list[Path] = []
            skipped: list[Path] = []
            for p in retry_paths:
                s = p.stem
                flag = _hit_horizon(s)
                if flag is True:
                    skipped.append(p)
                else:
                    kept.append(p)

            if skipped:
                print(
                    f"Skipping {len(skipped)} problem(s) that previously exceeded the planning horizon; "
                    "use --retry_all_failed to include them."
                )
            if not kept:
                print("No non-horizon failures remain to retry; exiting.")
                return
            problem_paths = kept
        else:
            problem_paths = retry_paths

    if getattr(args, "continue_run", False):
        previous_summary = _load_previous_run_summary(summary_file_path)
        if not previous_summary.get("summary_exists", False):
            # No prior summary: treat all problems without per-problem summaries as unseen
            unseen_paths: list[Path] = []
            if model_summary_output_dir is not None:
                for p in problem_paths:
                    sf = model_summary_output_dir / f"{p.stem}_summary.txt"
                    if not sf.exists():
                        unseen_paths.append(p)
            else:
                unseen_paths = list(problem_paths)
            if not unseen_paths:
                print(
                    f"No previous summary at {summary_file_path} and no unseen problems; nothing to continue."
                )
                return
            problem_paths = unseen_paths
            resuming = True
            print(
                f"Continuing with {len(problem_paths)} unseen problem(s) (no prior summary found)."
            )
        else:
            resume_pending = list(previous_summary.get("pending_problems", []))
            if not resume_pending:
                if previous_summary.get("interrupted", False):
                    print(
                        "Previous run does not list any pending problems; nothing to continue."
                    )
                else:
                    print(
                        "Previous run does not appear to have been interrupted; nothing to continue."
                    )
                return

        resolved_map = {str(path.resolve()): path for path in problem_paths}
        selected_paths: list[Path] = []
        selected_pending_strings: set[str] = set()
        missing_pending: list[str] = []
        for pending_problem in resume_pending:
            candidate = resolved_map.get(pending_problem)
            if candidate is not None:
                selected_paths.append(candidate)
                selected_pending_strings.add(pending_problem)
            else:
                missing_pending.append(pending_problem)

        if missing_pending:
            print(
                "Warning: the following pending problems could not be matched to the provided problem set:"
            )
            for missing_entry in missing_pending:
                print(f" - {missing_entry}")

        # Also include any problems that have not yet produced a summary file
        unseen_paths: list[Path] = []
        if model_summary_output_dir is not None:
            for p in problem_paths:
                sf = model_summary_output_dir / f"{p.stem}_summary.txt"
                if not sf.exists():
                    unseen_paths.append(p)
        else:
            # Fallback: include all problems not explicitly selected
            unseen_paths = [p for p in problem_paths if p not in selected_paths]

            for p in unseen_paths:
                if p not in selected_paths:
                    selected_paths.append(p)

            if not selected_paths:
                print("No pending or unseen problems to continue; exiting.")
                return

            problem_paths = selected_paths
            preexisting_reports = [
                {
                    "problem": report["problem"],
                    "summary_lines": list(report["summary_lines"]),
                }
                for report in previous_summary.get("per_problem_reports", [])
                if report["problem"] not in selected_pending_strings
            ]
            preexisting_stats = [
                dict(entry)
                for entry in previous_summary.get("all_stats", [])
                if entry["problem"] not in selected_pending_strings
            ]
            resuming = True

    total_problems = len(problem_paths)
    if total_problems == 0:
        print("No problems to solve.")
        return

    if resuming:
        print(f"Resuming with {total_problems} problem(s) to process.")
    else:
        print(f"Found {total_problems} problem(s) to solve.")

    wandb_info_path = _resolve_wandb_info_path(
        args,
        storage_output_dir,
        model_plan_output_dir,
        model_summary_output_dir,
    )
    wandb_run = _init_wandb_run(wandb_info_path)

    per_problem_reports: list[dict[str, object]] = list(preexisting_reports)
    all_stats: list[dict] = list(preexisting_stats)
    interrupted = False

    run_verbosity = getattr(args, "verbosity", "summary")
    progress_bar = None
    progress_enabled = (
        run_verbosity == "quiet" and total_problems > 0 and tqdm is not None
    )
    if progress_enabled:
        progress_bar = tqdm(total=total_problems, unit="problem", leave=False)

    def _process_result(
        idx: int,
        result: dict[str, Any],
    ) -> None:
        nonlocal progress_bar
        problem_path = Path(result["problem"])
        problem_abs = problem_path.resolve()
        problem_str = str(problem_abs)
        solved = bool(result["solved"])
        plan_steps: list[str] = result["plan_steps"]
        stats = result.get("stats")
        search_stats = result.get("search_stats")
        failure_reason = result.get("failure_reason")
        horizon_limit = result.get("horizon", PLAN_HORIZON)
        horizon_exceeded = bool(result.get("horizon_exceeded"))
        num_choices = result.get("num_choices")

        if run_verbosity != "quiet":
            print(
                f"\n===== Problem {idx}/{total_problems}: {problem_str} ====="
            )
            if solved:
                plan_line_count = len(plan_steps)
                if run_verbosity == "full":
                    for line in plan_steps:
                        print(line)
                elif run_verbosity == "summary":
                    if plan_line_count <= SUMMARY_PLAN_LINE_THRESHOLD:
                        for line in plan_steps:
                            print(line)
                    else:
                        print(
                            f"Plan has {plan_line_count} lines; showing first "
                            f"{SUMMARY_SNIPPET_LINES} lines. Use --verbosity full "
                            "to print the entire plan."
                        )
                        for line in plan_steps[:SUMMARY_SNIPPET_LINES]:
                            print(line)
                        print("...")
                        print(
                            f"(Plan truncated; {max(plan_line_count - SUMMARY_SNIPPET_LINES, 0)} "
                            "additional lines omitted.)"
                        )
                else:
                    print(f"Solved with plan of {plan_line_count} line(s).")
            else:
                print(f"Failed to find a solution for: {problem_str}")
                if isinstance(failure_reason, str) and failure_reason:
                    print(f"Reason: {failure_reason}")
                if horizon_exceeded and (
                    not isinstance(failure_reason, str)
                    or "Plan horizon exceeded" not in failure_reason
                ):
                    print(
                        f"Plan horizon exceeded: attempted {num_choices} choices "
                        f"(limit {horizon_limit})."
                    )

        plan_lines: list[str] = [
            f"Problem: {problem_str}",
            f"Solved: {'yes' if solved else 'no'}",
        ]
        if isinstance(search_stats, dict):
            plan_lines.append(_format_search_stats_line(search_stats))
        if solved and plan_steps:
            plan_lines.extend(plan_steps)
        elif not solved:
            plan_lines.append("No plan found.")
            if isinstance(failure_reason, str) and failure_reason:
                plan_lines.append(f"Reason: {failure_reason}")
            if horizon_exceeded:
                if not (
                    isinstance(failure_reason, str)
                    and "Plan horizon exceeded" in failure_reason
                ):
                    plan_lines.append(
                        f"Plan horizon exceeded: attempted {num_choices} choices "
                        f"(limit {horizon_limit})."
                    )
            if plan_steps:
                plan_lines.append("Attempted trace:")
                plan_lines.extend(plan_steps)

        summary_lines: list[str] = [
            f"Problem: {problem_str}",
            f"Solved: {'yes' if solved else 'no'}",
        ]
        if isinstance(search_stats, dict):
            summary_lines.append(_format_search_stats_line(search_stats))
        if isinstance(stats, dict):
            summary_lines.append(
                "subgoals={n_subgoals}, primitive_steps={n_primitives}, "
                "avg_Q={avg_q}, avg_prob={avg_prob}, return={total_return}".format(
                    n_subgoals=stats["n_subgoals"],
                    n_primitives=stats["n_primitives"],
                    avg_q=_format_optional_float(stats["avg_q"]),
                    avg_prob=_format_optional_float(stats["avg_prob"]),
                    total_return=_format_optional_float(stats["total_return"]),
                )
            )
        elif not solved:
            if isinstance(failure_reason, str) and failure_reason:
                summary_lines.append(f"Reason: {failure_reason}")
            if horizon_exceeded:
                if not (
                    isinstance(failure_reason, str)
                    and "Plan horizon exceeded" in failure_reason
                ):
                    summary_lines.append(
                        f"Plan horizon exceeded: attempted {num_choices} choices "
                        f"(limit {horizon_limit})."
                    )

        if model_plan_output_dir is not None:
            # Always use .plan extension; keep per-model subfolder
            plan_filename = f"{problem_path.stem}.plan"
            plan_file = model_plan_output_dir / plan_filename
            plan_file.write_text(
                "\n".join(plan_lines) + ("\n" if plan_lines else "")
            )

        if model_summary_output_dir is not None:
            summary_file = (
                model_summary_output_dir / f"{problem_path.stem}_summary.txt"
            )
            summary_file.write_text(
                "\n".join(summary_lines) + ("\n" if summary_lines else "")
            )

        per_problem_reports.append(
            {
                "problem": problem_str,
                "summary_lines": summary_lines,
                "index": idx,
            }
        )

        if isinstance(stats, dict):
            all_stats.append({"problem": problem_str, "solved": True, **stats})
        else:
            failure_entry: dict[str, object] = {
                "problem": problem_str,
                "solved": False,
            }
            if isinstance(num_choices, int):
                failure_entry["num_choices"] = num_choices
            if horizon_exceeded:
                failure_entry["horizon_exceeded"] = True
            all_stats.append(failure_entry)

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_description(
                f"Solved {sum(1 for entry in all_stats if entry['solved'])}, failed {sum(1 for entry in all_stats if not entry['solved'])}"
            )

    # Determine worker count
    worker_count = args.num_workers
    if worker_count == -1:
        worker_count = os.cpu_count() or 1
    if worker_count < 1:
        worker_count = 1

    jobs = [
        (idx, str(path.resolve()))
        for idx, path in enumerate(problem_paths, start=1)
    ]

    # Pre-create minimal summary placeholders so that if the job is terminated
    # by the scheduler (e.g., walltime), we still leave an artifact per problem.
    if model_summary_output_dir is not None:
        for _, prob_str in jobs:
            prob_path = Path(prob_str)
            placeholder = (
                model_summary_output_dir / f"{prob_path.stem}_summary.txt"
            )
            if not placeholder.exists():
                _write_minimal_summary(
                    placeholder,
                    prob_str,
                    "Note: pending; process may be terminated before conclusion.",
                )

    # Register handlers to write an emergency summary for the in-progress problem
    atexit.register(_on_exit)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_interrupt)
        except Exception:
            pass

    if jobs:
        if worker_count == 1:
            _worker_init(
                str(args.domain),
                str(args.model),
                str(model_trace_output_dir),
                args.verbosity != "quiet",
                args.device,
                args.cpu,
                args.plan_horizon,
                args.device_strategy,
                args.search,
                args.closed_set_greedy,
                args.search_context_reset_interval,
                args.flat,
                args.iw1_precheck_add_effect_novelty,
                args.iw1_atom_first_mode,
                args.iw1_atom_first_ratio,
                args.iw1_incremental_first_applicability,
                args.iw1_incremental_first_applicability_debug_crosscheck,
                args.iw_depth_override_requested,
                args.iw_depth_mode,
                args.iw_depth_precedes_mode,
                args.iw_depth_precedes_scan,
                args.lgan_override_requested,
                args.include_lgan_edges,
                args.lgan_anchor_sources,
                args.lgan_tn_edge_pos,
                args.lgan_nn_edge_pos,
                args.lgan_rr_edge_pos,
                args.rw_return_override_requested,
                args.rw_return_feature_mode,
                args.rw_return_k,
                args.rw_return_include_depth_nodes,
                args.rw_return_lazy,
                args.rw_return_lazy_beta,
                args.rw_return_block_size,
                args.rw_return_force_block_threshold,
            )
            try:
                for job in jobs:
                    idx, prob_str = job
                    # Mark current problem so interrupts can write a minimal summary
                    if model_summary_output_dir is not None:
                        prob_path = Path(prob_str)
                        _CURRENT_PROBLEM_SUMMARY_PATH = (
                            model_summary_output_dir
                            / f"{prob_path.stem}_summary.txt"
                        )
                        _CURRENT_PROBLEM_STR = prob_str
                    idx, result = _solve_problem(job)
                    _process_result(idx, result)
                    _CURRENT_PROBLEM_SUMMARY_PATH = None
                    _CURRENT_PROBLEM_STR = None
            except KeyboardInterrupt:
                print(
                    "\nKeyboard interrupt received; stopping remaining problems."
                )
                interrupted = True
            finally:
                global _WORKER_CONTEXT
                _WORKER_CONTEXT = None
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(
                processes=worker_count,
                initializer=_worker_init,
                initargs=(
                    str(args.domain),
                    str(args.model),
                    str(model_trace_output_dir),
                    args.verbosity != "quiet",
                    args.device,
                    args.cpu,
                    args.plan_horizon,
                    args.device_strategy,
                    args.search,
                    args.closed_set_greedy,
                    args.search_context_reset_interval,
                    args.flat,
                    args.iw1_precheck_add_effect_novelty,
                    args.iw1_atom_first_mode,
                    args.iw1_atom_first_ratio,
                    args.iw1_incremental_first_applicability,
                    args.iw1_incremental_first_applicability_debug_crosscheck,
                    args.iw_depth_override_requested,
                    args.iw_depth_mode,
                    args.iw_depth_precedes_mode,
                    args.iw_depth_precedes_scan,
                    args.lgan_override_requested,
                    args.include_lgan_edges,
                    args.lgan_anchor_sources,
                    args.lgan_tn_edge_pos,
                    args.lgan_nn_edge_pos,
                    args.lgan_rr_edge_pos,
                    args.rw_return_override_requested,
                    args.rw_return_feature_mode,
                    args.rw_return_k,
                    args.rw_return_include_depth_nodes,
                    args.rw_return_lazy,
                    args.rw_return_lazy_beta,
                    args.rw_return_block_size,
                    args.rw_return_force_block_threshold,
                ),
            )
            try:
                for idx, result in pool.imap_unordered(
                    _solve_problem, jobs, chunksize=1
                ):
                    _process_result(idx, result)
            except KeyboardInterrupt:
                print(
                    "\nKeyboard interrupt received; terminating worker pool..."
                )
                interrupted = True
                pool.terminate()
                pool.join()
            else:
                pool.close()
                pool.join()

    if progress_bar is not None:
        progress_bar.close()

    if interrupted:
        processed_problems = {entry["problem"] for entry in all_stats}
        for pending_path in problem_paths:
            pending_str = str(pending_path.resolve())
            if pending_str in processed_problems:
                continue
            per_problem_reports.append(
                {
                    "problem": pending_str,
                    "summary_lines": [
                        f"Problem: {pending_str}",
                        "Solved: no (not run; interrupted before processing)",
                    ],
                }
            )
            all_stats.append({"problem": pending_str, "solved": False})
            processed_problems.add(pending_str)

    print("\n===== Per-Problem Summary =====")

    # Order final summary by input index if available, otherwise by problem path
    def _report_sort_key(rep: dict[str, object]):
        idx = rep.get("index") if isinstance(rep, dict) else None
        try:
            return (
                (0, int(idx))
                if idx is not None
                else (1, str(rep.get("problem", "")))
            )
        except Exception:
            return (1, str(rep.get("problem", "")))

    for report in sorted(per_problem_reports, key=_report_sort_key):
        for line in report["summary_lines"]:
            print(line)
        print()

    print("===== Aggregated Summary over all problems =====")
    n_total = len(all_stats)
    n_solved = sum(1 for s in all_stats if s.get("solved"))
    print(f"Solved: {n_solved}/{n_total}")

    # Gather available numeric metrics only from solved problems
    def _gather(key: str) -> list[float]:
        vals: list[float] = []
        for s in all_stats:
            if not s.get("solved"):
                continue
            v = s.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    subgoals = _gather("n_subgoals")
    primitives = _gather("n_primitives")
    avg_qs = _gather("avg_q")
    avg_probs = _gather("avg_prob")
    returns = _gather("total_return")

    print(
        f"Avg subgoals/problem (solved): {_format_optional_float(mean(subgoals) if subgoals else None)}"
    )
    print(
        f"Avg primitive steps/problem (solved): {_format_optional_float(mean(primitives) if primitives else None)}"
    )
    print(
        f"Avg of avg-Q (solved): {_format_optional_float(mean(avg_qs) if avg_qs else None)}"
    )
    print(
        f"Avg of avg-prob (solved): {_format_optional_float(mean(avg_probs) if avg_probs else None)}"
    )
    print(
        f"Avg return/problem (solved): {_format_optional_float(mean(returns) if returns else None)}"
    )

    if wandb_run is not None:
        table_columns = [
            "problem",
            "solved",
            "n_subgoals",
            "n_primitives",
            "avg_q",
            "avg_prob",
            "total_return",
        ]
        table_rows = [
            [
                entry.get("problem"),
                bool(entry.get("solved")),
                entry.get("n_subgoals"),
                entry.get("n_primitives"),
                entry.get("avg_q"),
                entry.get("avg_prob"),
                entry.get("total_return"),
            ]
            for entry in all_stats
        ]
        planning_table = wandb.Table(columns=table_columns, data=table_rows)
        wandb_run.log(
            {
                "planning/results": planning_table,
                "planning/solved_count": n_solved,
                "planning/unsolved_count": n_total - n_solved,
            }
        )

    unsolved = [
        entry["problem"] for entry in all_stats if not entry.get("solved")
    ]
    if unsolved:
        unsolved_file_path.write_text("\n".join(unsolved) + "\n")
    else:
        if unsolved_file_path.exists():
            unsolved_file_path.unlink()

    print("\n===== Unsolved Problems =====")
    if unsolved:
        for item in unsolved:
            print(item)
        print(f"Stored list of unsolved problems at {unsolved_file_path}")
    else:
        print("None")

    if interrupted:
        print("\nRun interrupted by user; partial results saved above.")

    # Persist full run summary (per-problem summaries, aggregated stats, and unsolved list)
    full_summary_lines: list[str] = []

    # Per-problem summaries (ordered)
    full_summary_lines.append("===== Per-Problem Summary =====")
    for report in sorted(per_problem_reports, key=_report_sort_key):
        # each report["summary_lines"] is a list[str]
        for line in report["summary_lines"]:
            full_summary_lines.append(line)
        full_summary_lines.append("")

    # Aggregated stats
    full_summary_lines.append(
        "===== Aggregated Summary over all problems ====="
    )
    full_summary_lines.append(f"Solved: {n_solved}/{n_total}")
    full_summary_lines.append(
        f"Avg subgoals/problem (solved): "
        f"{_format_optional_float(mean(subgoals) if subgoals else None)}"
    )
    full_summary_lines.append(
        f"Avg primitive steps/problem (solved): "
        f"{_format_optional_float(mean(primitives) if primitives else None)}"
    )
    full_summary_lines.append(
        f"Avg of avg-Q (solved): "
        f"{_format_optional_float(mean(avg_qs) if avg_qs else None)}"
    )
    full_summary_lines.append(
        f"Avg of avg-prob (solved): "
        f"{_format_optional_float(mean(avg_probs) if avg_probs else None)}"
    )
    full_summary_lines.append(
        f"Avg return/problem (solved): "
        f"{_format_optional_float(mean(returns) if returns else None)}"
    )
    full_summary_lines.append("")

    # Unsolved problems section
    full_summary_lines.append("===== Unsolved Problems =====")
    if unsolved:
        for item in unsolved:
            full_summary_lines.append(item)
        full_summary_lines.append(
            f"Stored list of unsolved problems at {unsolved_file_path}"
        )
    else:
        full_summary_lines.append("None")

    if interrupted:
        full_summary_lines.append(
            "Run interrupted by user; partial results saved above."
        )

    # Finally, write the summary file next to unsolved_problems.txt
    summary_file_path.write_text("\n".join(full_summary_lines) + "\n")
    print(f"\nSummary log written to: {summary_file_path}")

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    _main(_parse_arguments())


if __name__ == "__main__":
    main()
