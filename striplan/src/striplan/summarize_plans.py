from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-problem plan results")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help=(
            "Path to a run directory (best_*) to summarize, or to a variant/plans "
            "folder when used with --all."
        ),
    )
    parser.add_argument(
        "--problems-list",
        type=Path,
        default=None,
        help=(
            "Optional file containing absolute problem paths (one per line); when "
            "set, only include summaries for these problems."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Summarize all best_* run folders under the given path. Accepts either "
            "a variant directory containing 'plans/', or the plans/ directory itself."
        ),
    )
    return parser.parse_args(argv)


def _parse_metrics_line(line: str) -> dict[str, Optional[float | int]]:
    metrics: dict[str, Optional[float | int]] = {}
    for chunk in line.split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not value or value.lower() == "n/a":
            metrics[key] = None
            continue
        try:
            if key in {"subgoals", "primitive_steps"}:
                metrics[key] = int(float(value))
            else:
                metrics[key] = float(value)
        except ValueError:
            metrics[key] = None
    return metrics


def summarize_model_dir(
    root: Path, allow_stems: Optional[set[str]] = None
) -> tuple[int, int]:
    candidates = [
        path
        for path in root.glob("*_summary.txt")
        if path.is_file() and path.name != "planning_summary.txt"
    ]
    latest_by_stem: dict[str, Path] = {}
    for path in candidates:
        stem = path.stem.replace("_summary", "")
        previous = latest_by_stem.get(stem)
        try:
            mtime = path.stat().st_mtime
            previous_mtime = previous.stat().st_mtime if previous and previous.exists() else -1
        except Exception:
            mtime, previous_mtime = 0.0, -1
        if previous is None or mtime >= previous_mtime:
            latest_by_stem[stem] = path

    summary_files = [latest_by_stem[stem] for stem in sorted(latest_by_stem.keys())]
    per_problem_sections: list[list[str]] = []
    all_stats: list[dict[str, object]] = []

    for summary_file in summary_files:
        lines = summary_file.read_text(errors="ignore").splitlines()
        problem = None
        solved = False
        metrics_line = None
        for line in lines:
            if line.startswith("Problem:"):
                problem = line.split("Problem:", 1)[1].strip()
            elif line.startswith("Solved:"):
                solved = line.split("Solved:", 1)[1].strip().lower().startswith("yes")
            elif ("subgoals=" in line) or ("primitive_steps=" in line):
                metrics_line = line

        if not problem:
            problem = summary_file.stem.replace("_summary", "")

        if allow_stems is not None:
            try:
                problem_stem = Path(str(problem)).stem
            except Exception:
                problem_stem = str(problem)
            if problem_stem not in allow_stems:
                continue

        if lines:
            per_problem_sections.append(lines + [""])

        entry: dict[str, object] = {"problem": problem, "solved": solved}
        if solved and metrics_line:
            metrics = _parse_metrics_line(metrics_line)
            if metrics.get("subgoals") is not None:
                entry["n_subgoals"] = int(metrics["subgoals"])  # type: ignore[arg-type]
            if metrics.get("primitive_steps") is not None:
                entry["n_primitives"] = int(metrics["primitive_steps"])  # type: ignore[arg-type]
            if metrics.get("avg_Q") is not None:
                entry["avg_q"] = float(metrics["avg_Q"])  # type: ignore[arg-type]
            if metrics.get("avg_prob") is not None:
                entry["avg_prob"] = float(metrics["avg_prob"])  # type: ignore[arg-type]
            if metrics.get("return") is not None:
                entry["total_return"] = float(metrics["return"])  # type: ignore[arg-type]
        all_stats.append(entry)

    n_total = len(all_stats)
    n_solved = sum(1 for stats in all_stats if stats.get("solved"))

    def _gather(key: str) -> list[float]:
        values: list[float] = []
        for stats in all_stats:
            if not stats.get("solved"):
                continue
            value = stats.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    subgoals = _gather("n_subgoals")
    primitives = _gather("n_primitives")
    avg_qs = _gather("avg_q")
    avg_probs = _gather("avg_prob")
    returns = _gather("total_return")

    def _fmt(value: Optional[float]) -> str:
        return f"{value:.5f}" if value is not None else "n/a"

    lines: list[str] = ["===== Per-Problem Summary ====="]
    for section in per_problem_sections:
        lines.extend(section)

    lines.append("===== Aggregated Summary over all problems =====")
    lines.append(f"Solved: {n_solved}/{n_total}")
    lines.append(
        f"Avg subgoals/problem (solved): {_fmt(mean(subgoals) if subgoals else None)}"
    )
    lines.append(
        "Avg primitive steps/problem (solved): "
        f"{_fmt(mean(primitives) if primitives else None)}"
    )
    lines.append(f"Avg of avg-Q (solved): {_fmt(mean(avg_qs) if avg_qs else None)}")
    lines.append(
        f"Avg of avg-prob (solved): {_fmt(mean(avg_probs) if avg_probs else None)}"
    )
    lines.append(
        f"Avg return/problem (solved): {_fmt(mean(returns) if returns else None)}"
    )
    lines.append("")

    unsolved = [stats["problem"] for stats in all_stats if not stats.get("solved")]
    lines.append("===== Unsolved Problems =====")
    if unsolved:
        lines.extend(str(item) for item in unsolved)
    else:
        lines.append("None")

    (root / "planning_summary.txt").write_text("\n".join(lines) + "\n")
    unsolved_path = root / "unsolved_problems.txt"
    if unsolved:
        unsolved_path.write_text("\n".join(map(str, unsolved)) + "\n")
    elif unsolved_path.exists():
        unsolved_path.unlink()
    return n_solved, n_total


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    base = args.model_dir.resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {base}")

    allow_stems: set[str] | None = None
    if args.problems_list and args.problems_list.exists():
        lines = [
            line.strip() for line in args.problems_list.read_text().splitlines() if line.strip()
        ]
        allow_stems = {Path(line).stem for line in lines}

    if not args.all:
        n_solved, n_total = summarize_model_dir(base, allow_stems)
        print(f"[OK] Wrote planning_summary.txt in {base} (Solved: {n_solved}/{n_total})")
        return

    plans_dirs: list[Path] = []
    if base.name == "plans":
        plans_dirs.append(base)
    elif (base / "plans").is_dir():
        plans_dirs.append(base / "plans")
    else:
        plans_dirs.append(base)

    targets: list[Path] = []
    for plans_dir in plans_dirs:
        for directory in plans_dir.iterdir():
            if directory.is_dir() and directory.name.startswith("best_"):
                targets.append(directory)
        for run in plans_dir.iterdir():
            if not run.is_dir() or run.name == "logs":
                continue
            for directory in run.iterdir():
                if directory.is_dir() and directory.name.startswith("best_"):
                    targets.append(directory)

    if not targets:
        print(f"[WARN] No best_* directories found under {base}")
        return

    seen: set[Path] = set()
    for target in sorted(targets):
        if target in seen:
            continue
        seen.add(target)
        n_solved, n_total = summarize_model_dir(target, allow_stems)
        print(f"[OK] {target}: Solved {n_solved}/{n_total}")


if __name__ == "__main__":
    main()
