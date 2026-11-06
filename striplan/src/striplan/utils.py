import pymimir as mm
import torch

from pathlib import Path


def create_device(force_cpu: bool) -> torch.device:
    if not force_cpu and torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    return device


def filter_problems(problems: list[mm.Problem], max_nr_objects: int | None) -> list[mm.Problem]:
    if max_nr_objects is None:
        return problems
    print(f'Filtering problems with more than {max_nr_objects} objects.')
    print(f'Original number of problems: {len(problems)}')
    problems = [p for p in problems if len(p.get_objects()) <= max_nr_objects]
    print(f'Filtered number of problems: {len(problems)}')
    return problems


def parse_instances(input: Path, ground: bool) -> tuple[mm.Domain, list[mm.Problem]]:
    if input.is_file():
        domain_path = str(input.parent / 'domain.pddl')
        problem_paths = [str(input)]
    else:
        domain_path = input / "domain.pddl"
        if not domain_path.exists():
            domain_path = input.parent / "domain.pddl"
        if not domain_path.exists():
            raise RuntimeError(
                f"Cannot find domain file in {input} or its parent directory."
            )
        domain_path = str(domain_path)
        problem_paths = [
            str(file) for file in input.glob("*.pddl") if file.name != "domain.pddl"
        ]
        problem_paths.sort()
    domain = mm.Domain(domain_path)
    problems = [mm.Problem(domain, problem_path, 'grounded' if ground else 'lifted') for problem_path in problem_paths]
    unsolved_problems = [problem for problem in problems if not problem.get_goal_condition().holds(problem.get_initial_state())]
    return domain, unsolved_problems


def create_device(device: int | None = 0, force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        index = device if device is not None else 0
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")
