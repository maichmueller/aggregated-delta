import pymimir as mm

from pymimir_rl import EvaluationCriteria, Trajectory


class RegretCriteria(EvaluationCriteria):
    """Calculates the regret over a set of problems.

    Args:
        problems (list[mm.Problem]): The problems to evaluate on.
        optimal_returns (list[float]): The optimal returns for each problem.
        gamma (float): The discount factor.
    """
    def __init__(self, problems: list[mm.Problem], optimal_returns: list[float], gamma: float) -> None:
        super().__init__()
        self.problems = problems
        self.optimal_returns = optimal_returns
        self.gamma = gamma

    def evaluate(self, trajectories: list[Trajectory]) -> int:
        total_regret = 0.0
        for trajectory in trajectories:
            trajectory_problem = trajectory.problem
            # Only consider trajectories that are from the problems given in the constructor
            if trajectory_problem not in self.problems:
                continue
            problem_index = self.problems.index(trajectory_problem)
            optimal_return = self.optimal_returns[problem_index]
            discounted_return = 0.0
            for i, transition in enumerate(trajectory.transitions):
                discounted_return += (self.gamma ** i) * transition.immediate_reward
            regret = optimal_return - discounted_return
            total_regret += regret
        return round(total_regret)
    
    def compare(self, x: float, y: float) -> int:
        return (x < y) - (x > y)