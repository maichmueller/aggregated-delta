import pymimir as mm

from typing import Literal, overload
from pymimir.advanced.search import (
    BeamNoveltyMode,
    GoalCountLayerOrderingStrategy,
    InOrderLayerOrderingStrategy,
    RandomizedLayerOrderingStrategy,
    ReverseOrderLayerOrderingStrategy,
)

from ._native_backend import (
    expansion_parent_edges,
    generate_iw_expansion_native,
)


def _atom_order_key(atom: mm.GroundAtom) -> tuple[int, int]:
    tag_id = 0 if atom.is_static() else 1 if atom.is_fluent() else 2
    return (
        tag_id,
        atom.get_index(),
    )


class IWAction(mm.Action):
    def __init__(self) -> None:
        self.name = "iw_action"

    def get_index(self) -> int:
        raise NotImplementedError()

    def get_arity(self) -> int:
        raise NotImplementedError()

    def get_name(self) -> str:
        return self.name

    def get_parameters(self) -> list[mm.Variable]:
        raise NotImplementedError()

    def get_precondition(self) -> mm.ConjunctiveCondition:
        raise NotImplementedError()

    def get_conditional_effect(self) -> list[mm.ConditionalEffect]:
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IWAction) and self.name == other.name


class IWGroundAction(mm.GroundAction):
    def __init__(
        self,
        start_state: mm.State,
        final_state: mm.State,
        actions: list[mm.GroundAction],
    ) -> None:
        assert isinstance(start_state, mm.State)
        assert isinstance(final_state, mm.State)
        assert isinstance(actions, list)
        assert start_state.get_problem() == final_state.get_problem()
        self.name = "iw_ground_action"
        self.start_state = start_state
        self.final_state = final_state
        self.actions = actions
        self.problem = start_state.get_problem()
        self.effect = None  # only on demand

    def get_index(self) -> int:
        raise NotImplementedError()

    def get_problem(self) -> mm.Problem:
        return self.problem

    def get_action(self) -> mm.Action:
        return IWAction()

    def get_underlying_actions(self) -> list[mm.GroundAction]:
        return self.actions

    def get_objects(self) -> list[mm.Object]:
        raise NotImplementedError()

    def get_precondition(self) -> mm.GroundConjunctiveCondition:
        raise NotImplementedError()

    def get_conditional_effect(self) -> list[mm.GroundConditionalEffect]:
        raise NotImplementedError()

    def is_applicable(self, state: mm.State) -> bool:
        assert isinstance(state, mm.State), "Invalid state type."
        assert self.get_problem() == state.get_problem(), (
            "State and action belong to different problems."
        )
        return self.start_state == state

    @overload
    def apply(self, state: mm.State) -> mm.State: ...
    @overload
    def apply(
        self, state: mm.State, return_cost: Literal[False]
    ) -> mm.State: ...
    @overload
    def apply(
        self, state: mm.State, return_cost: Literal[True]
    ) -> tuple[mm.State, float]: ...
    @overload
    def apply(
        self, state: mm.State, return_cost: bool
    ) -> mm.State | tuple[mm.State, float]: ...

    def apply(
        self, state: mm.State, return_cost: bool = False
    ) -> mm.State | tuple[mm.State, float]:
        assert isinstance(state, mm.State), "Invalid state type."
        assert state.get_problem() == self.get_problem(), (
            "State and action belong to different problems."
        )
        assert self.is_applicable(state), (
            "IW action is not applicable in the given state."
        )
        return (self.final_state, 1.0) if return_cost else self.final_state

    def atom_changes(self):
        start_atoms = set(self.start_state.get_atoms(ignore_static=True))
        final_atoms = set(self.final_state.get_atoms(ignore_static=True))
        add_atoms = final_atoms - start_atoms
        delete_atoms = start_atoms - final_atoms
        return add_atoms, delete_atoms

    def get_effect(self) -> list[mm.GroundLiteral]:
        if self.effect is None:
            # Compute effect
            add_atoms, delete_atoms = self.atom_changes()
            self.effect: list[mm.GroundLiteral] = []
            self.effect.extend(
                mm.GroundLiteral.new(atom, True, self.problem)
                for atom in sorted(add_atoms, key=_atom_order_key)
            )
            self.effect.extend(
                mm.GroundLiteral.new(atom, False, self.problem)
                for atom in sorted(delete_atoms, key=_atom_order_key)
            )
        return self.effect

    def __str__(self):
        return f"{self.name}({len(self.effect)} updates)"

    def __repr__(self):
        return f"{self.name}({len(self.effect)} updates)"

    def __hash__(self) -> int:
        return (
            hash(self.name)
            + hash(self.problem)
            + sum(hash(x) for x in self.actions)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IWGroundAction):
            return False
        if self.problem != other.problem:
            return False
        if self.start_state != other.start_state:
            return False
        if self.final_state != other.final_state:
            return False
        return self.actions == other.actions


# ---- JointIWGroundAction class ----
class JointIWGroundAction(mm.GroundAction):
    def __init__(
        self,
        prefix_action: IWGroundAction,
        continuation_action: mm.GroundAction,
    ) -> None:
        assert isinstance(prefix_action, IWGroundAction)
        assert isinstance(continuation_action, mm.GroundAction)
        self.name = "joint_iw_ground_action"
        self.prefix_action = prefix_action
        self.continuation_action = continuation_action
        self.problem = prefix_action.get_problem()
        self.start_state = prefix_action.start_state
        self.mid_state = prefix_action.final_state
        self.final_state = continuation_action.apply(self.mid_state)
        self.actions = [
            *prefix_action.get_underlying_actions(),
            continuation_action,
        ]
        self.effect = None  # only on demand

    def get_index(self) -> int:
        raise NotImplementedError()

    def get_problem(self) -> mm.Problem:
        return self.problem

    def get_action(self) -> mm.Action:
        return IWAction()

    def get_underlying_actions(self) -> list[mm.GroundAction]:
        return self.actions

    def get_objects(self) -> list[mm.Object]:
        raise NotImplementedError()

    def get_precondition(self) -> mm.GroundConjunctiveCondition:
        raise NotImplementedError()

    def get_conditional_effect(self) -> list[mm.GroundConditionalEffect]:
        raise NotImplementedError()

    def is_applicable(self, state: mm.State) -> bool:
        assert isinstance(state, mm.State), "Invalid state type."
        assert self.get_problem() == state.get_problem(), (
            "State and action belong to different problems."
        )
        return self.start_state == state

    @overload
    def apply(self, state: mm.State) -> mm.State: ...
    @overload
    def apply(
        self, state: mm.State, return_cost: Literal[False]
    ) -> mm.State: ...
    @overload
    def apply(
        self, state: mm.State, return_cost: Literal[True]
    ) -> tuple[mm.State, float]: ...
    @overload
    def apply(
        self, state: mm.State, return_cost: bool
    ) -> mm.State | tuple[mm.State, float]: ...

    def apply(
        self, state: mm.State, return_cost: bool = False
    ) -> mm.State | tuple[mm.State, float]:
        assert isinstance(state, mm.State), "Invalid state type."
        assert state.get_problem() == self.get_problem(), (
            "State and action belong to different problems."
        )
        assert self.is_applicable(state), (
            "Joint IW action is not applicable in the given state."
        )
        return (self.final_state, 1.0) if return_cost else self.final_state

    def atom_changes(self):
        start_atoms = set(self.start_state.get_atoms(ignore_static=True))
        final_atoms = set(self.final_state.get_atoms(ignore_static=True))
        add_atoms = final_atoms - start_atoms
        delete_atoms = start_atoms - final_atoms
        return add_atoms, delete_atoms

    def get_effect(self) -> list[mm.GroundLiteral]:
        if self.effect is None:
            add_atoms, delete_atoms = self.atom_changes()
            self.effect = []
            self.effect.extend(
                mm.GroundLiteral.new(atom, True, self.problem)
                for atom in sorted(add_atoms, key=_atom_order_key)
            )
            self.effect.extend(
                mm.GroundLiteral.new(atom, False, self.problem)
                for atom in sorted(delete_atoms, key=_atom_order_key)
            )
        return self.effect

    def __str__(self):
        return (
            f"{self.name}(prefix={len(self.prefix_action.get_underlying_actions())}, "
            f"total={len(self.actions)})"
        )

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return (
            hash(self.name)
            + hash(self.problem)
            + hash(self.prefix_action)
            + hash(self.continuation_action)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointIWGroundAction):
            return False
        return (
            self.problem == other.problem
            and self.prefix_action == other.prefix_action
            and self.continuation_action == other.continuation_action
        )


def _resolve_layer_ordering_strategy(name: str | None, problem: mm.Problem):
    if name is None:
        return None
    if name == "in_order":
        return InOrderLayerOrderingStrategy()
    if name == "reverse_order":
        return ReverseOrderLayerOrderingStrategy()
    if name == "randomized":
        return RandomizedLayerOrderingStrategy()
    if name == "goal_count_more":
        return GoalCountLayerOrderingStrategy(problem._advanced_problem, True)
    if name == "goal_count_less":
        return GoalCountLayerOrderingStrategy(problem._advanced_problem, False)
    raise ValueError(f"Unknown IW layer ordering strategy: {name}")


def _resolve_beam_novelty_mode(name: str) -> BeamNoveltyMode:
    if name == "all_tested":
        return BeamNoveltyMode.ALL_TESTED
    if name == "survivors_only":
        return BeamNoveltyMode.SURVIVORS_ONLY
    raise ValueError(f"Unknown IW beam novelty mode: {name}")


def generate_iw_actions(
    start_state: mm.State,
    include_start: bool = False,
    projective_iw1: bool = False,
    typed_projection: bool = False,
    abstracted_iw: bool = False,
    abstracted_iw_width: int = 1,
    base_abstracted_iw: bool = False,
    preserve_goal_atoms: bool = True,
    legacy_projective_iw1: bool = False,
    keep_depth_one_novel: bool = True,
    keep_goal_nonunary_atoms: bool = False,
    layer_ordering_strategy_name: str | None = None,
    max_next_layer_states: int = -1,
    beam_width: int = -1,
    beam_novelty_mode: str = "all_tested",
    max_depth: int = -1,
    iw1_precheck_add_effect_novelty: bool = False,
    iw1_atom_first_mode: bool = False,
    iw1_atom_first_ratio: float = 1.0,
    iw1_incremental_first_applicability: bool = False,
    iw1_incremental_first_applicability_debug_crosscheck: bool = False,
    relaxed_survivors_only_beam: bool = False,
    randomize_equal_score_ties: bool = False,
    equal_score_tie_seed: int | None = None,
    num_threads: int = -1,
    chunk_size: int = -1,
) -> tuple[list[IWGroundAction], list[tuple[int, mm.GroundAction, int]]]:
    if abstracted_iw:
        resolved_base_abstracted_iw = (
            (not typed_projection) if projective_iw1 else base_abstracted_iw
        )
        resolved_preserve_goal_atoms = (
            keep_goal_nonunary_atoms if projective_iw1 else preserve_goal_atoms
        )
        expansion = generate_iw_expansion_native(
            start_state=start_state,
            include_start=include_start,
            abstracted_iw=True,
            abstracted_iw_width=abstracted_iw_width,
            base_abstracted_iw=resolved_base_abstracted_iw,
            preserve_goal_atoms=resolved_preserve_goal_atoms,
            keep_depth_one_novel=keep_depth_one_novel,
            layer_ordering_strategy_name=layer_ordering_strategy_name,
            max_next_layer_states=max_next_layer_states,
            beam_width=beam_width,
            beam_novelty_mode=beam_novelty_mode,
            max_depth=max_depth,
            iw1_precheck_add_effect_novelty=iw1_precheck_add_effect_novelty,
            iw1_atom_first_mode=iw1_atom_first_mode,
            iw1_atom_first_ratio=iw1_atom_first_ratio,
            iw1_incremental_first_applicability=(
                iw1_incremental_first_applicability
            ),
            iw1_incremental_first_applicability_debug_crosscheck=(
                iw1_incremental_first_applicability_debug_crosscheck
            ),
            relaxed_survivors_only_beam=relaxed_survivors_only_beam,
            randomize_equal_score_ties=randomize_equal_score_ties,
            equal_score_tie_seed=equal_score_tie_seed,
            num_threads=num_threads,
            chunk_size=chunk_size,
        )
        iw_actions = list(expansion.actions_py())
        iw_edges = [
            (
                edge.src_transition_index(),
                edge.incoming_action_py(),
                edge.tgt_transition_index(),
            )
            for edge in expansion_parent_edges(expansion)
        ]
        return iw_actions, iw_edges

    # Construct the search space graph.
    predecessors: dict[mm.State, tuple[mm.GroundAction, mm.State]] = {}
    distances: dict[mm.State, int] = {start_state: 0}

    def add_transition(
        current_state: mm.State,
        action: mm.GroundAction,
        _: float,
        successor_state: mm.State,
    ) -> None:
        nonlocal predecessors, distances
        current_distance = distances[current_state]
        successor_distance = current_distance + 1
        if (successor_state not in distances) or (
            successor_distance < distances[successor_state]
        ):
            distances[successor_state] = successor_distance
            predecessors[successor_state] = (action, current_state)

    search_kwargs = {
        "layer_ordering_strategy": _resolve_layer_ordering_strategy(
            layer_ordering_strategy_name, start_state.get_problem()
        ),
        "max_next_layer_states": max_next_layer_states,
        "beam_width": beam_width,
        "beam_novelty_mode": _resolve_beam_novelty_mode(beam_novelty_mode),
        "max_depth": max_depth,
        "iw1_precheck_add_effect_novelty": iw1_precheck_add_effect_novelty,
        "iw1_atom_first_mode": iw1_atom_first_mode,
        "iw1_atom_first_ratio": iw1_atom_first_ratio,
        "iw1_incremental_first_applicability": (
            iw1_incremental_first_applicability
        ),
        "iw1_incremental_first_applicability_debug_crosscheck": (
            iw1_incremental_first_applicability_debug_crosscheck
        ),
        "relaxed_survivors_only_beam": relaxed_survivors_only_beam,
        "randomize_equal_score_ties": randomize_equal_score_ties,
        "equal_score_tie_seed": equal_score_tie_seed,
        "num_threads": num_threads,
        "chunk_size": chunk_size,
        "on_generate_new_state": add_transition,
    }

    if projective_iw1:
        mm.projective_iw(
            start_state.get_problem(),
            start_state,
            typed_projection=typed_projection,
            keep_depth_one_novel=keep_depth_one_novel,
            keep_goal_nonunary_atoms=keep_goal_nonunary_atoms,
            **search_kwargs,
        )
    else:
        mm.iw(
            start_state.get_problem(),
            start_state,
            1,
            **search_kwargs,
        )
    # For each state, backtrack to produce an IW ground action.
    if include_start:
        iw_indices: dict[mm.State, int] = {start_state: 0}
    else:
        iw_indices: dict[mm.State, int] = {}
    iw_actions: list[IWGroundAction] = []
    for final_state in predecessors.keys():
        actions: list[mm.GroundAction] = []
        state = final_state
        while state in predecessors:
            action, state = predecessors[state]
            actions.append(action)
        actions.reverse()
        assert state == start_state
        assert actions[0].is_applicable(start_state)
        iw_indices[final_state] = len(iw_indices)
        iw_actions.append(IWGroundAction(start_state, final_state, actions))
    # Construct the search tree.
    iw_edges: list[tuple[int, mm.GroundAction, int]] = []
    for state, (action, predecessor_state) in predecessors.items():
        if (predecessor_state != start_state) or include_start:
            iw_edges.append(
                (iw_indices[predecessor_state], action, iw_indices[state])
            )
    return iw_actions, iw_edges


def generate_iw_expansion(
    start_state: mm.State,
    include_start: bool = False,
    projective_iw1: bool = False,
    typed_projection: bool = False,
    abstracted_iw: bool = False,
    abstracted_iw_width: int = 1,
    base_abstracted_iw: bool = False,
    preserve_goal_atoms: bool = True,
    legacy_projective_iw1: bool = False,
    keep_depth_one_novel: bool = True,
    keep_goal_nonunary_atoms: bool = False,
    layer_ordering_strategy_name: str | None = None,
    max_next_layer_states: int = -1,
    beam_width: int = -1,
    beam_novelty_mode: str = "all_tested",
    max_depth: int = -1,
    iw1_precheck_add_effect_novelty: bool = False,
    iw1_atom_first_mode: bool = False,
    iw1_atom_first_ratio: float = 1.0,
    iw1_incremental_first_applicability: bool = False,
    iw1_incremental_first_applicability_debug_crosscheck: bool = False,
    relaxed_survivors_only_beam: bool = False,
    randomize_equal_score_ties: bool = False,
    equal_score_tie_seed: int | None = None,
    num_threads: int = -1,
    chunk_size: int = -1,
):
    return generate_iw_expansion_native(
        start_state=start_state,
        include_start=include_start,
        projective_iw1=projective_iw1,
        typed_projection=typed_projection,
        abstracted_iw=abstracted_iw,
        abstracted_iw_width=abstracted_iw_width,
        base_abstracted_iw=base_abstracted_iw,
        preserve_goal_atoms=preserve_goal_atoms,
        legacy_projective_iw1=legacy_projective_iw1,
        keep_depth_one_novel=keep_depth_one_novel,
        keep_goal_nonunary_atoms=keep_goal_nonunary_atoms,
        layer_ordering_strategy_name=layer_ordering_strategy_name,
        max_next_layer_states=max_next_layer_states,
        beam_width=beam_width,
        beam_novelty_mode=beam_novelty_mode,
        max_depth=max_depth,
        iw1_precheck_add_effect_novelty=iw1_precheck_add_effect_novelty,
        iw1_atom_first_mode=iw1_atom_first_mode,
        iw1_atom_first_ratio=iw1_atom_first_ratio,
        iw1_incremental_first_applicability=iw1_incremental_first_applicability,
        iw1_incremental_first_applicability_debug_crosscheck=(
            iw1_incremental_first_applicability_debug_crosscheck
        ),
        relaxed_survivors_only_beam=relaxed_survivors_only_beam,
        randomize_equal_score_ties=randomize_equal_score_ties,
        equal_score_tie_seed=equal_score_tie_seed,
        num_threads=num_threads,
        chunk_size=chunk_size,
    )
