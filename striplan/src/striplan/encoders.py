from itertools import chain, repeat, product

import pymimir as mm
import pymimir_rgnn as rgnn
import pymimir_rgnn.model as rgnn_model
import torch
from pymimir import (
    GroundAtom,
    GroundLiteral,
    GroundConjunctiveCondition,
    GroundAction,
)
from pymimir_rgnn import Encoder
from pymimir_rgnn.bases import EncodedLists, EncodingContext
from pymimir_rgnn.utils import (
    get_effect_name,
    get_effect_relation_name,
    get_action_name,
    get_predicate_name,
    get_atom_name,
)

from ._native_backend import (
    NATIVE_BATCH_ENCODING_AVAILABLE,
    NATIVE_FLAT_BATCH_ENCODING_AVAILABLE,
    encode_flat_batch as native_encode_flat_batch,
    encode_transition_tree_batch as native_encode_transition_tree_batch,
    expansion_includes_root_handle,
    expansion_parent_edges,
    expansion_transitions,
    is_iw_expansion,
    is_native_iw_expansion,
    parent_edge_incoming_action,
    parent_edge_src_index,
    parent_edge_tgt_index,
    transition_depth_feature_value,
    transition_effect_literals,
    transition_index,
    transition_raw_depth,
    transition_underlying_actions,
)
from .iw import IWGroundAction
from .iw_depth import (
    IWDepthConfig,
    IW_DEPTH_INIT_SCALAR,
    IW_DEPTH_NODES,
    IW_DEPTH_PRECEDES_FACTORIZED,
    IW_DEPTH_PRECEDES_MLP,
    IW_DEPTH_PRECEDES_RELATION,
    IW_IN_DEPTH_RELATION,
)
from .lgan import (
    LGANConfig,
    LGAN_ANCHOR_ACTIONS,
    LGAN_ANCHOR_GOALS,
    LGAN_ANCHOR_STATES,
)
from .rw_return import (
    RWReturnConfig,
    build_rw_return_operator_from_relations,
    compute_rw_return_features_from_csr,
    relation_arities_from_input_spec,
)

from typing import Any, Callable, Sequence


FORCE_PYTHON_TRANSITION_TREE_BATCH_FALLBACK = False
FORCE_PYTHON_FLAT_BATCH_FALLBACK = False


class NullStateEncoder(rgnn.StateEncoder):
    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "EncodingContext",
    ) -> None:
        pass


class EnhancedStateEncoder(rgnn.StateEncoder):
    """Encoder for planning states.

    This encoder transforms a planning state (which contains atoms/facts that are
    currently true) into nodes and relations for the graph neural network.
    Objects become nodes, and atoms become relations between objects.
    """

    def __init__(
        self, suffix: str = "", add_source_state_handle: bool = False
    ) -> None:
        """Initialize the state encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), "Suffix must be a string."
        self.suffix = suffix
        self.add_source_state_handle = add_source_state_handle

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for state atoms.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for all predicates except 'number'.
        """
        ignored_predicate_names = ["number"]
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        return [
            (
                get_predicate_name(predicate, False, True, self.suffix),
                predicate.get_arity() + 1 * self.add_source_state_handle,
            )
            for predicate in predicates
        ]

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "ExtendedEncodingContext",
    ) -> None:
        assert isinstance(input_value, mm.State), (
            f"StateEncoder expected a State, got {type(input_value)}"
        )

        # Add atom relations for all atoms in the state
        if self.add_source_state_handle:
            state_handle_id = context.new_action_id(root=True)
            ext_object_indices = [state_handle_id]
        else:
            ext_object_indices = []
        for atom in input_value.get_atoms():
            relation_name = rgnn.utils.get_atom_name(
                atom, state, False, self.suffix
            )
            object_indices: list[int] = ext_object_indices + [
                context.get_object_id(obj.get_index())
                for obj in atom.get_terms()
            ]
            encoding.flattened_relations.setdefault(relation_name, []).extend(
                object_indices
            )


class ExtendedEncodingContext(EncodingContext):
    class AuxObject:
        def __init__(self, name: str, index: int):
            self.name = name
            self.index = index

        def __eq__(self, o: Any) -> bool:
            return (
                isinstance(o, ExtendedEncodingContext.AuxObject)
                and o.name == self.name
                and o.index == self.index
            )

        def get_index(self):
            return self.index

    def __init__(self, problem: mm.Problem, id_offset: int) -> None:
        super().__init__(problem, id_offset)
        objects = problem.get_objects() + problem.get_domain().get_constants()
        self._max_object_idx = max(o.get_index() for o in objects)
        self.aux_object_map = {}
        self.auxiliary_ids: dict[Any, int] = {}
        self.action_depths: list[float] = []
        self.action_depth_masks: list[bool] = []
        self.root_action_ids: list[int] = []
        self.depth_node_ids: list[int] = []
        self.depth_node_depths: list[int] = []
        self.depth_to_node_id: dict[int, int] = {}

    def get_root_action_ids(self) -> list[int]:
        return self.root_action_ids

    def _get_next_id(self) -> int:
        return (
            self.id_offset
            + len(self.object_index_to_id)
            + len(self.action_ids)
            + len(self.virtual_ids)
            + len(self.auxiliary_ids)
            + len(self.depth_node_ids)
        )

    def get_action_depths(self) -> list[float]:
        return self.action_depths

    def get_action_depth_masks(self) -> list[bool]:
        return self.action_depth_masks

    def get_depth_node_ids(self) -> list[int]:
        return self.depth_node_ids

    def get_depth_node_depths(self) -> list[int]:
        return self.depth_node_depths

    def get_depth_node_count(self) -> int:
        return len(self.depth_node_ids)

    def get_auxiliary_ids(self) -> list[int]:
        return list(self.auxiliary_ids.values())

    def get_auxiliary_count(self) -> int:
        return len(self.auxiliary_ids)

    def new_action_id(
        self,
        root: bool = False,
        depth: float = 0.0,
        depth_mask: bool = False,
    ) -> int:
        act_id = self._get_next_id()
        self.action_ids.append(act_id)
        self.action_depths.append(float(depth))
        self.action_depth_masks.append(bool(depth_mask))
        if root:
            self.root_action_ids.append(act_id)
        return act_id

    def new_virtual_id(self) -> int:
        virtual_id = self._get_next_id()
        self.virtual_ids.append(virtual_id)
        return virtual_id

    def new_or_existing_virtual_id(self) -> int:
        return self.virtual_ids[-1] if self.virtual_ids else self.new_virtual_id()

    def new_or_existing_auxiliary_id(
        self,
        key: Any = None,
        depth: float = 0.0,
        depth_mask: bool = False,
    ) -> int:
        if key is None:
            aux_id = self._get_next_id()
            self.auxiliary_ids[key] = aux_id
        else:
            aux_id = self.auxiliary_ids.get(key)
            if aux_id is None:
                aux_id = self._get_next_id()
                self.auxiliary_ids[key] = aux_id
        self.action_depths.append(float(depth))
        self.action_depth_masks.append(bool(depth_mask))
        return aux_id

    def get_or_new_object_id(self, key: Any) -> int:
        if key in self.aux_object_map:
            return self.object_index_to_id[self.aux_object_map[key].get_index()]
        object_id = (
            self.id_offset
            + len(self.object_index_to_id)
            + len(self.action_ids)
            + len(self.virtual_ids)
            + len(self.auxiliary_ids)
            + len(self.depth_node_ids)
        )
        obj_index = self._max_object_idx + 1
        self._max_object_idx += 1
        self.aux_object_map[key] = self.AuxObject(key, obj_index)
        self.object_index_to_id[obj_index] = object_id
        return object_id

    def new_or_existing_depth_node_id(self, depth: int) -> int:
        if depth in self.depth_to_node_id:
            return self.depth_to_node_id[depth]
        depth_node_id = self._get_next_id()
        self.depth_to_node_id[depth] = depth_node_id
        self.depth_node_ids.append(depth_node_id)
        self.depth_node_depths.append(depth)
        return depth_node_id

    def get_node_count(self) -> int:
        return (
            self.get_object_count()
            + self.get_action_count()
            + self.get_virtual_count()
            + self.get_auxiliary_count()
            + self.get_depth_node_count()
        )


class ExtendedEncodedLists(EncodedLists):
    def __init__(self):
        super().__init__()
        self.auxiliary_indices = []
        self.auxiliary_sizes = []
        self.root_action_indices = []
        self.action_depths = []
        self.action_depth_masks = []
        encode_parent_relations: bool = True,
        self.depth_node_indices = []
        self.depth_node_sizes = []
        self.depth_node_depths = []
        self.lgan_tn_relation_indices = []
        self.lgan_tn_entity_indices = []
        self.lgan_nn_relation_indices = []
        self.lgan_nn_entity_indices = []
        self.lgan_rr_src_relation_indices = []
        self.lgan_rr_dst_relation_indices = []
        self.encode_parent_relations = encode_parent_relations
        self.lgan_tn_sizes = []
        self.lgan_nn_sizes = []
        self.lgan_rr_sizes = []
        self.lgan_relation_source_families = {}


class ExtendedEncodedTensors(rgnn.bases.EncodedTensors):
    def __init__(self):
        super().__init__()
        self.auxiliary_indices = torch.IntTensor()
        self.auxiliary_sizes = torch.IntTensor()
        self.root_action_indices = torch.IntTensor()
        self.action_depths = torch.FloatTensor()
        self.action_depth_mask = torch.BoolTensor()
        self.depth_node_indices = torch.IntTensor()
        self.depth_node_sizes = torch.IntTensor()
        self.depth_node_depths = torch.IntTensor()
        self.lgan_tn_relation_indices = torch.IntTensor()
        self.lgan_tn_entity_indices = torch.IntTensor()
        self.lgan_nn_relation_indices = torch.IntTensor()
        self.lgan_nn_entity_indices = torch.IntTensor()
        self.lgan_rr_src_relation_indices = torch.IntTensor()
        self.lgan_rr_dst_relation_indices = torch.IntTensor()
        self.lgan_tn_sizes = torch.IntTensor()
        self.lgan_nn_sizes = torch.IntTensor()
        self.lgan_rr_sizes = torch.IntTensor()
        self.rw_return_features = torch.FloatTensor()


def _transition_tree_encoder_from_input_spec(
    input_specification: tuple[Encoder, ...],
) -> "TransitionTreeEncoder | None":
    for encoder in input_specification:
        if isinstance(encoder, TransitionTreeEncoder):
            return encoder
    return None


def _attach_python_rw_return_features(
    *,
    encoding_tensors: ExtendedEncodedTensors,
    input: list[tuple],
    input_specification: tuple[Encoder, ...],
) -> None:
    transition_encoder = _transition_tree_encoder_from_input_spec(
        input_specification
    )
    if transition_encoder is None:
        return
    if not transition_encoder.rw_return_config.enabled:
        return
    if not input:
        encoding_tensors.rw_return_features = torch.empty(
            (encoding_tensors.node_count, 0),
            dtype=torch.float,
            device=encoding_tensors.node_sizes.device,
        )
        return
    first_state = input[0][0]
    assert isinstance(first_state, mm.State)
    relation_arities = relation_arities_from_input_spec(
        input_specification, first_state.get_problem().get_domain()
    )
    row_ptr, col_ind, values, kept_to_global = (
        build_rw_return_operator_from_relations(
            num_total_nodes=encoding_tensors.node_count,
            flattened_relations=encoding_tensors.flattened_relations,
            relation_arities=relation_arities,
            object_indices=encoding_tensors.object_indices,
            action_indices=encoding_tensors.action_indices,
            root_action_indices=encoding_tensors.root_action_indices,
            auxiliary_indices=encoding_tensors.auxiliary_indices,
            depth_node_indices=encoding_tensors.depth_node_indices,
            separate_action_nodes=transition_encoder.separate_action_nodes,
            config=transition_encoder.rw_return_config,
        )
    )
    with torch.no_grad():
        encoding_tensors.rw_return_features = (
            compute_rw_return_features_from_csr(
                row_ptr=row_ptr,
                col_ind=col_ind,
                values=values,
                kept_to_global=kept_to_global,
                num_total_nodes=encoding_tensors.node_count,
                config=transition_encoder.rw_return_config,
                device=encoding_tensors.node_sizes.device,
            )
        )


def _ordered_relation_names_from_input_spec(
    input_specification: Sequence[Any],
    domain: mm.Domain,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for encoder in input_specification:
        get_relations = getattr(encoder, "get_relations", None)
        if not callable(get_relations):
            continue
        for relation_name, _ in get_relations(domain):
            if relation_name in seen:
                continue
            seen.add(relation_name)
            ordered.append(relation_name)
    return ordered


def _tree_lgan_relation_source_families(
    input_specification: tuple[Encoder, ...],
    domain: mm.Domain,
) -> dict[str, frozenset[str]]:
    relation_sources: dict[str, set[str]] = {}

    if input_specification:
        state_get_relations = getattr(input_specification[0], "get_relations", None)
        if callable(state_get_relations):
            for relation_name, _ in state_get_relations(domain):
                relation_sources.setdefault(relation_name, set()).add(
                    LGAN_ANCHOR_STATES
                )

        goal_get_relations = getattr(input_specification[-1], "get_relations", None)
        if callable(goal_get_relations):
            for relation_name, _ in goal_get_relations(domain):
                relation_sources.setdefault(relation_name, set()).add(
                    LGAN_ANCHOR_GOALS
                )

    transition_encoder = _transition_tree_encoder_from_input_spec(
        input_specification
    )
    if transition_encoder is not None:
        ignored_predicate_names = {"number"}
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        for positive in (True, False):
            for predicate in predicates:
                relation_sources.setdefault(
                    get_effect_name(
                        predicate,
                        positive,
                        True,
                        transition_encoder.suffix,
                    ),
                    set(),
                ).add(LGAN_ANCHOR_GOALS)
        if transition_encoder.encode_shared_relations:
            for predicate in predicates:
                relation_sources.setdefault(
                    get_shared_atom_name(
                        predicate,
                        transition_encoder.shared_label,
                        transition_encoder.suffix,
                    ),
                    set(),
                ).add(LGAN_ANCHOR_STATES)

    return {
        relation_name: frozenset(sources)
        for relation_name, sources in relation_sources.items()
    }


def _transition_tree_lgan_row_source_families(
    transition_encoder: "TransitionTreeEncoder",
    domain: mm.Domain,
) -> dict[str, frozenset[str]]:
    relation_sources: dict[str, frozenset[str]] = {}
    ignored_predicate_names = {"number"}
    predicates = [
        predicate
        for predicate in domain.get_predicates()
        if predicate.get_name() not in ignored_predicate_names
    ]
    for positive in (True, False):
        for predicate in predicates:
            relation_sources[
                get_effect_name(
                    predicate,
                    positive,
                    True,
                    transition_encoder.suffix,
                )
            ] = frozenset({LGAN_ANCHOR_GOALS})
    if transition_encoder.encode_shared_relations:
        for predicate in predicates:
            relation_sources[
                get_shared_atom_name(
                    predicate,
                    transition_encoder.shared_label,
                    transition_encoder.suffix,
                )
            ] = frozenset({LGAN_ANCHOR_STATES})
    return relation_sources


def _row_source_families_for_encoder(
    *,
    encoder_index: int,
    encoder: Encoder,
    input_specification: tuple[Encoder, ...],
    domain: mm.Domain,
) -> dict[str, frozenset[str]]:
    get_relations = getattr(encoder, "get_relations", None)
    if not callable(get_relations):
        return {}
    if encoder_index == 0:
        return {
            relation_name: frozenset({LGAN_ANCHOR_STATES})
            for relation_name, _ in get_relations(domain)
        }
    if encoder_index == len(input_specification) - 1:
        return {
            relation_name: frozenset({LGAN_ANCHOR_GOALS})
            for relation_name, _ in get_relations(domain)
        }
    if isinstance(encoder, TransitionTreeEncoder):
        return _transition_tree_lgan_row_source_families(encoder, domain)
    return {}


def _record_sample_lgan_relation_sources(
    *,
    sample_encoding: ExtendedEncodedLists,
    before_relation_lengths: dict[str, int],
    relation_arities: dict[str, int],
    row_source_families: dict[str, frozenset[str]],
) -> None:
    for relation_name, relation_values in sample_encoding.flattened_relations.items():
        before_length = before_relation_lengths.get(relation_name, 0)
        added_length = len(relation_values) - before_length
        if added_length <= 0:
            continue
        arity = relation_arities[relation_name]
        if added_length % arity != 0:
            raise RuntimeError(
                "LGAN source-family provenance requires relation rows to be divisible by arity, "
                f"got relation={relation_name!r}, added_length={added_length}, arity={arity}."
            )
        row_count = added_length // arity
        source_family = row_source_families.get(relation_name, frozenset())
        sample_encoding.lgan_relation_source_families.setdefault(
            relation_name, []
        ).extend([source_family] * row_count)


def _validate_tree_lgan_anchor_sources(lgan_config: LGANConfig) -> None:
    allowed_sources = {
        LGAN_ANCHOR_ACTIONS,
        LGAN_ANCHOR_GOALS,
        LGAN_ANCHOR_STATES,
    }
    unsupported = [
        source
        for source in lgan_config.lgan_anchor_sources
        if source not in allowed_sources
    ]
    if unsupported:
        raise RuntimeError(
            "Transition-tree LGAN currently supports only actions, goals, and states "
            f"anchor sources, got unsupported={unsupported!r}."
        )


def _tree_lgan_anchor_node_sets(
    context: ExtendedEncodingContext,
) -> tuple[set[int], set[int]]:
    root_action_ids = set(context.get_root_action_ids())
    action_anchor_ids = set(context.get_action_ids()) - root_action_ids
    state_anchor_ids = set(context.get_auxiliary_ids()) | root_action_ids
    return action_anchor_ids, state_anchor_ids


def _tree_lgan_slot_anchor_mask(
    row: Sequence[int],
    *,
    source_families: frozenset[str],
    selected_sources: set[str],
    action_anchor_ids: set[int],
    state_anchor_ids: set[int],
) -> list[bool]:
    anchor_mask: list[bool] = []
    explicit_state_anchor = False
    for entity_id in row:
        is_anchor = False
        if (
            LGAN_ANCHOR_ACTIONS in selected_sources
            and entity_id in action_anchor_ids
        ):
            is_anchor = True
        if (
            LGAN_ANCHOR_STATES in selected_sources
            and entity_id in state_anchor_ids
        ):
            is_anchor = True
            explicit_state_anchor = True
        anchor_mask.append(is_anchor)
    if (
        LGAN_ANCHOR_GOALS in selected_sources
        and LGAN_ANCHOR_GOALS in source_families
    ):
        return [True] * len(row)
    if (
        LGAN_ANCHOR_STATES in selected_sources
        and LGAN_ANCHOR_STATES in source_families
        and not explicit_state_anchor
    ):
        return [True] * len(row)
    return anchor_mask


def _append_python_tree_lgan_features(
    *,
    encoding_lists: ExtendedEncodedLists,
    sample_encoding: ExtendedEncodedLists,
    context: ExtendedEncodingContext,
    input_specification: tuple[Encoder, ...],
    state: mm.State,
) -> None:
    transition_encoder = _transition_tree_encoder_from_input_spec(
        input_specification
    )
    if transition_encoder is None or not transition_encoder.lgan_config.enabled:
        encoding_lists.lgan_tn_sizes.append(0)
        encoding_lists.lgan_nn_sizes.append(0)
        encoding_lists.lgan_rr_sizes.append(0)
        return

    _validate_tree_lgan_anchor_sources(transition_encoder.lgan_config)

    domain = state.get_problem().get_domain()
    relation_arities = relation_arities_from_input_spec(
        input_specification, domain
    )
    ordered_relation_names = _ordered_relation_names_from_input_spec(
        input_specification, domain
    )
    declared_relation_names = set(ordered_relation_names)
    unexpected_relations = [
        relation_name
        for relation_name in sample_encoding.flattened_relations
        if relation_name not in declared_relation_names
    ]
    if unexpected_relations:
        raise RuntimeError(
            "LGAN relation-instance ordering requires all emitted relations to be "
            f"declared by the input spec, got unexpected={unexpected_relations!r}."
        )

    relation_sources = _tree_lgan_relation_source_families(
        input_specification, domain
    )
    selected_sources = set(transition_encoder.lgan_config.lgan_anchor_sources)
    action_anchor_ids, state_anchor_ids = _tree_lgan_anchor_node_sets(context)

    relation_index = 0
    local_tn_count = 0
    local_nn_count = 0
    local_rr_count = 0
    anchor_to_relations: dict[int, list[int]] = {}
    for relation_name in ordered_relation_names:
        relation_values = sample_encoding.flattened_relations.get(relation_name)
        if not relation_values:
            continue
        arity = relation_arities[relation_name]
        if arity <= 0:
            raise RuntimeError(
                f"LGAN requires positive relation arities, got {arity} for "
                f"relation={relation_name!r}."
            )
        if len(relation_values) % arity != 0:
            raise RuntimeError(
                "LGAN canonical relation-instance ordering requires relation rows to "
                f"be divisible by arity, got relation={relation_name!r}, "
                f"len={len(relation_values)}, arity={arity}."
            )
        row_source_families = sample_encoding.lgan_relation_source_families.get(
            relation_name, []
        )
        for row_start in range(0, len(relation_values), arity):
            row_index = row_start // arity
            row = [
                int(value)
                for value in relation_values[row_start : row_start + arity]
            ]
            source_families = (
                row_source_families[row_index]
                if row_index < len(row_source_families)
                else relation_sources.get(relation_name, frozenset())
            )
            anchor_mask = _tree_lgan_slot_anchor_mask(
                row,
                source_families=source_families,
                selected_sources=selected_sources,
                action_anchor_ids=action_anchor_ids,
                state_anchor_ids=state_anchor_ids,
            )
            anchor_entities = [
                entity_id
                for entity_id, is_anchor in zip(row, anchor_mask, strict=True)
                if is_anchor
            ]
            neighbor_entities = [
                entity_id
                for entity_id, is_anchor in zip(row, anchor_mask, strict=True)
                if not is_anchor
            ]
            if anchor_entities:
                encoding_lists.lgan_tn_relation_indices.extend(
                    [relation_index] * len(anchor_entities)
                )
                encoding_lists.lgan_tn_entity_indices.extend(anchor_entities)
                encoding_lists.lgan_nn_relation_indices.extend(
                    [relation_index] * len(neighbor_entities)
                )
                encoding_lists.lgan_nn_entity_indices.extend(neighbor_entities)
                local_tn_count += len(anchor_entities)
                local_nn_count += len(neighbor_entities)
                for anchor_id in dict.fromkeys(anchor_entities):
                    anchor_to_relations.setdefault(anchor_id, []).append(
                        relation_index
                    )
            relation_index += 1

    for relation_indices in anchor_to_relations.values():
        if len(relation_indices) < 2:
            continue
        for src_relation_index in relation_indices:
            for dst_relation_index in relation_indices:
                if src_relation_index == dst_relation_index:
                    continue
                encoding_lists.lgan_rr_src_relation_indices.append(
                    src_relation_index
                )
                encoding_lists.lgan_rr_dst_relation_indices.append(
                    dst_relation_index
                )
                local_rr_count += 1

    encoding_lists.lgan_tn_sizes.append(local_tn_count)
    encoding_lists.lgan_nn_sizes.append(local_nn_count)
    encoding_lists.lgan_rr_sizes.append(local_rr_count)


def _can_use_native_transition_tree_batch(
    input: list[tuple],
    input_specification: tuple[Encoder, ...],
) -> bool:
    if FORCE_PYTHON_TRANSITION_TREE_BATCH_FALLBACK:
        return False
    if not NATIVE_BATCH_ENCODING_AVAILABLE:
        return False
    if len(input_specification) != 3:
        return False
    state_encoder, transition_encoder, goal_encoder = input_specification
    if not isinstance(transition_encoder, TransitionTreeEncoder):
        return False
    if not isinstance(goal_encoder, rgnn.GoalEncoder):
        return False
    if isinstance(state_encoder, EnhancedStateEncoder):
        if state_encoder.add_source_state_handle != (
            transition_encoder.connect_to_root_handle
        ):
            return False
    elif isinstance(state_encoder, rgnn.StateEncoder):
        if transition_encoder.connect_to_root_handle:
            return False
    else:
        return False
    for instance in input:
        if not isinstance(instance, tuple) or len(instance) != 3:
            return False
        if not isinstance(instance[0], mm.State):
            return False
        if not is_native_iw_expansion(instance[1]):
            return False
        if not isinstance(instance[2], mm.GroundConjunctiveCondition):
            return False
    return True


def _requires_native_transition_tree_batch(
    input_specification: tuple[Encoder, ...],
) -> bool:
    if FORCE_PYTHON_TRANSITION_TREE_BATCH_FALLBACK:
        return False
    if len(input_specification) != 3:
        return False
    state_encoder, transition_encoder, goal_encoder = input_specification
    if not isinstance(transition_encoder, TransitionTreeEncoder):
        return False
    if not isinstance(goal_encoder, rgnn.GoalEncoder):
        return False
    return isinstance(state_encoder, (EnhancedStateEncoder, rgnn.StateEncoder))


def _native_flat_batch_layout(
    input_specification: tuple[Encoder, ...],
) -> tuple[str, list[int], list[str], int, str, int, str, int, str] | None:
    mode = "objects"
    state_positions: list[int] = []
    state_suffixes: list[str] = []
    goal_position = -1
    goal_suffix = ""
    effects_position = -1
    effects_suffix = ""
    actions_position = -1
    actions_suffix = ""

    for encoder_index, encoder in enumerate(input_specification):
        if type(encoder) is rgnn.StateEncoder:
            state_positions.append(encoder_index)
            state_suffixes.append(getattr(encoder, "suffix", ""))
            continue
        if isinstance(encoder, rgnn.GoalEncoder):
            if goal_position != -1:
                return None
            goal_position = encoder_index
            goal_suffix = getattr(encoder, "suffix", "")
            continue
        if isinstance(encoder, EffectsEncoder):
            if effects_position != -1:
                return None
            if (
                encoder.encode_shared_relations
                or encoder.add_target_state_handle
                or encoder.connect_source_and_target_handle
            ):
                return None
            effects_position = encoder_index
            effects_suffix = encoder.suffix
            continue
        if isinstance(encoder, rgnn.GroundActionsEncoder):
            if actions_position != -1 or effects_position != -1:
                return None
            mode = "ground_actions"
            actions_position = encoder_index
            actions_suffix = getattr(encoder, "suffix", "")
            continue
        if isinstance(encoder, rgnn.TransitionEffectsEncoder):
            if actions_position != -1 or effects_position != -1:
                return None
            mode = "transition_effects"
            effects_position = encoder_index
            effects_suffix = getattr(encoder, "suffix", "")
            continue
        return None

    if not state_positions or goal_position == -1:
        return None
    return (
        mode,
        state_positions,
        state_suffixes,
        goal_position,
        goal_suffix,
        effects_position,
        effects_suffix,
        actions_position,
        actions_suffix,
    )


def _requires_native_flat_batch(
    input_specification: tuple[Encoder, ...],
) -> bool:
    if FORCE_PYTHON_FLAT_BATCH_FALLBACK:
        return False
    return _native_flat_batch_layout(input_specification) is not None


def _can_use_native_flat_batch(
    input: list[tuple],
    input_specification: tuple[Encoder, ...],
    layout: tuple[str, list[int], list[str], int, str, int, str, int, str] | None = None,
) -> bool:
    if FORCE_PYTHON_FLAT_BATCH_FALLBACK:
        return False
    if not NATIVE_FLAT_BATCH_ENCODING_AVAILABLE:
        return False
    if layout is None:
        layout = _native_flat_batch_layout(input_specification)
    if layout is None:
        return False

    (
        mode,
        state_positions,
        _,
        goal_position,
        _,
        effects_position,
        _,
        actions_position,
        _,
    ) = layout
    for instance in input:
        if not isinstance(instance, tuple) or len(instance) != len(input_specification):
            return False
        for state_position in state_positions:
            if not isinstance(instance[state_position], mm.State):
                return False
        if not isinstance(instance[goal_position], mm.GroundConjunctiveCondition):
            return False
        if mode == "ground_actions":
            actions_value = instance[actions_position]
            if not isinstance(actions_value, list):
                return False
            if any(not isinstance(action, mm.GroundAction) for action in actions_value):
                return False
        elif effects_position != -1:
            effects_value = instance[effects_position]
            if isinstance(effects_value, tuple):
                if len(effects_value) == 0:
                    return False
                effects_value = effects_value[0]
            if not isinstance(effects_value, (list, tuple)):
                return False
    return True


def _encode_depth_node_structure(
    *,
    context: ExtendedEncodingContext,
    encoding: EncodedLists,
    depth_config: IWDepthConfig,
    realized_depths: list[int],
    attachments: list[tuple[int, int]],
) -> None:
    if not depth_config.uses_depth_nodes:
        return
    sorted_depths = sorted(set(realized_depths))
    if not sorted_depths:
        return
    depth_node_by_depth = {
        depth: context.new_or_existing_depth_node_id(depth)
        for depth in sorted_depths
    }
    for depth, node_id in attachments:
        encoding.flattened_relations.setdefault(IW_IN_DEPTH_RELATION, []).extend(
            [node_id, depth_node_by_depth[depth]]
        )
    if depth_config.precedes_mode == IW_DEPTH_PRECEDES_MLP:
        for depth_index, source_depth in enumerate(sorted_depths):
            source_depth_id = depth_node_by_depth[source_depth]
            for target_depth in sorted_depths[depth_index + 1 :]:
                encoding.flattened_relations.setdefault(
                    IW_DEPTH_PRECEDES_RELATION, []
                ).extend([source_depth_id, depth_node_by_depth[target_depth]])


def get_input_from_encoders_extended(
    input: list[tuple],
    input_specification: tuple[Encoder, ...],
    device: torch.device,
    filter_out_root_action_ids: bool = False,
) -> rgnn.encoders.EncodedTensors:
    """Encode input using a collection of encoders.

    This function processes a batch of input instances using the provided
    encoder specification and returns the encoded graph representation
    ready for use in the graph neural network.

    Args:
        input: List of input tuples, where each tuple contains the inputs
              corresponding to the encoder specification.
        input_specification: Tuple of encoder instances that define how to
                            process each element of the input tuples.
        device: The torch device to place the resulting tensors on.

    Returns:
        EncodedTensors object containing the graph representation.

    Raises:
        AssertionError: If input format doesn't match specification or if
                       no StateEncoder is found in the specification.
    """
    if _requires_native_transition_tree_batch(input_specification):
        if not NATIVE_BATCH_ENCODING_AVAILABLE:
            raise RuntimeError(
                "TransitionTreeEncoder requires the native striplan batch encoder, "
                "but the native backend is unavailable."
            )
    if _can_use_native_transition_tree_batch(input, input_specification):
        return native_encode_transition_tree_batch(
            batch_instances=input,  # type: ignore[arg-type]
            tree_encoder=input_specification[1],
            device=device,
            filter_out_root_action_ids=filter_out_root_action_ids,
        )

    flat_batch_layout = _native_flat_batch_layout(input_specification)
    if _requires_native_flat_batch(input_specification):
        if not NATIVE_FLAT_BATCH_ENCODING_AVAILABLE:
            raise RuntimeError(
                "The requested flat encoder layout requires the native striplan "
                "flat batch encoder, but the native backend is unavailable."
            )
    if _can_use_native_flat_batch(
        input, input_specification, flat_batch_layout
    ):
        (
            mode,
            state_positions,
            state_suffixes,
            goal_position,
            goal_suffix,
            effects_position,
            effects_suffix,
            actions_position,
            actions_suffix,
        ) = flat_batch_layout
        return native_encode_flat_batch(
            batch_instances=input,
            mode=mode,
            state_positions=state_positions,
            state_suffixes=state_suffixes,
            goal_position=goal_position,
            goal_suffix=goal_suffix,
            effects_position=effects_position,
            effects_suffix=effects_suffix,
            actions_position=actions_position,
            actions_suffix=actions_suffix,
            device=device,
        )

    encoding_lists = ExtendedEncodedLists()

    # Process each input instance
    for instance in input:
        assert isinstance(instance, tuple), "Input instance must be a tuple."
        assert len(instance) == len(input_specification), (
            "Mismatch between the length of an input instance and the input specification."
        )

        # Find the state encoder to get the state
        state = None
        for i, encoder in enumerate(input_specification):
            if isinstance(encoder, DifferenceViaObjectsStateEncoder):
                state = instance[i][0]
                break
            if isinstance(encoder, DifferenceViaRelationsStateEncoder):
                state = instance[i][0]
                break
            if isinstance(encoder, rgnn.StateEncoder):
                state = instance[i]
                assert isinstance(state, mm.State), (
                    f"Expected a State at position {i}, got {type(state)}"
                )
                break

        assert state is not None, (
            "Input specification must contain a StateEncoder."
        )

        # Track nodes added for this instance.
        sample_encoding = ExtendedEncodedLists()
        context = ExtendedEncodingContext(
            state.get_problem(), encoding_lists.node_count
        )
        relation_arities = relation_arities_from_input_spec(
            input_specification, state.get_problem().get_domain()
        )

        # Process each encoder with its corresponding input value
        for encoder_index, encoder in enumerate(input_specification):
            input_value = instance[encoder_index]
            before_relation_lengths = {
                relation_name: len(relation_values)
                for relation_name, relation_values in sample_encoding.flattened_relations.items()
            }
            encoder.encode(input_value, state, sample_encoding, context)
            _record_sample_lgan_relation_sources(
                sample_encoding=sample_encoding,
                before_relation_lengths=before_relation_lengths,
                relation_arities=relation_arities,
                row_source_families=_row_source_families_for_encoder(
                    encoder_index=encoder_index,
                    encoder=encoder,
                    input_specification=input_specification,
                    domain=state.get_problem().get_domain(),
                ),
            )

        _append_python_tree_lgan_features(
            encoding_lists=encoding_lists,
            sample_encoding=sample_encoding,
            context=context,
            input_specification=input_specification,
            state=state,
        )

        for relation_name, relation_values in sample_encoding.flattened_relations.items():
            encoding_lists.flattened_relations.setdefault(
                relation_name, []
            ).extend(relation_values)

        # Update global encoding with instance results
        encoding_lists.object_indices.extend(context.get_object_ids())
        encoding_lists.action_indices.extend(context.get_action_ids())
        encoding_lists.root_action_indices.extend(context.get_root_action_ids())
        encoding_lists.auxiliary_indices.extend(context.get_auxiliary_ids())
        encoding_lists.virtual_indices.extend(context.get_virtual_ids())
        encoding_lists.depth_node_indices.extend(context.get_depth_node_ids())
        encoding_lists.depth_node_depths.extend(
            context.get_depth_node_depths()
        )
        encoding_lists.action_depths.extend(context.get_action_depths())
        encoding_lists.action_depth_masks.extend(
            context.get_action_depth_masks()
        )
        encoding_lists.object_sizes.append(context.get_object_count())
        encoding_lists.action_sizes.append(context.get_action_count())
        encoding_lists.auxiliary_sizes.append(context.get_auxiliary_count())
        encoding_lists.virtual_sizes.append(context.get_virtual_count())
        encoding_lists.depth_node_sizes.append(context.get_depth_node_count())
        encoding_lists.node_sizes.append(context.get_node_count())
        encoding_lists.node_count += context.get_node_count()

    if filter_out_root_action_ids:
        root_ids = set(encoding_lists.root_action_indices)
        filtered_action_data = [
            (index, depth, depth_mask)
            for index, depth, depth_mask in zip(
                encoding_lists.action_indices,
                encoding_lists.action_depths,
                encoding_lists.action_depth_masks,
            )
            if index not in root_ids
        ]
        encoding_lists.action_indices = [
            index for index, _, _ in filtered_action_data
        ]
        encoding_lists.action_depths = [
            depth for _, depth, _ in filtered_action_data
        ]
        encoding_lists.action_depth_masks = [
            depth_mask for _, _, depth_mask in filtered_action_data
        ]
        encoding_lists.action_sizes = [
            size - 1 for size in encoding_lists.action_sizes
        ]

    # Convert the lists to tensors on the correct device
    encoding_tensors = ExtendedEncodedTensors()
    encoding_tensors.flattened_relations = rgnn.utils.relations_to_tensors(
        encoding_lists.flattened_relations, device
    )
    encoding_tensors.node_count = encoding_lists.node_count
    encoding_tensors.node_sizes = torch.tensor(
        encoding_lists.node_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.object_indices = torch.tensor(
        encoding_lists.object_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.object_sizes = torch.tensor(
        encoding_lists.object_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.action_indices = torch.tensor(
        encoding_lists.action_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.root_action_indices = torch.tensor(
        encoding_lists.root_action_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.action_sizes = torch.tensor(
        encoding_lists.action_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.action_depths = torch.tensor(
        encoding_lists.action_depths,
        dtype=torch.float,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.action_depth_mask = torch.tensor(
        encoding_lists.action_depth_masks,
        dtype=torch.bool,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.auxiliary_indices = torch.tensor(
        encoding_lists.auxiliary_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.auxiliary_sizes = torch.tensor(
        encoding_lists.auxiliary_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.virtual_indices = torch.tensor(
        encoding_lists.virtual_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.virtual_sizes = torch.tensor(
        encoding_lists.virtual_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.depth_node_indices = torch.tensor(
        encoding_lists.depth_node_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.depth_node_sizes = torch.tensor(
        encoding_lists.depth_node_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.depth_node_depths = torch.tensor(
        encoding_lists.depth_node_depths,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_tn_relation_indices = torch.tensor(
        encoding_lists.lgan_tn_relation_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_tn_entity_indices = torch.tensor(
        encoding_lists.lgan_tn_entity_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_nn_relation_indices = torch.tensor(
        encoding_lists.lgan_nn_relation_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_nn_entity_indices = torch.tensor(
        encoding_lists.lgan_nn_entity_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_rr_src_relation_indices = torch.tensor(
        encoding_lists.lgan_rr_src_relation_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_rr_dst_relation_indices = torch.tensor(
        encoding_lists.lgan_rr_dst_relation_indices,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_tn_sizes = torch.tensor(
        encoding_lists.lgan_tn_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_nn_sizes = torch.tensor(
        encoding_lists.lgan_nn_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    encoding_tensors.lgan_rr_sizes = torch.tensor(
        encoding_lists.lgan_rr_sizes,
        dtype=torch.int,
        device=device,
        requires_grad=False,
    )
    _attach_python_rw_return_features(
        encoding_tensors=encoding_tensors,
        input=input,
        input_specification=input_specification,
    )
    return encoding_tensors


class ExtendedRGNN(rgnn.RelationalGraphNeuralNetwork):
    def forward(self, x: list[tuple], **kwargs) -> rgnn.ForwardState:  # type: ignore
        """Perform a forward pass through the R-GNN.

        Args:
            x: List of input tuples, where each tuple contains the inputs specified
               by the input_spec (e.g., state, goal, actions).

        Returns:
            ForwardState object containing the final layer index and readout functions
            for extracting outputs.
        """
        assert isinstance(x, list), "Expected input to be a list."
        input = get_input_from_encoders_extended(
            x, self._input_spec, self.get_device(), **kwargs
        )
        return self._internal_forward(input)

    def clone(self):
        model_clone = type(self)(
            self._hparam_config,
            self._module_config,
            self._input_spec,
            self._output_spec,
        )
        model_clone.load_state_dict(self.state_dict())
        return model_clone.to(self.get_device())


class FactorizedDepthPrecedesRelation(torch.nn.Module):
    MAXIMUM_SMOOTHNESS = 12.0

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self._embedding_size = embedding_size
        self._source_projection = torch.nn.Linear(
            embedding_size, embedding_size, bias=False
        )
        self._target_projection = torch.nn.Linear(
            embedding_size, embedding_size, bias=False
        )

    def explicit_messages(
        self,
        node_embeddings: torch.Tensor,
        input: ExtendedEncodedTensors,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input.depth_node_indices.numel() == 0:
            return (
                torch.empty(
                    (0, self._embedding_size),
                    dtype=node_embeddings.dtype,
                    device=node_embeddings.device,
                ),
                torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=node_embeddings.device,
                ),
            )
        depth_node_indices = input.depth_node_indices.long()
        depth_embeddings = node_embeddings.index_select(0, depth_node_indices)
        source_terms = self._source_projection(depth_embeddings)
        target_terms = self._target_projection(depth_embeddings)
        message_parts: list[torch.Tensor] = []
        index_parts: list[torch.Tensor] = []
        offset = 0
        for size in input.depth_node_sizes.tolist():
            if size > 1:
                sample_source = source_terms[offset : offset + size]
                sample_target = target_terms[offset : offset + size]
                sample_indices = depth_node_indices[offset : offset + size]
                src_positions, tgt_positions = torch.triu_indices(
                    size, size, offset=1, device=node_embeddings.device
                )
                if src_positions.numel() > 0:
                    message_parts.append(
                        sample_source.index_select(0, src_positions)
                        + sample_target.index_select(0, tgt_positions)
                    )
                    index_parts.append(
                        sample_indices.index_select(0, tgt_positions)
                    )
            offset += size
        if not message_parts:
            return (
                torch.empty(
                    (0, self._embedding_size),
                    dtype=node_embeddings.dtype,
                    device=node_embeddings.device,
                ),
                torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=node_embeddings.device,
                ),
            )
        return torch.cat(message_parts, dim=0), torch.cat(index_parts, dim=0)

    def scan_smax_summary(
        self,
        node_embeddings: torch.Tensor,
        input: ExtendedEncodedTensors,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summary_max = torch.zeros_like(node_embeddings)
        summary_sum = torch.zeros_like(node_embeddings)
        if input.depth_node_indices.numel() == 0:
            return summary_max, summary_sum
        depth_node_indices = input.depth_node_indices.long()
        depth_embeddings = node_embeddings.index_select(0, depth_node_indices)
        source_terms = self._source_projection(depth_embeddings)
        target_terms = self._target_projection(depth_embeddings)
        smoothness = self.MAXIMUM_SMOOTHNESS
        offset = 0
        for size in input.depth_node_sizes.tolist():
            sample_source = source_terms[offset : offset + size]
            sample_target = target_terms[offset : offset + size]
            sample_indices = depth_node_indices[offset : offset + size]
            running_max = torch.zeros(
                (self._embedding_size,),
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            )
            running_sum = torch.zeros_like(running_max)
            have_prefix = False
            for row in range(size):
                if have_prefix:
                    summary_max[sample_indices[row]] = (
                        sample_target[row] + running_max
                    )
                    summary_sum[sample_indices[row]] = running_sum
                source_row = sample_source[row]
                if not have_prefix:
                    running_max = source_row
                    running_sum = torch.ones_like(source_row)
                    have_prefix = True
                else:
                    next_max = torch.maximum(running_max, source_row)
                    running_sum = (
                        running_sum
                        * torch.exp(smoothness * (running_max - next_max))
                        + torch.exp(
                            smoothness * (source_row - next_max)
                        )
                    )
                    running_max = next_max
            offset += size
        return summary_max, summary_sum


class DepthAwareRelationalLayerStackModule(torch.nn.Module):
    def __init__(
        self,
        hparam_config: rgnn.HyperparameterConfig,
        module_config: rgnn.ModuleConfig,
        depth_config: IWDepthConfig | None = None,
        rw_return_config: RWReturnConfig | None = None,
        input_specification: tuple[Encoder, ...] | None = None,
        lgan_config: LGANConfig | None = None,
    ) -> None:
        super().__init__()
        self._config = hparam_config
        self._depth_config = (
            IWDepthConfig(mode=IW_DEPTH_INIT_SCALAR)
            if depth_config is None
            else depth_config.canonicalized()
        )
        self._rw_return_config = (
            RWReturnConfig()
            if rw_return_config is None
            else rw_return_config.canonicalized()
        )
        self._lgan_config = (
            LGANConfig()
            if lgan_config is None
            else lgan_config.canonicalized()
        )
        self._message = module_config.message_function
        self._aggregation = module_config.aggregation_function
        self._update = module_config.update_function
        if self._lgan_config.enabled:
            if input_specification is None:
                raise RuntimeError(
                    "LGAN tree message passing requires an input specification."
                )
            if not isinstance(self._message, rgnn.PredicateMLPMessages):
                raise RuntimeError(
                    "Transition-tree LGAN requires PredicateMLPMessages."
                )
            self._lgan_ordered_relation_names = (
                _ordered_relation_names_from_input_spec(
                    input_specification, hparam_config.domain
                )
            )
            self._lgan_relation_arities = relation_arities_from_input_spec(
                input_specification, hparam_config.domain
            )
            self._lgan_fusion_update = rgnn.MLP(
                4 * hparam_config.embedding_size,
                hparam_config.embedding_size,
            )
        if self._depth_config.uses_init_scalar:
            self._depth_projection = torch.nn.Linear(
                1, hparam_config.embedding_size, bias=False
            )
        if self._rw_return_config.enabled:
            self._rw_return_projection = torch.nn.Linear(
                self._rw_return_config.k_max,
                hparam_config.embedding_size,
                bias=False,
            )
        if (
            self._depth_config.uses_depth_nodes
            and self._depth_config.precedes_mode
            == IW_DEPTH_PRECEDES_FACTORIZED
        ):
            self._factorized_precedes = FactorizedDepthPrecedesRelation(
                hparam_config.embedding_size
            )
        if hparam_config.global_readout:
            self._global_readout = rgnn.SumReadout(
                hparam_config.embedding_size, hparam_config.embedding_size
            )
            self._global_update = rgnn.MLP(
                2 * hparam_config.embedding_size,
                hparam_config.embedding_size,
            )
        if hparam_config.normalize_updates:
            self._update_normalization = torch.nn.LayerNorm(
                hparam_config.embedding_size
            )
        self._hooks: list[Callable[[int, torch.Tensor], None]] = []

    def _notify_hooks(self, iteration: int, embeddings: torch.Tensor) -> None:
        for hook in self._hooks:
            hook(iteration, embeddings)

    def add_hook(self, hook_func: Callable[[int, torch.Tensor], None]) -> None:
        self._hooks.append(hook_func)

    def clear_hooks(self) -> None:
        self._hooks.clear()

    def _initialize_node_embeddings(
        self, input: ExtendedEncodedTensors
    ) -> torch.Tensor:
        node_embeddings = torch.zeros(
            [int(input.node_sizes.sum()), self._config.embedding_size],
            dtype=torch.float,
            device=input.node_sizes.device,
        )
        if self._depth_config.uses_init_scalar and input.action_depths.numel() > 0:
            assert input.action_depths.shape == input.action_depth_mask.shape, (
                "Expected action_depths and action_depth_mask to have the same shape."
            )
            assert input.action_depths.shape == input.action_indices.shape, (
                "Expected action depth tensors to align with action_indices."
            )
            masked_action_indices = input.action_indices[
                input.action_depth_mask
            ].long()
            if masked_action_indices.numel() > 0:
                depth_embeddings = self._depth_projection(
                    input.action_depths[input.action_depth_mask].view(-1, 1)
                )
                node_embeddings = node_embeddings.index_copy(
                    0, masked_action_indices, depth_embeddings
                )
        if self._rw_return_config.enabled:
            if input.rw_return_features.numel() == 0:
                raise RuntimeError(
                    "RW-return features were requested but are missing from the encoded batch."
                )
            if input.rw_return_features.shape[0] != node_embeddings.shape[0]:
                raise RuntimeError(
                    "RW-return features must align with the full node embedding matrix."
                )
            if input.rw_return_features.shape[1] != self._rw_return_config.k_max:
                raise RuntimeError(
                    "RW-return feature width does not match the configured k."
                )
            node_embeddings = node_embeddings + self._rw_return_projection(
                input.rw_return_features.to(dtype=node_embeddings.dtype)
            )
        return node_embeddings

    def _collect_lgan_slot_messages(
        self,
        node_embeddings: torch.Tensor,
        relations: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        assert isinstance(self._message, rgnn.PredicateMLPMessages)
        unexpected_relations = [
            relation_name
            for relation_name, argument_indices in relations.items()
            if argument_indices.numel() > 0
            and relation_name not in self._lgan_relation_arities
        ]
        if unexpected_relations:
            raise RuntimeError(
                "LGAN relation-instance ordering requires all emitted relations "
                f"to be declared by the input spec, got unexpected={unexpected_relations!r}."
            )

        output_messages_list: list[torch.Tensor] = []
        output_indices_list: list[torch.Tensor] = []
        slot_to_relation_parts: list[torch.Tensor] = []
        relation_index = 0
        for relation_name in self._lgan_ordered_relation_names:
            argument_indices = relations.get(relation_name)
            if argument_indices is None or argument_indices.numel() == 0:
                continue
            argument_indices = argument_indices.long()
            arity = self._lgan_relation_arities[relation_name]
            if arity <= 0:
                raise RuntimeError(
                    f"LGAN requires positive relation arities, got {arity} for "
                    f"relation={relation_name!r}."
                )
            if int(argument_indices.numel()) % arity != 0:
                raise RuntimeError(
                    "LGAN canonical relation-instance ordering requires relation rows to "
                    f"be divisible by arity, got relation={relation_name!r}, "
                    f"len={int(argument_indices.numel())}, arity={arity}."
                )
            argument_embeddings = torch.index_select(
                node_embeddings, 0, argument_indices
            )
            argument_messages = self._message._forward_relation(
                relation_name, argument_embeddings
            )
            output_messages = (
                argument_embeddings.view_as(argument_messages)
                + argument_messages
            ).view(-1, self._config.embedding_size)
            output_messages_list.append(output_messages)
            output_indices_list.append(argument_indices)

            relation_count = int(argument_indices.numel()) // arity
            relation_rows = torch.arange(
                relation_index,
                relation_index + relation_count,
                dtype=torch.long,
                device=node_embeddings.device,
            )
            slot_to_relation_parts.append(relation_rows.repeat_interleave(arity))
            relation_index += relation_count

        if not output_messages_list:
            empty_messages = torch.empty(
                (0, self._config.embedding_size),
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            )
            empty_indices = torch.empty(
                (0,),
                dtype=torch.long,
                device=node_embeddings.device,
            )
            return empty_messages, empty_indices, empty_indices, 0

        return (
            torch.cat(output_messages_list, dim=0),
            torch.cat(output_indices_list, dim=0),
            torch.cat(slot_to_relation_parts, dim=0),
            relation_index,
        )

    def _aggregate_messages(
        self,
        *,
        reference_embeddings: torch.Tensor,
        target_count: int,
        messages: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        target_embeddings = reference_embeddings.new_zeros(
            (target_count, self._config.embedding_size)
        )
        if (
            target_count == 0
            or messages.numel() == 0
            or indices.numel() == 0
        ):
            return target_embeddings
        return self._aggregation.forward(
            target_embeddings,
            messages,
            indices.long(),
        )

    def _select_relation_messages(
        self,
        relation_embeddings: torch.Tensor,
        relation_indices: torch.Tensor,
    ) -> torch.Tensor:
        if relation_indices.numel() == 0:
            return relation_embeddings.new_zeros(
                (0, self._config.embedding_size)
            )
        return relation_embeddings.index_select(0, relation_indices.long())

    @staticmethod
    def _summarize_smax_messages(
        node_embeddings: torch.Tensor,
        messages: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summary_max = torch.zeros_like(node_embeddings)
        summary_sum = torch.zeros_like(node_embeddings)
        if messages.numel() == 0:
            return summary_max, summary_sum
        summary_max.index_reduce_(
            0,
            indices,
            messages,
            reduce="amax",
            include_self=False,
        )
        max_offsets = summary_max.index_select(0, indices)
        summary_sum.index_add_(
            0,
            indices,
            torch.exp(
                FactorizedDepthPrecedesRelation.MAXIMUM_SMOOTHNESS
                * (messages - max_offsets)
            ),
        )
        return summary_max, summary_sum

    def _aggregate_with_scanned_factorized_precedes(
        self,
        node_embeddings: torch.Tensor,
        generic_messages: torch.Tensor,
        generic_indices: torch.Tensor,
        input: ExtendedEncodedTensors,
    ) -> torch.Tensor:
        assert isinstance(self._aggregation, rgnn.SmoothMaximumAggregation), (
            "Scanned factorized depth precedes requires smooth-max aggregation."
        )
        assert hasattr(self, "_factorized_precedes")
        ordinary_max, ordinary_sum = self._summarize_smax_messages(
            node_embeddings, generic_messages, generic_indices.long()
        )
        special_max, special_sum = self._factorized_precedes.scan_smax_summary(
            node_embeddings, input
        )
        smoothness = FactorizedDepthPrecedesRelation.MAXIMUM_SMOOTHNESS
        combined_max = torch.maximum(ordinary_max, special_max)
        combined_sum = (
            ordinary_sum
            * torch.exp(smoothness * (ordinary_max - combined_max))
            + special_sum
            * torch.exp(smoothness * (special_max - combined_max))
        )
        return combined_max + (1.0 / smoothness) * torch.log(
            combined_sum + 1e-16
        )

    def forward(self, input: ExtendedEncodedTensors) -> torch.Tensor:
        self._message.setup(input.flattened_relations)
        node_embeddings = self._initialize_node_embeddings(input)
        for iteration in range(self._config.num_layers):
            if self._lgan_config.enabled:
                slot_messages, slot_indices, slot_to_relation_indices, relation_count = (
                    self._collect_lgan_slot_messages(
                        node_embeddings, input.flattened_relations
                    )
                )
                if (
                    self._depth_config.uses_depth_nodes
                    and self._depth_config.precedes_mode
                    == IW_DEPTH_PRECEDES_FACTORIZED
                ):
                    if self._depth_config.precedes_scan:
                        aggregated_messages = (
                            self._aggregate_with_scanned_factorized_precedes(
                                node_embeddings,
                                slot_messages,
                                slot_indices,
                                input,
                            )
                        )
                    else:
                        assert hasattr(self, "_factorized_precedes")
                        depth_messages, depth_indices = (
                            self._factorized_precedes.explicit_messages(
                                node_embeddings, input
                            )
                        )
                        node_messages = slot_messages
                        node_indices = slot_indices
                        if depth_messages.numel() > 0:
                            node_messages = torch.cat(
                                (node_messages, depth_messages), dim=0
                            )
                            node_indices = torch.cat(
                                (node_indices.long(), depth_indices.long()),
                                dim=0,
                            )
                        aggregated_messages = self._aggregate_messages(
                            reference_embeddings=node_embeddings,
                            target_count=int(node_embeddings.size(0)),
                            messages=node_messages,
                            indices=node_indices,
                        )
                else:
                    aggregated_messages = self._aggregate_messages(
                        reference_embeddings=node_embeddings,
                        target_count=int(node_embeddings.size(0)),
                        messages=slot_messages,
                        indices=slot_indices,
                    )

                relation_embeddings = self._aggregate_messages(
                    reference_embeddings=node_embeddings,
                    target_count=relation_count,
                    messages=slot_messages,
                    indices=slot_to_relation_indices,
                )
                rr_messages = self._aggregate_messages(
                    reference_embeddings=relation_embeddings,
                    target_count=relation_count,
                    messages=self._select_relation_messages(
                        relation_embeddings,
                        input.lgan_rr_src_relation_indices,
                    ),
                    indices=input.lgan_rr_dst_relation_indices,
                )
                relation_embeddings = relation_embeddings + rr_messages
                tn_messages = self._aggregate_messages(
                    reference_embeddings=node_embeddings,
                    target_count=int(node_embeddings.size(0)),
                    messages=self._select_relation_messages(
                        relation_embeddings,
                        input.lgan_tn_relation_indices,
                    ),
                    indices=input.lgan_tn_entity_indices,
                )
                nn_messages = self._aggregate_messages(
                    reference_embeddings=node_embeddings,
                    target_count=int(node_embeddings.size(0)),
                    messages=self._select_relation_messages(
                        relation_embeddings,
                        input.lgan_nn_relation_indices,
                    ),
                    indices=input.lgan_nn_entity_indices,
                )
                next_node_embeddings = self._lgan_fusion_update(
                    torch.cat(
                        (
                            aggregated_messages,
                            tn_messages,
                            nn_messages,
                            node_embeddings,
                        ),
                        dim=1,
                    )
                )
            else:
                generic_messages, generic_indices = self._message.forward(
                    node_embeddings, input.flattened_relations
                )
                if (
                    self._depth_config.uses_depth_nodes
                    and self._depth_config.precedes_mode
                    == IW_DEPTH_PRECEDES_FACTORIZED
                ):
                    if self._depth_config.precedes_scan:
                        aggregated_messages = (
                            self._aggregate_with_scanned_factorized_precedes(
                                node_embeddings,
                                generic_messages,
                                generic_indices,
                                input,
                            )
                        )
                    else:
                        assert hasattr(self, "_factorized_precedes")
                        depth_messages, depth_indices = (
                            self._factorized_precedes.explicit_messages(
                                node_embeddings, input
                            )
                        )
                        if depth_messages.numel() > 0:
                            generic_messages = torch.cat(
                                (generic_messages, depth_messages), dim=0
                            )
                            generic_indices = torch.cat(
                                (generic_indices.long(), depth_indices.long()),
                                dim=0,
                            )
                        aggregated_messages = self._aggregation.forward(
                            node_embeddings,
                            generic_messages,
                            generic_indices.long(),
                        )
                else:
                    aggregated_messages = self._aggregation.forward(
                        node_embeddings,
                        generic_messages,
                        generic_indices.long(),
                    )
                next_node_embeddings = self._update.forward(
                    node_embeddings, aggregated_messages
                )
            if self._config.normalize_updates:
                next_node_embeddings = self._update_normalization(
                    next_node_embeddings
                )
            if self._config.global_readout:
                global_embedding = self._global_readout(
                    node_embeddings, input.node_sizes
                )
                global_messages = self._global_update(
                    torch.cat(
                        (
                            node_embeddings,
                            global_embedding.repeat_interleave(
                                input.node_sizes, dim=0
                            ),
                        ),
                        1,
                    )
                )
                if self._config.normalize_updates:
                    global_messages = self._update_normalization(
                        global_messages
                    )
                next_node_embeddings = global_messages + next_node_embeddings
            if self._config.binarize_updates:
                next_node_embeddings = rgnn.utils.gumbel_sigmoid(
                    next_node_embeddings, hard=True
                )
            if self._config.residual_updates:
                next_node_embeddings = node_embeddings + next_node_embeddings
            node_embeddings = next_node_embeddings
            self._notify_hooks(iteration, node_embeddings)
        self._message.cleanup()
        return node_embeddings


class DepthAwareExtendedRGNN(ExtendedRGNN):
    def __init__(
        self,
        hparam_config: rgnn.HyperparameterConfig,
        module_config: rgnn.ModuleConfig,
        input_spec: tuple[Encoder, ...],
        output_spec: list[tuple[str, rgnn.Decoder]],
        depth_config: IWDepthConfig | None = None,
        rw_return_config: RWReturnConfig | None = None,
        lgan_config: LGANConfig | None = None,
    ) -> None:
        super().__init__(hparam_config, module_config, input_spec, output_spec)
        self._depth_config = (
            IWDepthConfig(mode=IW_DEPTH_INIT_SCALAR)
            if depth_config is None
            else depth_config.canonicalized()
        )
        self._rw_return_config = (
            RWReturnConfig()
            if rw_return_config is None
            else rw_return_config.canonicalized()
        )
        self._lgan_config = (
            LGANConfig() if lgan_config is None else lgan_config.canonicalized()
        )
        self._mpnn_module = DepthAwareRelationalLayerStackModule(
            hparam_config,
            module_config,
            self._depth_config,
            self._rw_return_config,
            input_spec,
            self._lgan_config,
        )

    def clone(self) -> "DepthAwareExtendedRGNN":
        model_clone = DepthAwareExtendedRGNN(
            self._hparam_config,
            self._module_config,
            self._input_spec,
            self._output_spec,
            self._depth_config,
            self._rw_return_config,
            self._lgan_config,
        )
        model_clone.load_state_dict(self.state_dict())
        return model_clone.to(self.get_device())


def get_shared_atom_name(
    predicate: mm.Predicate, shared_label, suffix: str
) -> str:
    """Generate a name for shared atom relations.

    Args:
        predicate: The predicate affected by the effect.

    Returns:
        String name for the effect relation.
    """
    return predicate.get_name() + suffix + shared_label


class GroundActionsAuxiliaryEncoder(Encoder):
    """Encoder for ground actions.

    This encoder is different from the GroundActionsEncoder in that it
    encodes the ground actions as auxiliary nodes that are not used in
    the readout, but are only there for expressivity purposes.
    """

    def __init__(self, suffix: str = "") -> None:
        """Initialize the ground actions encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), "Suffix must be a string."
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for actions.

        Args:
            domain: The PDDL domain containing action definitions.

        Returns:
            List of (relation_name, arity) pairs for all actions, where arity
            is the action's parameter count plus 1 (for the action node itself).
        """
        return [
            (get_action_name(action, self.suffix), action.get_arity() + 1)
            for action in domain.get_actions()
        ]

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "EncodingContext",
    ) -> None:
        assert isinstance(input_value, list), (
            f"GroundActionsEncoder expected a list, got {type(input_value)}"
        )

        for action in input_value:
            assert isinstance(action, mm.GroundAction), (
                f"Expected a GroundAction in the list, got {type(action)}"
            )
            relation_name = get_action_name(action, self.suffix)
            action_id = context.new_or_existing_auxiliary_id(relation_name)
            term_ids = [action_id] + [
                context.get_object_id(obj.get_index())
                for obj in action.get_objects()
            ]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = term_ids
            else:
                encoding.flattened_relations[relation_name].extend(term_ids)


class TransitionEffectsStateEncoder(Encoder):
    """Encoder for transition effects.

    This encoder transforms action effects (lists of literals describing state
    changes) into nodes and relations for the graph neural network. Each
    transition becomes a new node, with relations connecting it to affected atoms.

    The difference to TransitionEffectsEncoder is that this encoder does not create new
    action nodes for each transition, but instead encodes the nodes as object nodes,
    which means that they will not have their own Q-values in the readout.
    """

    def __init__(self, suffix: str = "") -> None:
        """Initialize the transition effects encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), "Suffix must be a string."
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for transition effects.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for effect relations,
            including positive/negative effects and goal-affecting variants.
        """
        ignored_predicate_names = ["number"]
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        relations = []
        relations.extend(
            [
                (
                    get_effect_name(predicate, True, False, self.suffix),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
            ]
        )
        relations.extend(
            [
                (
                    get_effect_name(predicate, False, False, self.suffix),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
            ]
        )
        relations.extend(
            [
                (
                    get_effect_name(predicate, True, True, self.suffix),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
            ]
        )
        relations.extend(
            [
                (
                    get_effect_name(predicate, False, True, self.suffix),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
            ]
        )
        relations.append((get_effect_relation_name(self.suffix), 2))
        return relations

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "EncodingContext",
    ) -> None:
        if isinstance(input_value, tuple):
            effects_list = input_value[0]
            effects_relations = input_value[1]
        elif isinstance(input_value, list):
            effects_list = input_value
            effects_relations = []
        else:
            raise AssertionError(
                f"TransitionEffectsEncoder expected a list or tuple, got {type(input_value)}"
            )

        problem = state.get_problem()
        goal_condition = problem.get_goal_condition()
        num_transitions = len(effects_list)

        transition_index_to_id: dict[int, int] = dict()

        for transition_index, effects in enumerate(effects_list):
            assert isinstance(effects, list) or isinstance(effects, tuple), (
                "Expected a list of lists of ground literals."
            )
            transition_id = context.new_or_existing_auxiliary_id(
                transition_index
            )
            transition_index_to_id[transition_index] = transition_id

            for effect_literal in effects:
                assert isinstance(effect_literal, mm.GroundLiteral), (
                    "Expected a list of lists of ground literals."
                )
                effect_atom = effect_literal.get_atom()
                effect_name = get_effect_name(
                    effect_atom.get_predicate(),
                    effect_literal.get_polarity(),
                    False,
                    self.suffix,
                )
                object_ids = [transition_id] + [
                    context.get_object_id(obj.get_index())
                    for obj in effect_atom.get_terms()
                ]

                if effect_name not in encoding.flattened_relations:
                    encoding.flattened_relations[effect_name] = object_ids
                else:
                    encoding.flattened_relations[effect_name].extend(object_ids)

                # Add literals stating how this transition affects the goal
                for goal_literal in goal_condition:
                    assert isinstance(goal_literal, mm.GroundLiteral), (
                        "Goal condition should contain ground literals."
                    )
                    assert goal_literal.get_polarity(), (
                        "Only positive literals are supported in the goal condition."
                    )
                    if effect_atom == goal_literal.get_atom():
                        goal_effect_name = get_effect_name(
                            effect_atom.get_predicate(),
                            effect_literal.get_polarity(),
                            True,
                            self.suffix,
                        )
                        goal_object_ids = [transition_id] + [
                            context.get_object_id(obj.get_index())
                            for obj in effect_atom.get_terms()
                        ]
                        if goal_effect_name not in encoding.flattened_relations:
                            encoding.flattened_relations[goal_effect_name] = (
                                goal_object_ids
                            )
                        else:
                            encoding.flattened_relations[
                                goal_effect_name
                            ].extend(goal_object_ids)
                        break  # No need to check other goal literals

        # Add relations between transitions if provided
        for from_index, to_index in effects_relations:
            assert isinstance(from_index, int) and isinstance(to_index, int), (
                "Effect relations must be pairs of integers."
            )
            assert from_index < num_transitions, (
                f"Invalid from_index {from_index} in effect relations."
            )
            assert to_index < num_transitions, (
                f"Invalid to_index {to_index} in effect relations."
            )
            from_id = transition_index_to_id[from_index]
            to_id = transition_index_to_id[to_index]
            effect_relation_name = get_effect_relation_name(self.suffix)
            relation_ids = [from_id, to_id]

            if effect_relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[effect_relation_name] = (
                    relation_ids
                )
            else:
                encoding.flattened_relations[effect_relation_name].extend(
                    relation_ids
                )


class DifferenceViaRelationsStateEncoder(rgnn.StateEncoder):
    effect = {
        0: "_both",
        1: "_curr-only",
        2: "_succ-only",
        3: "_neither",
    }

    # def __init__(self, suffix: str = "") -> None:
    #     """Initialize the state encoder.
    #
    #     Args:
    #         suffix: Optional suffix to append to relation names to avoid conflicts.
    #     """
    #     super().__init__()
    #     assert isinstance(suffix, str), "Suffix must be a string."
    #     self.suffix = suffix

    def _get_effect_name(
        self,
        predicate: mm.Predicate,
        key: None | tuple[bool, bool],
        affects_goal: bool,
        suffix: str,
    ) -> str:
        """Generate a name for effect relations.

        Args:
            predicate: The predicate affected by the effect.
            positive: Whether this is a positive (add) or negative (delete) effect.
            affects_goal: Whether this effect affects a goal atom.

        Returns:
            String name for the effect relation.
        """
        return (
            predicate.get_name()
            + suffix
            + self.effect[key]
            + ("_goal" if affects_goal else "")
        )

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for state atoms.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for all predicates except 'number'.
        """
        ignored_predicate_names = ["number"]
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        relations = [
            (
                self._get_effect_name(
                    predicate, key, affects_goal, self.suffix
                ),
                predicate.get_arity(),
            )
            for key in self.effect.keys()
            for affects_goal in (True, False)
            for predicate in predicates
            if predicate.get_name() not in ignored_predicate_names
        ]
        return relations

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "ExtendedEncodingContext",
    ) -> None:
        source_state, (added, deleted) = input_value
        goal_condition = state.get_problem().get_goal_condition()

        # Add atom relations for all atoms in the state
        effects = list(
            chain(zip(repeat(True), added), zip(repeat(False), deleted))
        )
        source_atoms = source_state.get_atoms()
        both = set(source_atoms)
        sat_added = set()
        sat_deleted = set()
        sat_both = set()
        for polarity, atom in effects:
            goal_satisfaction_effect = None
            for goal_literal in goal_condition:
                assert isinstance(goal_literal, mm.GroundLiteral), (
                    "Goal condition should contain ground literals."
                )
                assert goal_literal.get_polarity(), (
                    "Only positive literals are supported in the goal condition."
                )

                if atom == goal_literal.get_atom():
                    # effect satisfies goal if polarity matches goal polarity:
                    # since effects are exact atom diffs, the polarity of the effect
                    # tells us if we added goal satisfaction or deleted it
                    goal_satisfaction_effect = (
                        polarity == goal_literal.get_polarity()
                    )
            if goal_satisfaction_effect is not None:
                if goal_satisfaction_effect:
                    sat_added.add(atom)
                else:
                    sat_deleted.add(atom)

        for atom in source_atoms:
            if atom.is_static():
                continue

            for polarity, effect_atom in effects:
                if effect_atom == atom:
                    if polarity:
                        added.add(atom)
                    else:
                        deleted.add(atom)
                    both.remove(atom)
                    break

            for goal_literal in goal_condition:
                goal_atom = goal_literal.get_atom()
                if (
                    goal_atom not in sat_added
                    and goal_atom not in sat_deleted
                    and atom == goal_atom
                ):
                    sat_both.add(goal_atom)
                    break
        all_sat = sat_both | sat_deleted | sat_added
        sat_none = {
            lit.get_atom()
            for lit in goal_condition
            if lit.get_atom() not in all_sat
        }
        for affects_goal, key, atom_set in [
            (False, 0, both),
            (False, 1, deleted),
            (False, 2, added),
            (True, 0, sat_both),
            (True, 1, sat_deleted),
            (True, 2, sat_added),
            (True, 3, sat_none),
        ]:
            for atom in atom_set:
                relation_name = self._get_effect_name(
                    atom.get_predicate(), key, affects_goal, self.suffix
                )
                object_indices: list[int] = [
                    context.get_object_id(obj.get_index())
                    for obj in atom.get_terms()
                ]
                encoding.flattened_relations.setdefault(
                    relation_name, []
                ).extend(
                    object_indices,
                )


class DifferenceViaObjectsStateEncoder(rgnn.StateEncoder):
    atom_status_prefix = "atom_status"
    unary_change_relations = {
        True: "__marks_added__",
        False: "__marks_deleted__",
        None: "__marks_unchanged__",
    }

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        relations = super().get_relations(domain)
        relations = [(rel, arity + 1) for rel, arity in relations]
        relations.extend(
            (rel, 1) for rel in self.unary_change_relations.values()
        )
        return relations

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "ExtendedEncodingContext",
    ) -> None:
        source_state, effects = input_value

        for prefix, (key, relation) in zip(
            repeat(self.atom_status_prefix), self.unary_change_relations.items()
        ):
            encoding.flattened_relations[relation] = [
                context.get_or_new_object_id((prefix, key))
            ]

        # Add atom relations for all atoms in the state
        for atom in source_state.get_atoms():
            change = None
            if not atom.is_static():
                for effect in effects:
                    assert isinstance(effect, mm.GroundLiteral)
                    if effect.get_atom() == atom:
                        change = effect.get_polarity()
                        break
            relation_name = rgnn.utils.get_atom_name(
                atom, state, False, self.suffix
            )
            object_indices: list[int] = [
                context.get_or_new_object_id((self.atom_status_prefix, change))
            ] + [
                context.get_object_id(obj.get_index())
                for obj in atom.get_terms()
            ]
            encoding.flattened_relations.setdefault(relation_name, []).extend(
                object_indices,
            )


class DifferenceViaObjectsGoalEncoder(Encoder):
    """Encoder for goal conditions.

    This encoder transforms a goal condition (conjunctive condition of literals
    that must be satisfied) into relations for the graph neural network.
    It creates both goal-specific relations and marks which atoms are true/false
    in the current state relative to the goal.
    """

    sat_prefix = "sat"
    sat_change_prefix = "sat_change"
    unary_goal_satisfaction_relations = {
        True: "__marks_satisfied__",
        False: "__marks_unsatisfied__",
    }
    unary_goal_satisfaction_added_relations = {
        True: "__marks_satisfied_added__",
        False: "__marks_unsatisfied_added__",
        None: "__marks_satisfaction_unchanged__",
    }

    def __init__(self, suffix: str = "") -> None:
        """Initialize the goal encoder.

        Args:
            suffix: Suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), "Suffix must be a string."
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for goal conditions.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for goal predicates,
            including both true and false variants.
        """
        ignored_predicate_names = ["number"]
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        relations = []
        relations.extend(
            [
                (
                    get_predicate_name(predicate, True, is_true, self.suffix),
                    predicate.get_arity() + 2,
                )
                for is_true in [True, False]
                for predicate in predicates
            ]
        )
        relations.extend(
            (rel, 1)
            for rel in chain(
                self.unary_goal_satisfaction_relations.values(),
                self.unary_goal_satisfaction_added_relations.values(),
            )
        )
        return relations

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "ExtendedEncodingContext",
    ) -> None:
        target_state, goal_condition = input_value
        target_atoms = set(target_state.get_atoms(ignore_static=True))
        current_atoms = set(state.get_atoms(ignore_static=True))
        assert isinstance(goal_condition, mm.GroundConjunctiveCondition), (
            f"GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}"
        )

        for prefix, (key, relation) in chain(
            zip(
                repeat(self.sat_prefix),
                self.unary_goal_satisfaction_relations.items(),
            ),
            zip(
                repeat(self.sat_change_prefix),
                self.unary_goal_satisfaction_added_relations.items(),
            ),
        ):
            encoding.flattened_relations[relation] = [
                context.get_or_new_object_id((prefix, key))
            ]

        for literal in goal_condition:
            assert isinstance(literal, mm.GroundLiteral), (
                "Goal condition should contain ground literals."
            )
            assert literal.get_polarity(), (
                "Only positive literals are supported."
            )
            atom = literal.get_atom()
            goal_sat = atom in current_atoms
            goal_change = atom in target_atoms
            goal_change = None if (goal_change is goal_sat) else goal_change
            relation_name = get_predicate_name(
                atom.get_predicate(), True, False, self.suffix
            )
            # relation_name = rgnn.utils.get_atom_name(
            #     atom, state, False, self.suffix
            # )
            object_indices: list[int] = [
                context.get_object_id(obj.get_index())
                for obj in chain(
                    atom.get_terms(),
                )
            ]
            encoding.flattened_relations.setdefault(relation_name, []).extend(
                chain(
                    (
                        context.get_or_new_object_id(
                            (self.sat_prefix, goal_sat)
                        ),
                        context.get_or_new_object_id(
                            (self.sat_change_prefix, goal_change)
                        ),
                    ),
                    object_indices,
                )
            )


class TransitionTreeEncoder(Encoder):
    """Encoder for transition trees.

    This encoder transforms action effects (lists of literals describing state
    changes) into nodes and relations for the graph neural network. Each
    transition becomes a new node, with relations connecting it to affected atoms.
    """

    def __init__(
        self,
        suffix: str = "",
        connect_to_root_handle: bool = False,
        encode_actions: bool = True,
        encode_effects: bool = True,
        encode_parent_relations: bool = True,
        encode_shared_relations: bool = False,
        shared_label: str = "_shared_label",
        separate_action_nodes: bool = False,
        depth_config: IWDepthConfig | None = None,
        rw_return_config: RWReturnConfig | None = None,
        lgan_config: LGANConfig | None = None,
    ) -> None:
        """Initialize the transition effects encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), "Suffix must be a string."
        self.suffix = suffix
        self.separate_action_nodes = separate_action_nodes
        if self.separate_action_nodes:
            assert connect_to_root_handle, (
                "separate action nodes requires that we can connect to root handle. "
                "Depth 0 actions have to connect to a state as well (the root state)."
            )
            assert encode_actions, (
                "separate_action_nodes requires encode_actions=True."
            )
        self.connect_to_root_handle = connect_to_root_handle
        self.encode_actions = encode_actions
        self.encode_effects = encode_effects
        self.encode_parent_relations = encode_parent_relations
        self.encode_shared_relations = encode_shared_relations
        self.shared_label = shared_label
        self.depth_config = (
            IWDepthConfig() if depth_config is None else depth_config.canonicalized()
        )
        self.rw_return_config = (
            RWReturnConfig()
            if rw_return_config is None
            else rw_return_config.canonicalized()
        )
        self.lgan_config = (
            LGANConfig() if lgan_config is None else lgan_config.canonicalized()
        )

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for transition effects.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for effect relations,
            including positive/negative effects and goal-affecting variants.
        """
        ignored_predicate_names = ["number"]
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        relations = []
        if self.encode_effects:
            relations.extend(
                (
                    get_effect_name(
                        predicate, positive, affects_goal, self.suffix
                    ),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
                for positive in (True, False)
                for affects_goal in (False, True)
            )
        if self.encode_actions:
            relations.extend(
                (get_action_name(action, self.suffix), action.get_arity() + 1)
                for action in domain.get_actions()
            )
        if self.encode_shared_relations:
            relations.extend(
                (
                    get_shared_atom_name(
                        predicate, self.shared_label, self.suffix
                    ),
                    predicate.get_arity(),
                )
                for predicate in domain.get_predicates()
                if predicate.get_name() not in ignored_predicate_names
            )
        if getattr(self, "encode_parent_relations", True):
            relations.append(
                (
                    get_effect_relation_name(self.suffix),
                    3 if self.separate_action_nodes else 2,
                )
            )
        if self.depth_config.uses_depth_nodes:
            relations.append((IW_IN_DEPTH_RELATION, 2))
            if self.depth_config.precedes_mode == IW_DEPTH_PRECEDES_MLP:
                relations.append((IW_DEPTH_PRECEDES_RELATION, 2))
        return relations

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: EncodedLists,
        context: ExtendedEncodingContext,
    ) -> None:
        transition_data_is_expansion = is_iw_expansion(input_value)
        if transition_data_is_expansion:
            iwaction_list = expansion_transitions(input_value)
            effects_relations = [
                (
                    parent_edge_src_index(edge),
                    parent_edge_incoming_action(edge),
                    parent_edge_tgt_index(edge),
                )
                for edge in expansion_parent_edges(input_value)
            ]
            transition_depths = [
                transition_depth_feature_value(transition)
                for transition in iwaction_list
            ]
            raw_transition_depths = [
                transition_raw_depth(transition) for transition in iwaction_list
            ]
        elif isinstance(input_value, tuple):
            iwaction_list = input_value[0]
            effects_relations = input_value[1] if len(input_value) > 1 else []
            transition_depths = input_value[2] if len(input_value) > 2 else None
            raw_transition_depths = (
                [int(depth) for depth in transition_depths]
                if transition_depths is not None
                else []
            )
        elif isinstance(input_value, list):
            iwaction_list = input_value
            effects_relations = []
            transition_depths = None
            raw_transition_depths = []
        else:
            raise AssertionError(
                f"TransitionEffectsEncoder expected a list or tuple, got {type(input_value)}"
            )
        if transition_data_is_expansion and self.depth_config.uses_depth_nodes:
            transition_depths = raw_transition_depths
        use_scalar_depth = (
            transition_depths is not None and not self.depth_config.uses_depth_nodes
        )
        if transition_depths is None:
            transition_depths = [0.0] * len(iwaction_list)
            if not raw_transition_depths:
                raw_transition_depths = [0] * len(iwaction_list)
        assert len(transition_depths) == len(iwaction_list), (
            "Expected transition depths to align with IW actions."
        )
        if not raw_transition_depths:
            raw_transition_depths = [int(depth) for depth in transition_depths]
        if self.depth_config.uses_depth_nodes:
            for depth in transition_depths:
                if int(depth) != float(depth):
                    raise RuntimeError(
                        "depth_nodes mode expects raw integer IW depths."
                    )

        problem = state.get_problem()
        goal_condition = problem.get_goal_condition()
        num_transitions = len(iwaction_list)
        transition_index_to_id: dict[int, int] = dict()
        depth_attachments: list[tuple[int, int]] = []
        realized_depths = list(raw_transition_depths)
        transition_index_start = 0
        if self.connect_to_root_handle:
            if transition_data_is_expansion:
                assert expansion_includes_root_handle(input_value), (
                    "Native IW expansion must preserve the explicit root handle "
                    "when connect_to_root_handle=True."
                )
            assert context.root_action_ids, "No root state handle id found."
            # Rooted IW trees use index 0 for the root state handle.
            transition_index_to_id[0] = context.root_action_ids[0]
            transition_index_start = 1
            if self.depth_config.uses_depth_nodes:
                realized_depths.append(0)
                depth_attachments.append((0, context.root_action_ids[0]))
            if effects_relations:
                # Rooted edges should target indices 1..num_transitions, with 0
                # reserved for the explicit root state handle.
                max_target_transition_index = max(
                    tgt_index for _, _, tgt_index in effects_relations
                )
                assert max_target_transition_index == num_transitions, (
                    "connect_to_root_handle=True requires rooted IW edges "
                    "generated with include_start=True."
                )

        current_state_atoms: set[GroundAtom] = set()
        if self.encode_shared_relations:
            current_state_atoms = set(state.get_atoms(ignore_static=True))
        incoming_action_id_by_tgt_index: dict[int, int] = {}
        for depth_index, iwaction in enumerate(iwaction_list):
            if transition_data_is_expansion:
                transition_idx = transition_index(iwaction)
                effects = transition_effect_literals(iwaction)
                underlying_actions = transition_underlying_actions(iwaction)
            else:
                transition_idx = transition_index_start + depth_index
                if isinstance(iwaction, (list, tuple)):
                    effects = iwaction
                    underlying_actions = []
                else:
                    effects = iwaction.get_effect()
                    underlying_actions = iwaction.actions
            assert isinstance(effects, list) or isinstance(effects, tuple), (
                "Expected a list of lists of ground literals."
            )

            transition_id = self._new_transition_id(
                context,
                transition_depth=float(transition_depths[depth_index]),
                use_depth=use_scalar_depth,
                aux=self.separate_action_nodes,
                aux_key=transition_idx,
            )
            transition_index_to_id[transition_idx] = transition_id
            if self.depth_config.uses_depth_nodes:
                depth_attachments.append(
                    (raw_transition_depths[depth_index], transition_id)
                )

            # encode ground action with connection to the transition handle
            if self.encode_actions:
                assert underlying_actions, (
                    "encode_actions=True requires transitions to carry underlying actions."
                )
                last_prim_action = underlying_actions[-1]
                if self.separate_action_nodes:
                    action_id = self._new_transition_id(
                        context,
                        transition_depth=float(transition_depths[depth_index]),
                        use_depth=use_scalar_depth,
                        aux=False,
                    )
                    incoming_action_id_by_tgt_index[
                        transition_idx
                    ] = action_id
                else:
                    action_id = transition_id
                self._encode_action(
                    last_prim_action, action_id, context, encoding
                )

            if self.encode_shared_relations:
                self._encode_shared_relation(
                    current_state_atoms, iwaction, context, encoding
                )

            if self.encode_effects:
                # encode each literal in the transition
                for effect_literal in effects:
                    self._encode_effect_literal(
                        effect_literal,
                        transition_id,
                        goal_condition,
                        context,
                        encoding,
                    )

        # Add relations between transitions if provided
        if self.separate_action_nodes:
            expected_target_indices = set(
                range(
                    transition_index_start,
                    transition_index_start + num_transitions,
                )
            )
            assert (
                set(incoming_action_id_by_tgt_index)
                == expected_target_indices
            ), "Each successor state must have exactly one incoming action node."
            assert len(context.get_auxiliary_ids()) == num_transitions, (
                "Each successor state must have exactly one auxiliary state node."
            )
        if getattr(self, "encode_parent_relations", True):
            for src_idx, _, tgt_idx in effects_relations:
                self._encode_parent_relation(
                    encoding,
                    num_transitions,
                    src_idx,
                    tgt_idx,
                    transition_index_to_id=transition_index_to_id,
                    incoming_action_id_by_tgt_index=(
                        incoming_action_id_by_tgt_index
                        if self.separate_action_nodes
                        else None
                    ),
                )
        _encode_depth_node_structure(
            context=context,
            encoding=encoding,
            depth_config=self.depth_config,
            realized_depths=realized_depths,
            attachments=depth_attachments,
        )

    def _encode_parent_relation(
        self,
        encoding: EncodedLists,
        num_transitions: int,
        src_index,
        tgt_index,
        transition_index_to_id: dict[int, int],
        incoming_action_id_by_tgt_index: dict[int, int] | None = None,
    ):
        assert isinstance(src_index, int) and isinstance(tgt_index, int), (
            "Parent relation endpoints must be integers."
        )
        assert src_index < num_transitions, (
            f"Invalid source transition index {src_index} in effect relations."
        )
        assert tgt_index in transition_index_to_id, (
            f"Invalid target transition index {tgt_index} in effect relations."
        )
        assert src_index in transition_index_to_id, (
            f"Missing transition node for source index {src_index}."
        )
        src_id = transition_index_to_id[src_index]
        tgt_id = transition_index_to_id[tgt_index]
        effect_relation_name = get_effect_relation_name(self.suffix)
        if self.separate_action_nodes:
            assert incoming_action_id_by_tgt_index is not None, (
                "Missing incoming action lookup for separate action nodes."
            )
            assert tgt_index in incoming_action_id_by_tgt_index, (
                f"Missing action node for target index {tgt_index}."
            )
            relation_ids = [
                src_id,
                incoming_action_id_by_tgt_index[tgt_index],
                tgt_id,
            ]
        else:
            relation_ids = [src_id, tgt_id]
        encoding.flattened_relations.setdefault(
            effect_relation_name, []
        ).extend(relation_ids)

    def _new_transition_id(
        self,
        context: EncodingContext,
        transition_depth: float,
        use_depth: bool,
        aux: bool = False,
        aux_key: Any = None,
    ) -> int:
        if use_depth:
            assert isinstance(context, ExtendedEncodingContext)
            if aux:
                return context.new_or_existing_auxiliary_id(
                    aux_key, depth=transition_depth, depth_mask=True
                )
            else:
                return context.new_action_id(
                    depth=transition_depth,
                    depth_mask=True,
                )
        if aux:
            return context.new_or_existing_auxiliary_id(aux_key)
        else:
            return context.new_action_id()

    def _encode_effect_literal(
        self,
        effect_literal: GroundLiteral | Any,
        transition_id: int,
        goal_condition: GroundConjunctiveCondition,
        context: EncodingContext,
        encoding: EncodedLists,
    ):
        assert isinstance(effect_literal, mm.GroundLiteral), (
            "Expected a list of lists of ground literals."
        )
        effect_atom = effect_literal.get_atom()
        effect_name = get_effect_name(
            effect_atom.get_predicate(),
            effect_literal.get_polarity(),
            False,
            self.suffix,
        )
        object_ids = [transition_id] + [
            context.get_object_id(obj.get_index())
            for obj in effect_atom.get_terms()
        ]
        encoding.flattened_relations.setdefault(effect_name, []).extend(
            object_ids
        )

        # Add literals stating how this transition affects the goal
        for goal_literal in goal_condition:
            assert isinstance(goal_literal, mm.GroundLiteral), (
                "Goal condition should contain ground literals."
            )
            assert goal_literal.get_polarity(), (
                "Only positive literals are supported in the goal condition."
            )
            if effect_atom == goal_literal.get_atom():
                goal_effect_name = get_effect_name(
                    effect_atom.get_predicate(),
                    effect_literal.get_polarity(),
                    True,
                    self.suffix,
                )
                goal_object_ids = [transition_id] + [
                    context.get_object_id(obj.get_index())
                    for obj in effect_atom.get_terms()
                ]
                encoding.flattened_relations.setdefault(
                    goal_effect_name, []
                ).extend(goal_object_ids)
                break  # No need to check other goal literals

    def _encode_shared_relation(
        self,
        current_state_atoms: set[GroundAtom],
        iwaction: IWGroundAction,
        context: EncodingContext,
        encoding: EncodedLists,
    ):
        shared_atoms = current_state_atoms & set(
            iwaction.final_state.get_atoms(ignore_static=True)
        )
        for atom in shared_atoms:
            relation_name = get_shared_atom_name(
                atom.get_predicate(), self.shared_label, self.suffix
            )
            object_indices: list[int] = [
                context.get_object_id(obj.get_index())
                for obj in atom.get_terms()
            ]
            encoding.flattened_relations.setdefault(relation_name, []).extend(
                object_indices
            )

    def _encode_action(
        self,
        last_prim_action: GroundAction,
        transition_id: int,
        context: EncodingContext,
        encoding: EncodedLists,
    ):
        assert isinstance(last_prim_action, mm.GroundAction), (
            f"Expected a GroundAction in the list, got {type(last_prim_action)}"
        )
        relation_name = get_action_name(last_prim_action, self.suffix)
        term_ids = [transition_id] + [
            context.get_object_id(obj.get_index())
            for obj in last_prim_action.get_objects()
        ]
        encoding.flattened_relations.setdefault(relation_name, []).extend(
            term_ids
        )


class DifferentTransitionTreeEncoder(TransitionTreeEncoder):
    def __iter__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # we enforce it
        self.encode_shared_relations = False

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        relations = super().get_relations(domain)
        # +1 for atom change object
        return [(rel, arity + 1) for rel, arity in relations]


class EffectsEncoder(Encoder):
    """Encoder for planning states.

    This encoder transforms a planning state (which contains atoms/facts that are
    currently true) into nodes and relations for the graph neural network.
    Objects become nodes, and atoms become relations between objects.
    """

    def __init__(
        self,
        suffix: str = "",
        encode_shared_relations: bool = False,
        add_target_state_handle: bool = False,
        connect_source_and_target_handle: bool = False,
    ) -> None:
        """Initialize the state encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), "Suffix must be a string."
        self.suffix = suffix
        self.add_target_state_handle = add_target_state_handle
        self.connect_source_and_target_handle = connect_source_and_target_handle
        self.parent_relation_name = "relation_source_to_target"
        self.encode_shared_relations = encode_shared_relations
        self.shared_label = "_shared"

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for state atoms.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for all predicates except 'number'.
        """
        ignored_predicate_names = ["number"]
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        relations = []
        relations.extend(
            (
                get_effect_name(predicate, positive, affects_goal, self.suffix),
                predicate.get_arity() + 1 * self.add_target_state_handle,
            )
            for positive, affects_goal in product([True, False], repeat=2)
            for predicate in predicates
        )
        if self.connect_source_and_target_handle:
            relations.append((self.parent_relation_name, 2))
        if self.encode_shared_relations:
            relations.extend(
                (
                    get_shared_atom_name(
                        predicate, self.shared_label, self.suffix
                    ),
                    predicate.get_arity(),
                )
                for predicate in domain.get_predicates()
                if predicate.get_name() not in ignored_predicate_names
            )
        return relations

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "EncodingContext",
    ) -> None:
        effects, target_atoms = input_value
        assert isinstance(effects, list), "Effects must be a list."
        problem = state.get_problem()
        goal_condition = problem.get_goal_condition()

        if getattr(self, "add_target_state_handle", False):
            target_handle = context.new_action_id()
            base_obj_ids = [target_handle]
        else:
            target_handle = None
            base_obj_ids = []
        # encode each literal in the transition
        for effect_literal in effects:
            assert isinstance(effect_literal, mm.GroundLiteral), (
                "Expected a list of ground literals."
            )
            effect_atom = effect_literal.get_atom()
            effect_name = get_effect_name(
                effect_atom.get_predicate(),
                effect_literal.get_polarity(),
                False,
                self.suffix,
            )
            object_ids = base_obj_ids + [
                context.get_object_id(obj.get_index())
                for obj in effect_atom.get_terms()
            ]
            encoding.flattened_relations.setdefault(effect_name, []).extend(
                object_ids
            )

            # Add literals stating how this transition affects the goal
            for goal_literal in goal_condition:
                assert isinstance(goal_literal, mm.GroundLiteral), (
                    "Goal condition should contain ground literals."
                )
                assert goal_literal.get_polarity(), (
                    "Only positive literals are supported in the goal condition."
                )
                if effect_atom == goal_literal.get_atom():
                    goal_effect_name = get_effect_name(
                        effect_atom.get_predicate(),
                        effect_literal.get_polarity(),
                        True,
                        self.suffix,
                    )
                    goal_object_ids = base_obj_ids + [
                        context.get_object_id(obj.get_index())
                        for obj in effect_atom.get_terms()
                    ]
                    encoding.flattened_relations.setdefault(
                        goal_effect_name, []
                    ).extend(goal_object_ids)
                    break  # No need to check other goal literals
        if getattr(self, "connect_source_and_target_handle", False):
            assert len(context.action_ids) == 2, (
                "Expected only source and target state handle ids."
            )
            assert context.action_ids[0] != target_handle, (
                "Expected target state handle to be 2nd state handle. "
                "Check that the source handle been added in previous encoder. "
            )
            encoding.flattened_relations[self.parent_relation_name] = (
                context.action_ids
            )
        if getattr(self, "encode_shared_relations", False):
            shared_atoms = set(state.get_atoms(ignore_static=True)) & set(
                target_atoms
            )
            for atom in shared_atoms:
                relation_name = get_shared_atom_name(
                    atom.get_predicate(), self.shared_label, self.suffix
                )
                object_indices: list[int] = [
                    context.get_object_id(obj.get_index())
                    for obj in atom.get_terms()
                ]
                encoding.flattened_relations.setdefault(
                    relation_name, []
                ).extend(object_indices)


class LiberalGoalEncoder(Encoder):
    """Encoder for goal conditions.

    This encoder transforms a goal condition (conjunctive condition of literals
    that must be satisfied) into relations for the graph neural network.
    It creates both goal-specific relations and marks which atoms are true/false
    in the current state relative to the goal.
    """

    def __init__(self, suffix: str = "") -> None:
        """Initialize the goal encoder.

        Args:
            suffix: Suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), "Suffix must be a string."
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for goal conditions.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for goal predicates,
            including both true and false variants.
        """
        ignored_predicate_names = ["number"]
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        relations = []
        relations.extend(
            [
                (
                    get_predicate_name(predicate, True, False, self.suffix),
                    predicate.get_arity(),
                )
                for predicate in predicates
            ]
        )
        relations.extend(
            [
                (
                    get_predicate_name(predicate, True, True, self.suffix),
                    predicate.get_arity(),
                )
                for predicate in predicates
            ]
        )
        return relations

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "EncodingContext",
    ) -> None:
        assert isinstance(input_value, mm.GroundConjunctiveCondition), (
            f"GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}"
        )

        for literal in input_value:  # type: ignore
            assert isinstance(literal, mm.GroundLiteral), (
                "Goal condition should contain ground literals."
            )
            if not literal.get_polarity():
                continue
            atom = literal.get_atom()
            relation_name = get_atom_name(atom, state, True, self.suffix)
            object_indices: list[int] = [
                context.get_object_id(obj.get_index())
                for obj in atom.get_terms()
            ]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = object_indices
            else:
                encoding.flattened_relations[relation_name].extend(
                    object_indices
                )


from typing import Any, Callable


JOINT_PARENT_RELATION = "__joint_parent__"


class JointTransitionActionEncoder(Encoder):
    """Encoder for joint (IW successor, continuation action) handles.

    Each readout node represents one pair (s', a), where s' is an IW successor
    state and a is one primitive action applicable in s'. The same node carries
    both the delta(s') relations and the grounded action relation for a.
    """

    def __init__(
        self,
        suffix: str = "",
        depth_config: IWDepthConfig | None = None,
    ) -> None:
        super().__init__()
        assert isinstance(suffix, str), "Suffix must be a string."
        self.suffix = suffix
        self.depth_config = (
            IWDepthConfig() if depth_config is None else depth_config.canonicalized()
        )

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        ignored_predicate_names = ["number"]
        predicates = [
            predicate
            for predicate in domain.get_predicates()
            if predicate.get_name() not in ignored_predicate_names
        ]
        relations: list[tuple[str, int]] = []

        # Effect relations, exactly like TransitionEffectsStateEncoder.
        relations.extend(
            [
                (
                    get_effect_name(predicate, True, False, self.suffix),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
            ]
        )
        relations.extend(
            [
                (
                    get_effect_name(predicate, False, False, self.suffix),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
            ]
        )
        relations.extend(
            [
                (
                    get_effect_name(predicate, True, True, self.suffix),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
            ]
        )
        relations.extend(
            [
                (
                    get_effect_name(predicate, False, True, self.suffix),
                    predicate.get_arity() + 1,
                )
                for predicate in predicates
            ]
        )

        # Grounded primitive action relations, exactly like GroundActionsAuxiliaryEncoder.
        relations.extend(
            (get_action_name(action, self.suffix), action.get_arity() + 1)
            for action in domain.get_actions()
        )

        # Duplicated pairwise parent edges between joint handles.
        relations.append((JOINT_PARENT_RELATION + self.suffix, 2))
        if self.depth_config.uses_depth_nodes:
            relations.append((IW_IN_DEPTH_RELATION, 2))
            if self.depth_config.precedes_mode == IW_DEPTH_PRECEDES_MLP:
                relations.append((IW_DEPTH_PRECEDES_RELATION, 2))
        return relations

    def _new_joint_id(
        self,
        context: "EncodingContext",
        depth: float,
        depth_mask: bool,
    ) -> int:
        # Depth-aware models use ExtendedEncodingContext, otherwise fall back
        # to the default action-id creation.
        if isinstance(context, ExtendedEncodingContext):
            if self.depth_config.uses_depth_nodes:
                return context.new_action_id()
            return context.new_action_id(depth=depth, depth_mask=depth_mask)
        return context.new_action_id()

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: "EncodedLists",
        context: "EncodingContext",
    ) -> None:
        assert isinstance(input_value, tuple), (
            f"JointTransitionActionEncoder expected a tuple, got {type(input_value)}"
        )
        assert 3 <= len(input_value) <= 4, (
            "JointTransitionActionEncoder expects "
            "(effects_list, effects_relations, successor_actions[, transition_depths])."
        )

        effects_list = input_value[0]
        effects_relations = input_value[1]
        successor_actions = input_value[2]
        transition_depths = (
            input_value[3]
            if len(input_value) == 4
            else [0.0 for _ in range(len(effects_list))]
        )

        assert isinstance(effects_list, list), "Expected list of IW effects."
        assert isinstance(effects_relations, list), "Expected list of IW edges."
        assert isinstance(successor_actions, list), (
            "Expected list of per-IW-state successor action lists."
        )
        assert len(effects_list) == len(successor_actions), (
            "effects_list and successor_actions must have the same length."
        )
        assert len(transition_depths) == len(effects_list), (
            "transition_depths and effects_list must have the same length."
        )

        problem = state.get_problem()
        goal_condition = problem.get_goal_condition()

        # joint_ids_by_iw_index[i] contains all joint node ids for pairs (s'_i, a)
        joint_ids_by_iw_index: list[list[int]] = []
        raw_transition_depths = [int(depth) for depth in transition_depths]
        if self.depth_config.uses_depth_nodes:
            for depth in transition_depths:
                if int(depth) != float(depth):
                    raise RuntimeError(
                        "depth_nodes mode expects raw integer IW depths."
                    )
        depth_attachments: list[tuple[int, int]] = []

        # Create one joint node per (IW successor, primitive continuation action).
        for iw_index, (effects, actions, depth) in enumerate(
            zip(effects_list, successor_actions, transition_depths)
        ):
            assert isinstance(effects, (list, tuple)), (
                "Expected a list of lists of ground literals."
            )
            assert isinstance(actions, list), (
                "Expected successor_actions entries to be lists of ground actions."
            )

            joint_ids: list[int] = []
            for action in actions:
                assert isinstance(action, mm.GroundAction), (
                    f"Expected a GroundAction in successor_actions, got {type(action)}"
                )

                joint_id = self._new_joint_id(
                    context, float(depth), depth_mask=True
                )
                joint_ids.append(joint_id)
                if self.depth_config.uses_depth_nodes:
                    depth_attachments.append(
                        (raw_transition_depths[iw_index], joint_id)
                    )

                # Attach primitive action relation to the same node.
                action_relation_name = get_action_name(action, self.suffix)
                action_object_ids = [joint_id] + [
                    context.get_object_id(obj.get_index())
                    for obj in action.get_objects()
                ]
                encoding.flattened_relations.setdefault(
                    action_relation_name, []
                ).extend(action_object_ids)

                # Attach delta(s') relations to the same node.
                for effect_literal in effects:
                    assert isinstance(effect_literal, mm.GroundLiteral), (
                        "Expected a list of lists of ground literals."
                    )
                    effect_atom = effect_literal.get_atom()
                    effect_name = get_effect_name(
                        effect_atom.get_predicate(),
                        effect_literal.get_polarity(),
                        False,
                        self.suffix,
                    )
                    effect_object_ids = [joint_id] + [
                        context.get_object_id(obj.get_index())
                        for obj in effect_atom.get_terms()
                    ]
                    encoding.flattened_relations.setdefault(
                        effect_name, []
                    ).extend(effect_object_ids)

                    # Goal-affecting effect relation, same logic as TransitionEffectsStateEncoder.
                    for goal_literal in goal_condition:
                        assert isinstance(goal_literal, mm.GroundLiteral), (
                            "Goal condition should contain ground literals."
                        )
                        assert goal_literal.get_polarity(), (
                            "Only positive literals are supported in the goal condition."
                        )
                        if effect_atom == goal_literal.get_atom():
                            goal_effect_name = get_effect_name(
                                effect_atom.get_predicate(),
                                effect_literal.get_polarity(),
                                True,
                                self.suffix,
                            )
                            goal_object_ids = [joint_id] + [
                                context.get_object_id(obj.get_index())
                                for obj in effect_atom.get_terms()
                            ]
                            encoding.flattened_relations.setdefault(
                                goal_effect_name, []
                            ).extend(goal_object_ids)
                            break

            joint_ids_by_iw_index.append(joint_ids)

        # Duplicate pairwise tree edges:
        # if IW tree has i -> j, add parent(o_{i,a}, o_{j,a'}) for all a in A[s'_i], a' in A[s'_j].
        parent_relation_name = JOINT_PARENT_RELATION + self.suffix
        for parent_iw_index, child_iw_index in effects_relations:
            assert isinstance(parent_iw_index, int) and isinstance(
                child_iw_index, int
            ), "Effect relations must be pairs of integers."
            assert 0 <= parent_iw_index < len(joint_ids_by_iw_index), (
                f"Invalid parent IW index {parent_iw_index}."
            )
            assert 0 <= child_iw_index < len(joint_ids_by_iw_index), (
                f"Invalid child IW index {child_iw_index}."
            )

            for parent_joint_id, child_joint_id in product(
                joint_ids_by_iw_index[parent_iw_index],
                joint_ids_by_iw_index[child_iw_index],
            ):
                encoding.flattened_relations.setdefault(
                    parent_relation_name, []
                ).extend([parent_joint_id, child_joint_id])
        _encode_depth_node_structure(
            context=context,
            encoding=encoding,
            depth_config=self.depth_config,
            realized_depths=raw_transition_depths,
            attachments=depth_attachments,
        )
