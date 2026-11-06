from __future__ import annotations

from typing import Any, Sequence

import pymimir as mm
import torch

from .lgan import LGANConfig
from .rw_return import RW_WALK_NON_RETURNING, compute_rw_return_features_from_csr

try:
    from . import _iw_encoding_backend as _native_backend_module
    _native_backend_import_error: ImportError | None = None
except ImportError as exc:
    _native_backend_module: Any = None
    _native_backend_import_error = exc

_native_backend: Any = _native_backend_module


NATIVE_BACKEND_AVAILABLE = _native_backend is not None
NATIVE_BATCH_ENCODING_AVAILABLE = bool(
    _native_backend is not None
    and hasattr(_native_backend, "encode_transition_tree_batch")
)
NATIVE_FLAT_BATCH_ENCODING_AVAILABLE = bool(
    _native_backend is not None and hasattr(_native_backend, "encode_flat_batch")
)


def _object_call(obj: Any, attr: str) -> Any:
    value = getattr(obj, attr)
    return value() if callable(value) else value


def _advanced_attr_or_self(obj: Any, attr: str) -> Any:
    return getattr(obj, attr, obj)


def _problem_wrapper_from_state(state: mm.State) -> mm.Problem:
    problem = getattr(state, "_problem", None)
    if problem is not None:
        return problem
    get_problem = getattr(state, "get_problem", None)
    if callable(get_problem):
        return get_problem()
    raise AttributeError("State wrapper does not expose a problem wrapper.")


def _require_native_backend(function_name: str, attribute: str | None = None) -> Any:
    if _native_backend is None:
        detail = ""
        if _native_backend_import_error is not None:
            detail = (
                " Original import error: "
                f"{type(_native_backend_import_error).__name__}: "
                f"{_native_backend_import_error}"
            )
        raise RuntimeError(
            f"{function_name} requires the native striplan backend, but "
            "striplan._iw_encoding_backend could not be imported."
            f"{detail}"
        )
    if attribute is not None and not hasattr(_native_backend, attribute):
        raise RuntimeError(
            f"{function_name} requires native backend attribute {attribute!r}, "
            "but it is not available."
        )
    return _native_backend


def assert_native_backend_ready(context: str = "runtime") -> None:
    _require_native_backend(context)


def native_backend_status() -> dict[str, Any]:
    backend = _native_backend
    import_error = _native_backend_import_error
    return {
        "available": backend is not None,
        "batch_encoding_available": bool(
            backend is not None and hasattr(backend, "encode_transition_tree_batch")
        ),
        "flat_batch_encoding_available": bool(
            backend is not None and hasattr(backend, "encode_flat_batch")
        ),
        "module_name": "striplan._iw_encoding_backend",
        "module_file": getattr(backend, "__file__", None),
        "import_error_type": type(import_error).__name__ if import_error is not None else None,
        "import_error": str(import_error) if import_error is not None else None,
    }


def create_iw_expansion(
    *,
    start_state: mm.State,
    final_states: Sequence[mm.State],
    underlying_actions_by_transition: Sequence[Sequence[mm.GroundAction]],
    iw_edges: Sequence[tuple[int, mm.GroundAction, int]],
    transition_depths: Sequence[float | int],
    include_root_handle: bool,
) -> Any:
    native_backend = _require_native_backend("create_iw_expansion", "create_iw_expansion")
    return native_backend.create_iw_expansion(
        start_state=_advanced_attr_or_self(start_state, "_advanced_state"),
        start_state_wrapper=start_state,
        final_states=[
            _advanced_attr_or_self(state, "_advanced_state") for state in final_states
        ],
        underlying_action_lists=[list(actions) for actions in underlying_actions_by_transition],
        iw_edges=list(iw_edges),
        transition_depths=list(transition_depths),
        include_root_handle=include_root_handle,
    )


def _tensor_from_native_array(value: Any, device: Any) -> torch.Tensor:
    if hasattr(torch, "from_dlpack") and hasattr(value, "__dlpack__"):
        tensor = torch.from_dlpack(value)
    else:
        tensor = torch.utils.dlpack.from_dlpack(value)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _bool_tensor_from_native_array(value: Any, device: Any) -> torch.Tensor:
    return _tensor_from_native_array(value, device).to(dtype=torch.bool)


def _extended_tensors_from_native_batch(batch: Any, device: Any) -> Any:
    from .encoders import ExtendedEncodedTensors

    tensors = ExtendedEncodedTensors()
    tensors.flattened_relations = {
        name: _tensor_from_native_array(values, device)
        for name, values in batch.flattened_relations().items()
    }
    tensors.node_count = batch.node_count()
    tensors.node_sizes = _tensor_from_native_array(batch.node_sizes(), device)
    tensors.object_indices = _tensor_from_native_array(batch.object_indices(), device)
    tensors.object_sizes = _tensor_from_native_array(batch.object_sizes(), device)
    tensors.action_indices = _tensor_from_native_array(batch.action_indices(), device)
    tensors.root_action_indices = _tensor_from_native_array(
        batch.root_action_indices(), device
    )
    tensors.action_sizes = _tensor_from_native_array(batch.action_sizes(), device)
    tensors.action_depths = _tensor_from_native_array(batch.action_depths(), device)
    tensors.action_depth_mask = _bool_tensor_from_native_array(
        batch.action_depth_mask(), device
    )
    tensors.auxiliary_indices = _tensor_from_native_array(
        batch.auxiliary_indices(), device
    )
    tensors.auxiliary_sizes = _tensor_from_native_array(batch.auxiliary_sizes(), device)
    tensors.virtual_indices = _tensor_from_native_array(batch.virtual_indices(), device)
    tensors.virtual_sizes = _tensor_from_native_array(batch.virtual_sizes(), device)
    tensors.depth_node_indices = _tensor_from_native_array(
        batch.depth_node_indices(), device
    )
    tensors.depth_node_sizes = _tensor_from_native_array(
        batch.depth_node_sizes(), device
    )
    tensors.depth_node_depths = _tensor_from_native_array(
        batch.depth_node_depths(), device
    )
    tensors.lgan_tn_relation_indices = _tensor_from_native_array(
        batch.lgan_tn_relation_indices(), device
    )
    tensors.lgan_tn_entity_indices = _tensor_from_native_array(
        batch.lgan_tn_entity_indices(), device
    )
    tensors.lgan_nn_relation_indices = _tensor_from_native_array(
        batch.lgan_nn_relation_indices(), device
    )
    tensors.lgan_nn_entity_indices = _tensor_from_native_array(
        batch.lgan_nn_entity_indices(), device
    )
    tensors.lgan_rr_src_relation_indices = _tensor_from_native_array(
        batch.lgan_rr_src_relation_indices(), device
    )
    tensors.lgan_rr_dst_relation_indices = _tensor_from_native_array(
        batch.lgan_rr_dst_relation_indices(), device
    )
    tensors.lgan_tn_sizes = _tensor_from_native_array(batch.lgan_tn_sizes(), device)
    tensors.lgan_nn_sizes = _tensor_from_native_array(batch.lgan_nn_sizes(), device)
    tensors.lgan_rr_sizes = _tensor_from_native_array(batch.lgan_rr_sizes(), device)
    return tensors


def encode_flat_batch(
    *,
    batch_instances: Sequence[tuple[Any, ...]],
    mode: str,
    state_positions: Sequence[int],
    state_suffixes: Sequence[str],
    goal_position: int,
    goal_suffix: str,
    effects_position: int,
    effects_suffix: str,
    actions_position: int,
    actions_suffix: str,
    device: Any,
) -> Any:
    native_backend = _require_native_backend("encode_flat_batch", "encode_flat_batch")
    spec = native_backend.FlatBatchEncodingSpec(
        mode=mode,
        state_positions=list(state_positions),
        state_suffixes=list(state_suffixes),
        goal_position=goal_position,
        goal_suffix=goal_suffix,
        effects_position=effects_position,
        effects_suffix=effects_suffix,
        actions_position=actions_position,
        actions_suffix=actions_suffix,
    )
    batch = native_backend.encode_flat_batch(list(batch_instances), spec, device)
    return _extended_tensors_from_native_batch(batch, device)


def generate_iw_expansion_native(
    *,
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
) -> Any:
    native_backend = _require_native_backend(
        "generate_iw_expansion_native", "generate_iw_expansion"
    )
    problem_wrapper = _problem_wrapper_from_state(start_state)
    return native_backend.generate_iw_expansion(
        start_state=_advanced_attr_or_self(start_state, "_advanced_state"),
        search_context=problem_wrapper._search_context,
        start_state_wrapper=start_state,
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


def encode_transition_tree_batch(
    *,
    batch_instances: Sequence[tuple[mm.State, Any, mm.GroundConjunctiveCondition]],
    tree_encoder: Any,
    device: Any,
    filter_out_root_action_ids: bool,
) -> Any:
    native_backend = _require_native_backend(
        "encode_transition_tree_batch", "encode_transition_tree_batch"
    )
    native_batch_instances = [
        (
            _advanced_attr_or_self(state, "_advanced_state"),
            expansion,
            _advanced_attr_or_self(goal_condition, "_advanced_condition"),
        )
        for state, expansion, goal_condition in batch_instances
    ]
    depth_config_native = native_backend.DepthConfigNative()
    depth_config_native.mode = tree_encoder.depth_config.mode
    depth_config_native.precedes_mode = tree_encoder.depth_config.precedes_mode
    depth_config_native.precedes_scan = tree_encoder.depth_config.precedes_scan
    rw_return_config_native = native_backend.RWReturnConfigNative()
    rw_return_config_native.mode = tree_encoder.rw_return_config.mode
    rw_return_config_native.walk_mode = tree_encoder.rw_return_config.walk_mode
    rw_return_config_native.k_max = tree_encoder.rw_return_config.k_max
    rw_return_config_native.include_depth_nodes = (
        tree_encoder.rw_return_config.include_depth_nodes
    )
    rw_return_config_native.lazy_walk = tree_encoder.rw_return_config.lazy_walk
    rw_return_config_native.lazy_beta = tree_encoder.rw_return_config.lazy_beta
    rw_return_config_native.block_size = tree_encoder.rw_return_config.block_size
    lgan_config = getattr(tree_encoder, "lgan_config", LGANConfig())
    lgan_config_native = native_backend.LganConfigNative()
    lgan_config_native.include_lgan_edges = lgan_config.include_lgan_edges
    lgan_config_native.anchor_sources = list(lgan_config.lgan_anchor_sources)
    lgan_config_native.lgan_tn_edge_pos = lgan_config.lgan_tn_edge_pos
    lgan_config_native.lgan_nn_edge_pos = lgan_config.lgan_nn_edge_pos
    lgan_config_native.lgan_rr_edge_pos = lgan_config.lgan_rr_edge_pos
    spec = native_backend.TransitionTreeEncodingSpec(
        suffix=tree_encoder.suffix,
        connect_to_root_handle=tree_encoder.connect_to_root_handle,
        encode_actions=tree_encoder.encode_actions,
        encode_effects=tree_encoder.encode_effects,
        encode_parent_relations=getattr(tree_encoder, "encode_parent_relations", True),
        encode_shared_relations=tree_encoder.encode_shared_relations,
        shared_label=tree_encoder.shared_label,
        separate_action_nodes=tree_encoder.separate_action_nodes,
        depth_config=depth_config_native,
        rw_return_config=rw_return_config_native,
        lgan_config=lgan_config_native,
    )
    batch = native_backend.encode_transition_tree_batch(
        native_batch_instances,
        spec,
        device,
        filter_out_root_action_ids,
    )
    tensors = _extended_tensors_from_native_batch(batch, device)
    if tree_encoder.rw_return_config.enabled:
        import pymimir_rgnn as rgnn

        from .encoders import EnhancedStateEncoder, _attach_python_rw_return_features

        state_encoder: Any
        if tree_encoder.connect_to_root_handle:
            state_encoder = EnhancedStateEncoder(add_source_state_handle=True)
        else:
            state_encoder = rgnn.StateEncoder()
        _attach_python_rw_return_features(
            encoding_tensors=tensors,
            input=list(batch_instances),
            input_specification=(state_encoder, tree_encoder, rgnn.GoalEncoder()),
        )
    return tensors


def is_iw_expansion(value: Any) -> bool:
    return all(
        hasattr(value, attr)
        for attr in ("transitions_py", "parent_edges_py", "includes_root_handle")
    )


def is_native_iw_expansion(value: Any) -> bool:
    native_expansion_type = getattr(_native_backend, "IwExpansion", None)
    return native_expansion_type is not None and isinstance(value, native_expansion_type)


def expansion_transitions(expansion: Any) -> list[Any]:
    return list(_object_call(expansion, "transitions_py"))


def expansion_parent_edges(expansion: Any) -> list[Any]:
    return list(_object_call(expansion, "parent_edges_py"))


def expansion_includes_root_handle(expansion: Any) -> bool:
    return bool(_object_call(expansion, "includes_root_handle"))


def transition_index(transition: Any) -> int:
    return int(_object_call(transition, "index"))


def transition_underlying_actions(transition: Any) -> list[mm.GroundAction]:
    return list(_object_call(transition, "underlying_actions_py"))


def transition_effect_literals(transition: Any) -> list[mm.GroundLiteral]:
    return list(_object_call(transition, "effect_literals_py"))


def transition_raw_depth(transition: Any) -> int:
    return int(_object_call(transition, "raw_depth"))


def transition_depth_feature_value(transition: Any) -> float | int:
    if hasattr(transition, "depth_feature_value"):
        return _object_call(transition, "depth_feature_value")
    return float(_object_call(transition, "normalized_depth"))


def parent_edge_src_index(edge: Any) -> int:
    return int(_object_call(edge, "src_transition_index"))


def parent_edge_incoming_action(edge: Any) -> mm.GroundAction:
    return _object_call(edge, "incoming_action_py")


def parent_edge_tgt_index(edge: Any) -> int:
    return int(_object_call(edge, "tgt_transition_index"))
