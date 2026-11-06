from functools import partial
from dataclasses import dataclass

from .iw import generate_iw_expansion
from .encoders import *
from .iw_depth import (
    IWDepthConfig,
    IW_DEPTH_NODES,
    resolve_iw_depth_config,
)
from .lgan import LGANConfig
from .rw_return import (
    RWReturnConfig,
    RW_RETURN_NONE,
    resolve_rw_return_config,
)
import pymimir as mm
import pymimir_rgnn as rgnn
import pymimir_rl as rl
import time
import torch


from .iw import IWGroundAction, generate_iw_actions, JointIWGroundAction

from collections import deque
from pathlib import Path
from typing import Any, Sequence

IW_TREE_ENCODINGS = {
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
    "concurrent_internal_action_hybrid_tree",
    "concurrent_internal_joint_actions",
}
EXTENDED_ENCODINGS = {
    "concurrent_internal",
    "concurrent_internal_tree",
    "concurrent_internal_tree_rooted",
    "concurrent_internal_comparison_tree",
    "concurrent_internal_action_tree",
    "concurrent_internal_action_hybrid_tree",
    "concurrent_internal_actions",
    "concurrent_internal_actions_source_action",
    "concurrent_internal_actions_action_successor",
    "concurrent_internal_actions_source_action_successor",
    "concurrent_internal_actions_source_action_rooted",
    "concurrent_internal_actions_action_successor_rooted",
    "concurrent_internal_actions_source_action_successor_rooted",
    "sequential_delta_rooted",
    "sequential_comparison",
    "sequential_aux",
}

RW_RETURN_SUPPORTED_ENCODINGS = {
    "concurrent_internal_tree",
    "concurrent_internal_tree_rooted",
    "concurrent_internal_comparison_tree",
    "concurrent_internal_action_tree",
    "concurrent_internal_action_hybrid_tree",
    "concurrent_internal_actions",
    "concurrent_internal_actions_source_action",
    "concurrent_internal_actions_action_successor",
    "concurrent_internal_actions_source_action_successor",
    "concurrent_internal_actions_source_action_rooted",
    "concurrent_internal_actions_action_successor_rooted",
    "concurrent_internal_actions_source_action_successor_rooted",
}

LGAN_SUPPORTED_ENCODINGS = IW_TREE_ENCODINGS - {
    "concurrent_internal_joint_actions"
}

DEPTH_ORDER_SCORE_READOUT = "depth_order_score"


@dataclass(frozen=True)
class TransitionReadoutSpec:
    readout_name: str
    node_roles: tuple[str, ...] | None = None
    include_root_state: bool = False


CONCURRENT_INTERNAL_ACTIONS_READOUTS: dict[str, TransitionReadoutSpec] = {
    "concurrent_internal_actions": TransitionReadoutSpec("action_scalar"),
    "concurrent_internal_actions_source_action": TransitionReadoutSpec(
        "source_action_scalar", ("source", "action")
    ),
    "concurrent_internal_actions_action_successor": TransitionReadoutSpec(
        "action_successor_scalar", ("action", "successor")
    ),
    "concurrent_internal_actions_source_action_successor": TransitionReadoutSpec(
        "source_action_successor_scalar",
        ("source", "action", "successor"),
    ),
    "concurrent_internal_actions_source_action_rooted": TransitionReadoutSpec(
        "rooted_source_action_scalar", ("source", "action"), True
    ),
    "concurrent_internal_actions_action_successor_rooted": TransitionReadoutSpec(
        "rooted_action_successor_scalar", ("action", "successor"), True
    ),
    "concurrent_internal_actions_source_action_successor_rooted": TransitionReadoutSpec(
        "rooted_source_action_successor_scalar",
        ("source", "action", "successor"),
        True,
    ),
}


# ---------------------------------------------------
# Create different models based on the encoding type.
# ---------------------------------------------------


def create_aggregation_function(name: str) -> rgnn.AggregationFunction:
    if name == "smax":
        return rgnn.SmoothMaximumAggregation()
    elif name == "hmax":
        return rgnn.HardMaximumAggregation()
    elif name == "mean":
        return rgnn.MeanAggregation()
    elif name == "add":
        return rgnn.SumAggregation()
    else:
        raise RuntimeError(f"Unknown aggregation function: {name}.")


def _aggregation_name_from_function(
    aggregation: rgnn.AggregationFunction,
) -> str:
    if isinstance(aggregation, rgnn.SmoothMaximumAggregation):
        return "smax"
    if isinstance(aggregation, rgnn.HardMaximumAggregation):
        return "hmax"
    if isinstance(aggregation, rgnn.MeanAggregation):
        return "mean"
    if isinstance(aggregation, rgnn.SumAggregation):
        return "add"
    raise RuntimeError(
        f"Unknown aggregation function type: {type(aggregation)!r}."
    )


def _validate_iw_depth_config(
    encoding: str,
    iw_depth_config: IWDepthConfig,
    aggregation_name: str | None = None,
) -> IWDepthConfig:
    return iw_depth_config.validate(
        encoding=encoding,
        tree_encodings=IW_TREE_ENCODINGS,
        aggregation_name=aggregation_name,
    )


def _validate_iw_depth_order_aux(
    iw_depth_config: IWDepthConfig,
    iw_depth_order_aux_loss: bool,
    iw_depth_order_aux_weight: float,
) -> None:
    if iw_depth_order_aux_weight < 0.0:
        raise RuntimeError(
            "iw_depth_order_aux_weight must be non-negative."
        )
    if iw_depth_order_aux_loss and not iw_depth_config.uses_depth_nodes:
        raise RuntimeError(
            "iw_depth_order_aux_loss=True requires iw_depth_mode='depth_nodes'."
        )


def _validate_rw_return_config(
    encoding: str,
    rw_return_config: RWReturnConfig | None,
) -> RWReturnConfig:
    config = (
        RWReturnConfig()
        if rw_return_config is None
        else rw_return_config.canonicalized()
    )
    if config.enabled and encoding not in RW_RETURN_SUPPORTED_ENCODINGS:
        raise RuntimeError(
            f"rw-return features are not supported for encoding={encoding!r}."
        )
    return config


def _validate_lgan_config(
    encoding: str,
    lgan_config: LGANConfig | None,
) -> LGANConfig:
    config = LGANConfig() if lgan_config is None else lgan_config.canonicalized()
    if config.include_lgan_edges and encoding not in LGAN_SUPPORTED_ENCODINGS:
        raise RuntimeError(
            f"LGAN support is currently only wired for tree encodings, got encoding={encoding!r}."
        )
    return config


def _hydrate_legacy_transition_tree_input_spec(
    input_spec: tuple[Any, ...],
    iw_depth_config: IWDepthConfig,
    rw_return_config: RWReturnConfig,
    lgan_config: LGANConfig,
) -> tuple[Any, ...]:
    for encoder in input_spec:
        if isinstance(encoder, TransitionTreeEncoder):
            encoder.depth_config = iw_depth_config
            encoder.rw_return_config = rw_return_config
            encoder.lgan_config = lgan_config
    return input_spec


def _tree_model_class(
    encoding: str,
    iw_depth_config: IWDepthConfig,
    rw_return_config: RWReturnConfig | None = None,
    lgan_config: LGANConfig | None = None,
) -> type[rgnn.RelationalGraphNeuralNetwork]:
    if iw_depth_config.mode != "none":
        return DepthAwareExtendedRGNN
    if rw_return_config is not None and rw_return_config.enabled:
        return DepthAwareExtendedRGNN
    if lgan_config is not None and lgan_config.enabled:
        return DepthAwareExtendedRGNN
    if encoding in EXTENDED_ENCODINGS:
        return ExtendedRGNN
    return rgnn.RelationalGraphNeuralNetwork


def _append_depth_order_output_spec(
    output_spec: list[tuple[str, rgnn.Decoder]],
    hparam_config: rgnn.HyperparameterConfig,
    iw_depth_config: IWDepthConfig,
    iw_depth_order_aux_loss: bool,
) -> list[tuple[str, rgnn.Decoder]]:
    if iw_depth_order_aux_loss and iw_depth_config.uses_depth_nodes:
        output_spec.append(
            (
                DEPTH_ORDER_SCORE_READOUT,
                DepthNodeScalarDecoder(hparam_config),
            )
        )
    return output_spec


def _create_tree_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    *,
    encoding: str,
    iw_depth_feature: bool,
    iw_depth_config: IWDepthConfig | None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
    encode_actions: bool,
    encode_effects: bool,
    encode_parent_relations: bool = True,
    encode_shared_relations: bool = False,
    connect_to_root_handle: bool = False,
) -> dict[str, torch.nn.Module]:
    resolved_iw_depth_config = _validate_iw_depth_config(
        encoding,
        resolve_iw_depth_config(
            iw_depth_feature=iw_depth_feature,
            iw_depth_mode=(
                None if iw_depth_config is None else iw_depth_config.mode
            ),
            iw_depth_precedes_mode=(
                None
                if iw_depth_config is None
                else iw_depth_config.precedes_mode
            ),
            iw_depth_precedes_scan=(
                None
                if iw_depth_config is None
                else iw_depth_config.precedes_scan
            ),
        ),
        aggregation_name,
    )
    resolved_rw_return_config = _validate_rw_return_config(
        encoding, rw_return_config
    )
    resolved_lgan_config = _validate_lgan_config(encoding, lgan_config)
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    state_encoder: rgnn.StateEncoder = (
        EnhancedStateEncoder(add_source_state_handle=True)
        if connect_to_root_handle
        else rgnn.StateEncoder()
    )
    input_spec = (
        state_encoder,
        TransitionTreeEncoder(
            connect_to_root_handle=connect_to_root_handle,
            encode_actions=encode_actions,
            encode_effects=encode_effects,
            encode_parent_relations=encode_parent_relations,
            encode_shared_relations=encode_shared_relations,
            depth_config=resolved_iw_depth_config,
            lgan_config=resolved_lgan_config,
            rw_return_config=resolved_rw_return_config,
        ),
        rgnn.GoalEncoder(),
    )
    output_spec = _append_depth_order_output_spec(
        [("action_scalar", rgnn.ActionScalarDecoder(hparam_config))],
        hparam_config,
        resolved_iw_depth_config,
        iw_depth_order_aux_loss,
    )
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model_class = _tree_model_class(
        encoding,
        resolved_iw_depth_config,
        resolved_rw_return_config,
        resolved_lgan_config,
    )
    if model_class is DepthAwareExtendedRGNN:
        model = model_class(
            hparam_config,
            module_config,
            input_spec,
            output_spec,
            resolved_iw_depth_config,
            resolved_rw_return_config,
            resolved_lgan_config,
        )
    else:
        model = model_class(
            hparam_config, module_config, input_spec, output_spec
        )  # type: ignore[arg-type]
    return {"internal_model": model}


def _create_action_scoring_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    *,
    encoding: str,
    iw_depth_feature: bool,
    iw_depth_config: IWDepthConfig | None,
    iw_depth_order_aux_loss: bool = False,
    input_spec: tuple[Any, ...],
) -> dict[str, torch.nn.Module]:
    resolved_iw_depth_config = _validate_iw_depth_config(
        encoding,
        resolve_iw_depth_config(
            iw_depth_feature=iw_depth_feature,
            iw_depth_mode=(
                None if iw_depth_config is None else iw_depth_config.mode
            ),
            iw_depth_precedes_mode=(
                None
                if iw_depth_config is None
                else iw_depth_config.precedes_mode
            ),
            iw_depth_precedes_scan=(
                None
                if iw_depth_config is None
                else iw_depth_config.precedes_scan
            ),
        ),
        aggregation_name,
    )
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    output_spec = _append_depth_order_output_spec(
        [("action_scalar", rgnn.ActionScalarDecoder(hparam_config))],
        hparam_config,
        resolved_iw_depth_config,
        iw_depth_order_aux_loss,
    )
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model_class = _tree_model_class(encoding, resolved_iw_depth_config)
    if model_class is DepthAwareExtendedRGNN:
        model = model_class(
            hparam_config,
            module_config,
            input_spec,
            output_spec,
            resolved_iw_depth_config,
        )
    else:
        model = model_class(
            hparam_config, module_config, input_spec, output_spec
        )
    return {"internal_model": model}


class TransitionPairScalarDecoder(rgnn.Decoder):
    """Readout for rooted IW tree edges using arbitrary handle tuples."""

    def __init__(
        self,
        hparam_config: rgnn.HyperparameterConfig,
        *,
        pair_mode: str | None = None,
        node_roles: tuple[str, ...] | None = None,
        include_root_state: bool = False,
    ) -> None:
        super().__init__()
        if node_roles is None:
            if pair_mode not in {"source_action", "action_successor"}:
                raise ValueError(f"Unsupported pair mode: {pair_mode!r}.")
            node_roles = (
                ("source", "action")
                if pair_mode == "source_action"
                else ("action", "successor")
            )
        else:
            if pair_mode is not None and pair_mode not in {
                "source_action",
                "action_successor",
            }:
                raise ValueError(f"Unsupported pair mode: {pair_mode!r}.")
            if not node_roles:
                raise ValueError("node_roles must not be empty.")
            if len(set(node_roles)) != len(node_roles):
                raise ValueError("node_roles must not contain duplicates.")
            invalid_roles = set(node_roles) - {
                "source",
                "action",
                "successor",
            }
            if invalid_roles:
                raise ValueError(
                    f"Unsupported node roles: {sorted(invalid_roles)!r}."
                )
        self._node_roles = node_roles
        self._include_root_state = include_root_state
        self._object_readout = rgnn.SumReadout(
            hparam_config.embedding_size, hparam_config.embedding_size
        )
        input_size = (
            len(self._node_roles) + (1 if self._include_root_state else 0) + 1
        ) * hparam_config.embedding_size
        self._transition_value = rgnn.MLP(
            input_size,
            1,
        )
        # self._transition_value = torch.nn.Sequential(
        #     ResidualMLPBlock(input_size, expansion=2),
        #     torch.nn.Linear(2 * input_size, 1),
        # )

    @staticmethod
    def _parent_relation_rows(encoding: Any) -> torch.Tensor:
        relation_name = rgnn.utils.get_effect_relation_name("")
        if relation_name not in encoding.flattened_relations:
            raise RuntimeError(
                "Transition pair decoding requires rooted parent relations."
            )
        relation_rows = encoding.flattened_relations[relation_name]
        if int(relation_rows.numel()) % 3 != 0:
            raise RuntimeError(
                "Rooted parent relations must be stored as ternary triples."
            )
        return relation_rows.view(-1, 3)

    def forward(
        self, node_embeddings: torch.Tensor, encoding: Any
    ) -> list[torch.Tensor]:
        if encoding.action_indices.numel() == 0:
            empty = torch.empty(
                0, dtype=node_embeddings.dtype, device=node_embeddings.device
            )
            return [
                empty.clone() for _ in range(int(encoding.action_sizes.numel()))
            ]
        parent_rows = self._parent_relation_rows(encoding)
        if parent_rows.shape[0] != int(encoding.action_indices.numel()):
            raise RuntimeError(
                "Rooted parent triples must align one-to-one with scored actions."
            )
        source_ids = parent_rows[:, 0].long()
        action_ids = parent_rows[:, 1].long()
        successor_ids = parent_rows[:, 2].long()
        if not torch.equal(action_ids, encoding.action_indices.long()):
            raise RuntimeError(
                "Action ids in the parent triples must align with action indices."
            )
        if successor_ids.numel() != int(encoding.auxiliary_indices.numel()):
            raise RuntimeError(
                "Each scored transition must have exactly one successor-state node."
            )
        if not torch.equal(encoding.auxiliary_sizes, encoding.action_sizes):
            raise RuntimeError(
                "Each scored transition must have exactly one successor-state node."
            )
        role_to_ids = {
            "source": source_ids,
            "action": action_ids,
            "successor": successor_ids,
        }
        object_embeddings = node_embeddings.index_select(
            0, encoding.object_indices.long()
        )
        aggregated_objects = self._object_readout.forward(
            object_embeddings, encoding.object_sizes
        )
        repeated_objects = aggregated_objects.repeat_interleave(
            encoding.action_sizes.long(), dim=0
        )
        readout_components = []
        if self._include_root_state:
            root_action_indices = getattr(encoding, "root_action_indices", None)
            if root_action_indices is None:
                raise RuntimeError(
                    "Root-aware transition decoding requires root action indices."
                )
            root_action_indices = root_action_indices.long()
            if root_action_indices.numel() != int(
                encoding.action_sizes.numel()
            ):
                raise RuntimeError(
                    "Root-aware transition decoding requires one root state handle per encoded instance."
                )
            root_embeddings = node_embeddings.index_select(
                0, root_action_indices
            )
            readout_components.append(
                root_embeddings.repeat_interleave(
                    encoding.action_sizes.long(), dim=0
                )
            )
        for role in self._node_roles:
            readout_components.append(
                node_embeddings.index_select(0, role_to_ids[role])
            )
        readout_components.append(repeated_objects)
        pair_embeddings = torch.cat(readout_components, dim=1)
        values = self._transition_value.forward(pair_embeddings).view(-1)
        return [
            action_values.view(-1)
            for action_values in values.split(encoding.action_sizes.tolist())
        ]


class DepthNodeScalarDecoder(rgnn.Decoder):
    def __init__(self, hparam_config: rgnn.HyperparameterConfig) -> None:
        super().__init__()
        self._depth_score = torch.nn.Linear(
            hparam_config.embedding_size, 1
        )

    def forward(
        self, node_embeddings: torch.Tensor, encoding: Any
    ) -> list[torch.Tensor]:
        depth_node_indices = getattr(encoding, "depth_node_indices", None)
        depth_node_sizes = getattr(encoding, "depth_node_sizes", None)
        if depth_node_indices is None or depth_node_sizes is None:
            raise RuntimeError(
                "Depth-node scoring requires depth node metadata in the encoding."
            )
        size_list = depth_node_sizes.tolist()
        if depth_node_indices.numel() == 0:
            return [
                torch.empty(
                    0,
                    dtype=node_embeddings.dtype,
                    device=node_embeddings.device,
                )
                for _ in size_list
            ]
        depth_embeddings = node_embeddings.index_select(
            0, depth_node_indices.long()
        )
        depth_scores = self._depth_score(depth_embeddings).view(-1)
        return [scores.view(-1) for scores in depth_scores.split(size_list)]


def _create_concurrent_internal_actions_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    *,
    encoding: str,
    iw_depth_feature: bool,
    iw_depth_config: IWDepthConfig | None,
    lgan_config: LGANConfig | None,
    rw_return_config: RWReturnConfig | None,
    iw_depth_order_aux_loss: bool,
    readout_spec: TransitionReadoutSpec,
) -> dict[str, torch.nn.Module]:
    resolved_iw_depth_config = _validate_iw_depth_config(
        encoding,
        resolve_iw_depth_config(
            iw_depth_feature=iw_depth_feature,
            iw_depth_mode=(
                None if iw_depth_config is None else iw_depth_config.mode
            ),
            iw_depth_precedes_mode=(
                None
                if iw_depth_config is None
                else iw_depth_config.precedes_mode
            ),
            iw_depth_precedes_scan=(
                None
                if iw_depth_config is None
                else iw_depth_config.precedes_scan
            ),
        ),
        aggregation_name,
    )
    resolved_rw_return_config = _validate_rw_return_config(
        encoding, rw_return_config
    )
    resolved_lgan_config = _validate_lgan_config(encoding, lgan_config)
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    input_spec = (
        EnhancedStateEncoder(add_source_state_handle=True),
        TransitionTreeEncoder(
            connect_to_root_handle=True,
            encode_actions=True,
            encode_effects=True,
            separate_action_nodes=True,
            depth_config=resolved_iw_depth_config,
            lgan_config=resolved_lgan_config,
            rw_return_config=resolved_rw_return_config,
        ),
        rgnn.GoalEncoder(),
    )
    if readout_spec.node_roles is None:
        output_spec = [
            (readout_spec.readout_name, rgnn.ActionScalarDecoder(hparam_config))
        ]
    else:
        output_spec = [
            (
                readout_spec.readout_name,
                TransitionPairScalarDecoder(
                    hparam_config,
                    node_roles=readout_spec.node_roles,
                    include_root_state=readout_spec.include_root_state,
                ),
            )
        ]
    output_spec = _append_depth_order_output_spec(
        output_spec,
        hparam_config,
        resolved_iw_depth_config,
        iw_depth_order_aux_loss,
    )
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model_class = _tree_model_class(
        encoding,
        resolved_iw_depth_config,
        resolved_rw_return_config,
        resolved_lgan_config,
    )
    if model_class is DepthAwareExtendedRGNN:
        model = model_class(
            hparam_config,
            module_config,
            input_spec,
            output_spec,
            resolved_iw_depth_config,
            resolved_rw_return_config,
            resolved_lgan_config,
        )
    else:
        model = model_class(
            hparam_config, module_config, input_spec, output_spec
        )
    return {"internal_model": model}


def create_concurrent_internal_action_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_tree_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_action_tree",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        encode_actions=True,
        encode_effects=False,
    )


def create_concurrent_internal_action_hybrid_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_tree_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_action_hybrid_tree",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        encode_actions=True,
        encode_effects=True,
    )


def create_concurrent_internal_comparison_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_tree_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_comparison_tree",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        encode_actions=False,
        encode_effects=True,
        encode_shared_relations=True,
    )


def create_concurrent_internal_actions_model_flat(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    input_spec = (
        rgnn.StateEncoder(),
        TransitionEffectsStateEncoder(),
        rgnn.GoalEncoder(),
        rgnn.GroundActionsEncoder(),
    )
    return _create_action_scoring_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_actions_flat",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        input_spec=input_spec,
    )


def create_concurrent_internal_joint_actions_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    """Joint-handle variant with one readout node per (IW successor, action)."""
    resolved_iw_depth_config = resolve_iw_depth_config(
        iw_depth_feature=iw_depth_feature,
        iw_depth_mode=(
            None if iw_depth_config is None else iw_depth_config.mode
        ),
        iw_depth_precedes_mode=(
            None if iw_depth_config is None else iw_depth_config.precedes_mode
        ),
        iw_depth_precedes_scan=(
            None if iw_depth_config is None else iw_depth_config.precedes_scan
        ),
    )
    input_spec = (
        rgnn.StateEncoder(),
        JointTransitionActionEncoder(depth_config=resolved_iw_depth_config),
        rgnn.GoalEncoder(),
    )
    return _create_action_scoring_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_joint_actions",
        iw_depth_feature=resolved_iw_depth_config.uses_init_scalar,
        iw_depth_config=resolved_iw_depth_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        input_spec=input_spec,
    )


# def create_concurrent_internal_extended_model(
#     domain: mm.Domain,
#     embedding_size: int,
#     num_layers: int,
#     aggregation_name: str,
#     binary_embeddings: bool,
# ) -> dict[str, torch.nn.Module]:
#     hparam_config = rgnn.HyperparameterConfig(
#         domain=domain,
#         embedding_size=embedding_size,
#         num_layers=num_layers,
#         binarize_updates=binary_embeddings,
#         residual_updates=not binary_embeddings,
#         normalize_updates=not binary_embeddings,
#     )
#     input_spec = (
#         ExtendedStateEncoder(),
#         DifferentTransitionTreeEncoder(
#             encode_actions=False,
#             encode_effects=True,
#             encode_shared_relations=True,
#         ),
#         rgnn.GoalEncoder(),
#     )
#     output_spec = [("action_scalar", rgnn.ActionScalarDecoder(hparam_config))]
#     module_config = rgnn.ModuleConfig(
#         aggregation_function=create_aggregation_function(aggregation_name),
#         message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
#         update_function=rgnn.MLPUpdates(hparam_config),
#     )
#     model = rgnn.RelationalGraphNeuralNetwork(
#         hparam_config, module_config, input_spec, output_spec
#     )  # type: ignore
#     return {"internal_model": model}


def create_concurrent_internal_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_tree_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        encode_actions=False,
        encode_effects=True,
        encode_parent_relations=False,
    )


def create_concurrent_internal_rooted_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_tree_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_tree_rooted",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        encode_actions=True,
        encode_effects=True,
        connect_to_root_handle=True,
    )


def create_concurrent_internal_tree_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_tree_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_tree",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        encode_actions=False,
        encode_effects=True,
    )


def create_sequential_internal_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    input_spec = (
        rgnn.StateEncoder("_current"),
        rgnn.StateEncoder("_successor"),
        rgnn.GoalEncoder(),
    )
    output_spec = [
        ("transition_scalar", rgnn.ObjectsScalarDecoder(hparam_config))
    ]
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model = ExtendedRGNN(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    return {"internal_model": model}


def create_sequential_delta_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
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
        EffectsEncoder(encode_shared_relations=False),
        rgnn.GoalEncoder(),
    )
    output_spec = [
        ("transition_scalar", rgnn.ObjectsScalarDecoder(hparam_config))
    ]
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model = ExtendedRGNN(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    return {"internal_model": model}


def create_sequential_delta_rooted_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    input_spec = (
        EnhancedStateEncoder(add_source_state_handle=True),
        EffectsEncoder(
            encode_shared_relations=False,
            add_target_state_handle=True,
            connect_source_and_target_handle=True,
        ),
        rgnn.GoalEncoder(),
    )
    output_spec = [
        ("transition_scalar", rgnn.ObjectsScalarDecoder(hparam_config))
    ]
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model = ExtendedRGNN(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    return {"internal_model": model}


def create_sequential_comparison_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    input_spec = (
        NullStateEncoder(),  # just to fulfill requirements of api...
        DifferenceViaRelationsStateEncoder(),
        # EffectsEncoder(encode_shared_relations=True),
        # rgnn.GoalEncoder(),
    )
    output_spec = [
        ("transition_scalar", rgnn.ObjectsScalarDecoder(hparam_config))
    ]
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model = ExtendedRGNN(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    return {"internal_model": model}


def create_sequential_aux_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    input_spec = (
        DifferenceViaObjectsStateEncoder(),
        DifferenceViaObjectsGoalEncoder(),
    )
    output_spec = [
        ("transition_scalar", rgnn.ObjectsScalarDecoder(hparam_config))
    ]
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model = ExtendedRGNN(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    return {"internal_model": model}


def create_sequential_external_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=num_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    input_spec = (rgnn.StateEncoder(), rgnn.GoalEncoder())
    output_spec = [("embeddings", rgnn.ObjectsEmbeddingDecoder())]
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model = ExtendedRGNN(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    external_mlp = rgnn.MLP(
        2 * hparam_config.embedding_size, hparam_config.embedding_size
    )
    external_readout = rgnn.SumReadout(hparam_config.embedding_size, 1)
    return {
        "internal_model": model,
        "external_mlp": external_mlp,
        "external_readout": external_readout,
    }


def create_ground_actions_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
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
        rgnn.GroundActionsEncoder(),
        rgnn.GoalEncoder(),
    )
    output_spec = [("action_scalar", rgnn.ActionScalarDecoder(hparam_config))]
    module_config = rgnn.ModuleConfig(
        aggregation_function=create_aggregation_function(aggregation_name),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model = rgnn.RelationalGraphNeuralNetwork(
        hparam_config, module_config, input_spec, output_spec
    )  # type: ignore
    return {"internal_model": model}


def create_concurrent_internal_actions_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    """
    This model encodes rooted IW tree edges with separate action nodes.
    The root handle is filtered out before action readout.
    """
    return _create_concurrent_internal_actions_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_actions",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        readout_spec=CONCURRENT_INTERNAL_ACTIONS_READOUTS[
            "concurrent_internal_actions"
        ],
    )


def create_concurrent_internal_actions_source_action_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_concurrent_internal_actions_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_actions_source_action",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        readout_spec=CONCURRENT_INTERNAL_ACTIONS_READOUTS[
            "concurrent_internal_actions_source_action"
        ],
    )


def create_concurrent_internal_actions_action_successor_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_concurrent_internal_actions_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_actions_action_successor",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        readout_spec=CONCURRENT_INTERNAL_ACTIONS_READOUTS[
            "concurrent_internal_actions_action_successor"
        ],
    )


def create_concurrent_internal_actions_source_action_successor_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_concurrent_internal_actions_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_actions_source_action_successor",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        readout_spec=CONCURRENT_INTERNAL_ACTIONS_READOUTS[
            "concurrent_internal_actions_source_action_successor"
        ],
    )


def create_concurrent_internal_actions_source_action_rooted_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_concurrent_internal_actions_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_actions_source_action_rooted",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        readout_spec=CONCURRENT_INTERNAL_ACTIONS_READOUTS[
            "concurrent_internal_actions_source_action_rooted"
        ],
    )


def create_concurrent_internal_actions_action_successor_rooted_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_concurrent_internal_actions_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_actions_action_successor_rooted",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        readout_spec=CONCURRENT_INTERNAL_ACTIONS_READOUTS[
            "concurrent_internal_actions_action_successor_rooted"
        ],
    )


def create_concurrent_internal_actions_source_action_successor_rooted_model(
    domain: mm.Domain,
    embedding_size: int,
    num_layers: int,
    aggregation_name: str,
    binary_embeddings: bool,
    iw_depth_feature: bool = False,
    iw_depth_config: IWDepthConfig | None = None,
    lgan_config: LGANConfig | None = None,
    rw_return_config: RWReturnConfig | None = None,
    iw_depth_order_aux_loss: bool = False,
) -> dict[str, torch.nn.Module]:
    return _create_concurrent_internal_actions_model(
        domain,
        embedding_size,
        num_layers,
        aggregation_name,
        binary_embeddings,
        encoding="concurrent_internal_actions_source_action_successor_rooted",
        iw_depth_feature=iw_depth_feature,
        iw_depth_config=iw_depth_config,
        lgan_config=lgan_config,
        rw_return_config=rw_return_config,
        iw_depth_order_aux_loss=iw_depth_order_aux_loss,
        readout_spec=CONCURRENT_INTERNAL_ACTIONS_READOUTS[
            "concurrent_internal_actions_source_action_successor_rooted"
        ],
    )


ENCODING_CREATE_FUNCTION = {
    "ground_actions": create_ground_actions_model,
    "sequential_internal": create_sequential_internal_model,
    "sequential_external": create_sequential_external_model,
    "sequential_delta": create_sequential_delta_model,
    "sequential_delta_rooted": create_sequential_delta_rooted_model,
    "sequential_comparison": create_sequential_comparison_model,
    "sequential_aux": create_sequential_aux_model,
    "concurrent_internal": create_concurrent_internal_model,
    "concurrent_internal_tree": create_concurrent_internal_tree_model,
    "concurrent_internal_tree_rooted": create_concurrent_internal_rooted_model,
    "concurrent_internal_comparison_tree": create_concurrent_internal_comparison_model,
    "concurrent_internal_action_tree": create_concurrent_internal_action_model,
    "concurrent_internal_action_hybrid_tree": create_concurrent_internal_action_hybrid_model,
    "concurrent_internal_actions_flat": create_concurrent_internal_actions_model_flat,
    "concurrent_internal_actions": create_concurrent_internal_actions_model,
    "concurrent_internal_actions_source_action": create_concurrent_internal_actions_source_action_model,
    "concurrent_internal_actions_action_successor": create_concurrent_internal_actions_action_successor_model,
    "concurrent_internal_actions_source_action_successor": create_concurrent_internal_actions_source_action_successor_model,
    "concurrent_internal_actions_source_action_rooted": create_concurrent_internal_actions_source_action_rooted_model,
    "concurrent_internal_actions_action_successor_rooted": create_concurrent_internal_actions_action_successor_rooted_model,
    "concurrent_internal_actions_source_action_successor_rooted": create_concurrent_internal_actions_source_action_successor_rooted_model,
    "concurrent_internal_joint_actions": create_concurrent_internal_joint_actions_model,
}

# -----------------------------------------------
# Forward functions for different encoding types.
# -----------------------------------------------


# Cache the results of last IW searches.
iw_cache_dict: dict[
    mm.State, tuple[list[IWGroundAction], list[tuple[int, int]]]
] = {}
iw_cache_queue: deque[mm.State] = deque()
IW_CACHE_MAX_SIZE = 5000

# Track the number of generated candidates during the current collection phase.
_collection_novel_iw_state_count = 0
_collection_ground_action_count = 0

# Cache the results of the last of successor effects
successor_effects_cache_dict: dict[
    mm.State, tuple[list[mm.GroundAction], list[list[mm.GroundLiteral]]]
] = {}
successor_effects_cache_queue: deque[mm.State] = deque()
SUCCESSOR_EFFECTS_CACHE_MAX_SIZE = 5000


def clear_search_context_caches() -> None:
    iw_cache_dict.clear()
    iw_cache_queue.clear()
    successor_effects_cache_dict.clear()
    successor_effects_cache_queue.clear()
    reset_collection_action_counts()


def reset_collection_action_counts() -> None:
    global _collection_novel_iw_state_count
    global _collection_ground_action_count
    _collection_novel_iw_state_count = 0
    _collection_ground_action_count = 0


def record_novel_iw_state_count(count: int) -> None:
    global _collection_novel_iw_state_count
    _collection_novel_iw_state_count += count


def record_ground_action_count(count: int) -> None:
    global _collection_ground_action_count
    _collection_ground_action_count += count


def get_collection_action_counts() -> tuple[int, int]:
    return _collection_novel_iw_state_count, _collection_ground_action_count


def get_iw_actions(
    state: mm.State, **kwargs
) -> tuple[list[IWGroundAction], list[tuple[int, int]]]:
    global iw_cache_dict, iw_cache_queue, IW_CACHE_MAX_SIZE
    if state in iw_cache_dict:
        iw_actions, iw_edges = iw_cache_dict[state]
    else:
        iw_actions, iw_edges = generate_iw_actions(state, **kwargs)
        iw_cache_dict[state] = (iw_actions, iw_edges)
        iw_cache_queue.append(state)
        while len(iw_cache_queue) > IW_CACHE_MAX_SIZE:
            del iw_cache_dict[iw_cache_queue.popleft()]
        record_novel_iw_state_count(len(iw_actions))
    return iw_actions, iw_edges


def get_model_iw_actions(
    model: "IWModel", state: mm.State, **kwargs
) -> tuple[list[IWGroundAction], list[tuple[int, int]]]:
    kwargs.setdefault("projective_iw1", model.projective_iw)
    kwargs.setdefault("typed_projection", model.typed_projection)
    kwargs.setdefault("abstracted_iw", getattr(model, "abstracted_iw", False))
    kwargs.setdefault(
        "abstracted_iw_width", getattr(model, "abstracted_iw_width", 1)
    )
    kwargs.setdefault(
        "base_abstracted_iw", getattr(model, "base_abstracted_iw", False)
    )
    kwargs.setdefault(
        "preserve_goal_atoms", getattr(model, "preserve_goal_atoms", True)
    )
    kwargs.setdefault("keep_depth_one_novel", model.keep_depth_one_novel)
    kwargs.setdefault(
        "keep_goal_nonunary_atoms", model.keep_goal_nonunary_atoms
    )
    kwargs.setdefault(
        "layer_ordering_strategy_name", model.iw_layer_ordering_strategy
    )
    kwargs.setdefault("max_next_layer_states", model.iw_max_next_layer_states)
    kwargs.setdefault("beam_width", model.iw_beam_width)
    kwargs.setdefault("beam_novelty_mode", model.iw_beam_novelty_mode)
    kwargs.setdefault("max_depth", model.iw_max_depth)
    kwargs.setdefault(
        "iw1_precheck_add_effect_novelty",
        model.iw1_precheck_add_effect_novelty,
    )
    kwargs.setdefault("iw1_atom_first_mode", model.iw1_atom_first_mode)
    kwargs.setdefault("iw1_atom_first_ratio", model.iw1_atom_first_ratio)
    kwargs.setdefault(
        "iw1_incremental_first_applicability",
        model.iw1_incremental_first_applicability,
    )
    kwargs.setdefault(
        "iw1_incremental_first_applicability_debug_crosscheck",
        model.iw1_incremental_first_applicability_debug_crosscheck,
    )
    kwargs.setdefault(
        "relaxed_survivors_only_beam",
        model.iw_relaxed_survivors_only_beam,
    )
    kwargs.setdefault(
        "randomize_equal_score_ties", model.iw_randomize_equal_score_ties
    )
    kwargs.setdefault("equal_score_tie_seed", model.iw_equal_score_tie_seed)
    kwargs.setdefault("num_threads", model.iw_num_threads)
    kwargs.setdefault("chunk_size", model.iw_chunk_size)
    return get_iw_actions(state, **kwargs)


def get_model_iw_expansion(model: "IWModel", state: mm.State, **kwargs):
    kwargs.setdefault("projective_iw1", model.projective_iw)
    kwargs.setdefault("typed_projection", model.typed_projection)
    kwargs.setdefault("abstracted_iw", getattr(model, "abstracted_iw", False))
    kwargs.setdefault(
        "abstracted_iw_width", getattr(model, "abstracted_iw_width", 1)
    )
    kwargs.setdefault(
        "base_abstracted_iw", getattr(model, "base_abstracted_iw", False)
    )
    kwargs.setdefault(
        "preserve_goal_atoms", getattr(model, "preserve_goal_atoms", True)
    )
    kwargs.setdefault("keep_depth_one_novel", model.keep_depth_one_novel)
    kwargs.setdefault(
        "keep_goal_nonunary_atoms", model.keep_goal_nonunary_atoms
    )
    kwargs.setdefault(
        "layer_ordering_strategy_name", model.iw_layer_ordering_strategy
    )
    kwargs.setdefault("max_next_layer_states", model.iw_max_next_layer_states)
    kwargs.setdefault("beam_width", model.iw_beam_width)
    kwargs.setdefault("beam_novelty_mode", model.iw_beam_novelty_mode)
    kwargs.setdefault("max_depth", model.iw_max_depth)
    kwargs.setdefault(
        "iw1_precheck_add_effect_novelty",
        model.iw1_precheck_add_effect_novelty,
    )
    kwargs.setdefault("iw1_atom_first_mode", model.iw1_atom_first_mode)
    kwargs.setdefault("iw1_atom_first_ratio", model.iw1_atom_first_ratio)
    kwargs.setdefault(
        "iw1_incremental_first_applicability",
        model.iw1_incremental_first_applicability,
    )
    kwargs.setdefault(
        "iw1_incremental_first_applicability_debug_crosscheck",
        model.iw1_incremental_first_applicability_debug_crosscheck,
    )
    kwargs.setdefault(
        "relaxed_survivors_only_beam",
        model.iw_relaxed_survivors_only_beam,
    )
    kwargs.setdefault(
        "randomize_equal_score_ties", model.iw_randomize_equal_score_ties
    )
    kwargs.setdefault("equal_score_tie_seed", model.iw_equal_score_tie_seed)
    kwargs.setdefault("num_threads", model.iw_num_threads)
    kwargs.setdefault("chunk_size", model.iw_chunk_size)
    return generate_iw_expansion(state, **kwargs)


def get_action_effects(action: mm.GroundAction) -> list[mm.GroundLiteral]:
    effects: list[mm.GroundLiteral] = []
    for ground_conditional_effect in action.get_conditional_effect():
        ground_effect = ground_conditional_effect.get_effect()
        add_list = ground_effect.get_add_list()
        del_list = ground_effect.get_delete_list()
        effects.extend(
            mm.GroundLiteral.new(atom, True, action.get_problem())
            for atom in add_list
        )
        effects.extend(
            mm.GroundLiteral.new(atom, False, action.get_problem())
            for atom in del_list
        )
    return effects


def get_normalized_iw_action_depths(
    iw_actions: list[IWGroundAction],
) -> Sequence[float]:
    if not iw_actions:
        return []
    depths = [len(iw_action.actions) for iw_action in iw_actions]
    max_depth = max(depths)
    if max_depth <= 0:
        return [0.0 for _ in depths]
    return [depth / max_depth for depth in depths]


def get_raw_iw_action_depths(
    iw_actions: list[IWGroundAction],
) -> Sequence[int]:
    return [len(iw_action.actions) for iw_action in iw_actions]


def get_iw_action_depth_feature_values(
    iw_actions: list[IWGroundAction],
    depth_config: IWDepthConfig,
) -> Sequence[float | int]:
    if depth_config.mode == IW_DEPTH_NODES:
        return get_raw_iw_action_depths(iw_actions)
    return get_normalized_iw_action_depths(iw_actions)


def get_successor_effects_and_actions(
    current_state: mm.State,
) -> tuple[list[mm.GroundAction], list[list[mm.GroundLiteral]]]:
    problem = current_state.get_problem()
    effects: list[list[mm.GroundLiteral]] = []
    if current_state in successor_effects_cache_dict:
        return successor_effects_cache_dict[current_state]

    actions = current_state.generate_applicable_actions()
    record_ground_action_count(len(actions))
    current_atoms = set(current_state.get_atoms(ignore_static=True))
    for action in actions:
        successor_state = action.apply(current_state)
        # Compute effect
        successor_atoms = set(successor_state.get_atoms(ignore_static=True))
        add_atoms = successor_atoms - current_atoms
        delete_atoms = current_atoms - successor_atoms
        effect: list[mm.GroundLiteral] = []
        effect.extend(
            mm.GroundLiteral.new(atom, True, problem) for atom in add_atoms
        )
        effect.extend(
            mm.GroundLiteral.new(atom, False, problem) for atom in delete_atoms
        )
        effects.append(effect)
    successor_effects_cache_dict[current_state] = (actions, effects)
    successor_effects_cache_queue.append(current_state)
    while len(successor_effects_cache_queue) > SUCCESSOR_EFFECTS_CACHE_MAX_SIZE:
        del successor_effects_cache_dict[
            successor_effects_cache_queue.popleft()
        ]
    return actions, effects


def concurrent_internal_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        if flat:
            actions, effects = get_successor_effects_and_actions(state)
            transition_payload: Any = effects
            actions_list.append(actions)  # type: ignore
        else:
            expansion = get_model_iw_expansion(model, state, include_start=False)
            transition_payload = expansion
            actions_list.append(list(expansion.actions_py()))  # type: ignore
        input_list.append((state, transition_payload, goal))
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("action_scalar")  # type: ignore
    output = list(zip(scalars_list, actions_list))
    return output


def concurrent_internal_actions_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if flat:
        raise RuntimeError(
            "Flat mode must is NOT supported for concurrent_internal_actions_forward."
        )
    readout_spec = CONCURRENT_INTERNAL_ACTIONS_READOUTS.get(model.encoding)
    if readout_spec is None:
        raise RuntimeError(
            f"Unsupported concurrent_internal_actions encoding: {model.encoding!r}."
        )
    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        expansion = get_model_iw_expansion(model, state, include_start=True)
        input_list.append((state, expansion, goal))
        actions_list.append(list(expansion.actions_py()))  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    out = model.internal_model.forward(
        input_list,
        filter_out_root_action_ids=True,
    )
    # The root handle stays available as readout context through
    # root_action_indices, but it is never scored as an action.
    scalars_list: list[torch.Tensor] = out.readout(readout_spec.readout_name)  # type: ignore
    assert len(scalars_list) == len(actions_list), (
        "Action readout count must match the IW action count."
    )
    for scalars, actions in zip(scalars_list, actions_list):
        assert scalars.shape[0] == len(actions), (
            "Action readout count must match the IW action count."
        )
    output = list(zip(scalars_list, actions_list))
    return output


def concurrent_internal_tree_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if flat:
        raise RuntimeError(
            "Flat mode not supported for concurrent_internal_tree encoding."
        )
    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        expansion = get_model_iw_expansion(model, state, include_start=False)
        input_list.append((state, expansion, goal))
        actions_list.append(list(expansion.actions_py()))  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("action_scalar")  # type: ignore
    output = list(zip(scalars_list, actions_list))
    return output


def concurrent_internal_action_tree_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if flat:
        raise RuntimeError(
            "Flat mode not supported for concurrent_internal_action_tree encoding."
        )
    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        expansion = get_model_iw_expansion(model, state, include_start=False)
        input_list.append((state, expansion, goal))
        actions_list.append(list(expansion.actions_py()))  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("action_scalar")  # type: ignore
    output = list(zip(scalars_list, actions_list))
    return output


def sequential_internal_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    input_list: list[
        tuple[mm.State, mm.State, mm.GroundConjunctiveCondition]
    ] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        if flat:
            actions = state.generate_applicable_actions()
        else:
            actions, _ = get_model_iw_actions(model, state)
        input_list.extend(
            (state, action.apply(state), goal) for action in actions
        )
        actions_list.append(actions)  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: torch.Tensor = model.internal_model.forward(
        input_list
    ).readout("transition_scalar")  # type: ignore
    output = []
    offset = 0
    expected_scalars = sum(len(iwaction) for iwaction in actions_list)
    assert len(scalars_list) == expected_scalars, (
        f"{len(scalars_list)} != {expected_scalars}"
    )
    for actions in actions_list:
        num_actions = len(actions)
        scalars = scalars_list[offset : offset + num_actions]
        offset += num_actions
        output.append((scalars, actions))
    return output


def sequential_delta_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        if flat:
            actions, effects = get_successor_effects_and_actions(state)
        else:
            actions, _ = get_model_iw_actions(model, state)
            effects = [action.get_effect() for action in actions]
        input_list.extend((state, (effect, None), goal) for effect in effects)
        actions_list.append(actions)  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("transition_scalar")  # type: ignore
    output = []
    offset = 0
    for actions in actions_list:
        num_actions = len(actions)
        scalars = scalars_list[offset : offset + num_actions]
        offset += num_actions
        output.append((scalars, actions))
    return output


def sequential_comparison_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        if flat:
            actions, effects = get_successor_effects_and_actions(state)
            end_states = [action.apply(state) for action in actions]
        else:
            actions, _ = get_model_iw_actions(model, state)
            effects = [action.get_effect() for action in actions]
            end_states = [action.final_state for action in actions]
        input_list.extend(
            (state, (effect, end_state.get_atoms(ignore_static=True)), goal)
            for effect, end_state in zip(effects, end_states)
        )
        actions_list.append(actions)  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("transition_scalar")  # type: ignore
    output = []
    offset = 0
    for actions in actions_list:
        num_actions = len(actions)
        scalars = scalars_list[offset : offset + num_actions]
        offset += num_actions
        output.append((scalars, actions))
    return output


def sequential_external_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    input_list: list[tuple[mm.State, mm.GroundConjunctiveCondition]] = []
    actions_list: list[list[mm.GroundAction]] = []
    num_objects_list: list[int] = []
    expected_object_counts: list[int] = []
    for state, goal in state_goals:
        problem = state.get_problem()
        domain = problem.get_domain()
        num_objects = len(problem.get_objects()) + len(domain.get_constants())
        if flat:
            actions = state.generate_applicable_actions()
        else:
            actions, _ = get_model_iw_actions(model, state)
        input_list.append((state, goal))
        input_list.extend((action.apply(state), goal) for action in actions)
        actions_list.append(actions)  # type: ignore
        num_objects_list.append(num_objects)
        expected_object_counts.extend([num_objects] * (len(actions) + 1))
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    embeddings_readout = model.internal_model.forward(input_list).readout(
        "embeddings"
    )
    if isinstance(embeddings_readout, torch.Tensor):
        scalars_list = embeddings_readout
    else:
        object_embeddings = [
            embeddings
            for embeddings in embeddings_readout
            if isinstance(embeddings, torch.Tensor)
        ]
        if len(object_embeddings) != len(embeddings_readout):
            raise TypeError(
                "Expected the embeddings readout to contain tensors only."
            )
        if len(object_embeddings) != len(expected_object_counts):
            raise ValueError(
                "Expected one embedding tensor per input graph in sequential_external."
            )
        for embeddings, expected_object_count in zip(
            object_embeddings, expected_object_counts, strict=True
        ):
            if embeddings.shape[0] != expected_object_count:
                raise ValueError(
                    "The embeddings readout does not match the input graph ordering in sequential_external."
                )
        scalars_list = torch.cat(object_embeddings, dim=0)

    # TODO: This is potentially inefficient, optimize later. We should avoid calling the MLP and readout multiple times.
    output = []
    offset = 0
    for actions, num_objects in zip(actions_list, num_objects_list):
        num_actions = len(actions)
        if num_actions > 0:
            start_object_embeddings = scalars_list[
                offset : offset + num_objects
            ]
            final_object_embeddings = scalars_list[
                offset + num_objects : offset
                + num_objects
                + num_objects * num_actions
            ]
            # Repeat the start objects to match the number of actions.
            start_object_embeddings_repeated = (
                start_object_embeddings.view(1, -1)
                .repeat_interleave(num_actions, dim=0)
                .view(final_object_embeddings.shape)
            )
            pair_object_embeddings = torch.cat(
                (start_object_embeddings_repeated, final_object_embeddings),
                dim=1,
            )
            assert hasattr(model, "external_mlp") and hasattr(
                model, "external_readout"
            )
            assert isinstance(model.external_mlp, rgnn.MLP) and isinstance(
                model.external_readout, rgnn.SumReadout
            )
            delta_object_embeddings = model.external_mlp.forward(
                pair_object_embeddings
            )
            node_sizes = torch.full(
                (num_actions,),
                num_objects,
                dtype=torch.long,
                device=delta_object_embeddings.device,
            )
            scalars = model.external_readout.forward(
                delta_object_embeddings, node_sizes
            ).view(-1)
            output.append((scalars, actions))
        else:
            output.append((torch.empty(0, device=scalars_list.device), actions))
        offset += num_objects * num_actions + num_objects
    return output


non_decision_actions = ("ppush", "ppush-done")


def next_real_state_actions(state: mm.State):
    if "pushworld" not in state.get_problem().get_domain().get_name().lower():
        return state, state.generate_applicable_actions()
    return _next_real_state_actions_pushworld(state)


def _next_real_state_actions_pushworld(state: mm.State):
    actions = state.generate_applicable_actions()
    if len(actions) >= 1 and (
        actions[0].get_action().get_name() in non_decision_actions
    ):
        return _next_real_state_actions_pushworld(actions[0].apply(state))
    return state, actions


def ground_actions_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if not flat:
        raise RuntimeError(
            "Non-flat mode not supported for ground_actions encoding."
        )
    input_list: list[
        tuple[mm.State, list[mm.GroundAction], mm.GroundConjunctiveCondition]
    ] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        real_state, actions = next_real_state_actions(state)
        input_list.append((real_state, actions, goal))
        actions_list.append(actions)  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("action_scalar")  # type: ignore
    output = list(zip(scalars_list, actions_list))
    return output


def concurrent_internal_tree_rooted_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if flat:
        raise RuntimeError(
            "Flat mode not supported for concurrent_internal_tree_rooted encoding."
        )
    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    iw_actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        expansion = get_model_iw_expansion(model, state, include_start=True)
        input_list.append((state, expansion, goal))
        iw_actions_list.append(list(expansion.actions_py()))  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    out = model.internal_model.forward(
        input_list, filter_out_root_action_ids=True
    )
    scalars_list: list[torch.Tensor] = out.readout("action_scalar")  # type: ignore
    output = list(zip(scalars_list, iw_actions_list))
    return output


def concurrent_internal_extended_tree_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if flat:
        raise RuntimeError(
            "Flat mode not supported for concurrent_internal_extended_tree encoding."
        )
    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    iw_actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        iw_actions, iw_edges = get_model_iw_actions(model, state)
        iw_effects = (
            [iw_action.get_effect() for iw_action in iw_actions],
            iw_edges,
        )
        input_list.append((state, iw_effects, goal))
        iw_actions_list.append(iw_actions)  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("action_scalar")  # type: ignore
    output = list(zip(scalars_list, iw_actions_list))
    return output


def sequential_aux_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if flat:
        raise RuntimeError(
            "Flat mode not supported for sequential_aux encoding."
        )
    input_list: list[tuple] = []
    iw_actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        iw_actions, iw_edges = get_model_iw_actions(model, state)
        input_list.extend(
            (
                (
                    state,
                    iw_action.atom_changes(),
                ),
                (
                    iw_action.final_state,
                    goal,
                ),
            )
            for iw_action in iw_actions
        )
        iw_actions_list.append(iw_actions)
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("transition_scalar")
    output = []
    offset = 0
    for iw_actions in iw_actions_list:
        num_actions = len(iw_actions)
        scalars = scalars_list[offset : offset + num_actions]
        offset += num_actions
        output.append((scalars, iw_actions))
    return output


def concurrent_internal_actions_flat_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if not flat:
        raise RuntimeError(
            "Flat mode must be enabled for concurrent_internal_actions_flat_forward."
        )
    input_list: list[
        tuple[
            mm.State, Any, mm.GroundConjunctiveCondition, list[mm.GroundAction]
        ]
    ] = []
    actions_list: list[list[mm.GroundAction]] = []
    for state, goal in state_goals:
        actions, _ = get_successor_effects_and_actions(state)
        iw_actions, _ = get_iw_actions(state)
        iw_effects = [action.get_effect() for action in iw_actions]
        input_list.append((state, iw_effects, goal, actions))
        actions_list.append(actions)  # type: ignore
    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("action_scalar")  # type: ignore
    output = list(zip(scalars_list, actions_list))
    return output


def concurrent_internal_joint_actions_forward(
    model: "IWModel",
    state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    flat: bool = False,
) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
    if flat:
        raise RuntimeError(
            "Flat mode is not supported for concurrent_internal_joint_actions_forward."
        )

    input_list: list[tuple[mm.State, Any, mm.GroundConjunctiveCondition]] = []
    actions_list: list[list[mm.GroundAction]] = []

    for state, goal in state_goals:
        iw_actions, iw_edges = get_model_iw_actions(model, state)
        transition_depths = get_iw_action_depth_feature_values(
            iw_actions, model.iw_depth_config
        )
        iw_effects = [iw_action.get_effect() for iw_action in iw_actions]

        # For each IW successor state s', enumerate primitive continuations A[s'].
        successor_actions = [
            iw_action.final_state.generate_applicable_actions()
            for iw_action in iw_actions
        ]

        input_list.append(
            (
                state,
                (iw_effects, iw_edges, successor_actions, transition_depths),
                goal,
            )
        )

        # Keep the scored candidate list compatible with the rest of the RL stack:
        # each scored item must itself be an applicable GroundAction-like object.
        actions_list.append(
            [
                JointIWGroundAction(iw_action, action)
                for iw_action, actions in zip(iw_actions, successor_actions)
                for action in actions
            ]
        )

    assert hasattr(model, "internal_model")
    assert isinstance(model.internal_model, rgnn.RelationalGraphNeuralNetwork)
    scalars_list: list[torch.Tensor] = model.internal_model.forward(
        input_list
    ).readout("action_scalar")  # type: ignore

    output = list(zip(scalars_list, actions_list))
    return output


ENCODING_FORWARD_FUNCTION = {
    "concurrent_internal_action_tree": concurrent_internal_action_tree_forward,
    "concurrent_internal_action_hybrid_tree": concurrent_internal_action_tree_forward,
    "concurrent_internal_comparison_tree": concurrent_internal_action_tree_forward,
    "concurrent_internal": concurrent_internal_forward,
    "concurrent_internal_tree": concurrent_internal_tree_forward,
    "concurrent_internal_tree_rooted": concurrent_internal_tree_rooted_forward,
    "sequential_internal": sequential_internal_forward,
    "sequential_delta": sequential_delta_forward,
    "sequential_delta_rooted": sequential_delta_forward,
    "sequential_comparison": sequential_comparison_forward,
    "sequential_aux": sequential_aux_forward,
    "sequential_external": sequential_external_forward,
    "ground_actions": ground_actions_forward,
    "concurrent_internal_actions": concurrent_internal_actions_forward,
    "concurrent_internal_actions_source_action": concurrent_internal_actions_forward,
    "concurrent_internal_actions_action_successor": concurrent_internal_actions_forward,
    "concurrent_internal_actions_source_action_successor": concurrent_internal_actions_forward,
    "concurrent_internal_actions_source_action_rooted": concurrent_internal_actions_forward,
    "concurrent_internal_actions_action_successor_rooted": concurrent_internal_actions_forward,
    "concurrent_internal_actions_source_action_successor_rooted": concurrent_internal_actions_forward,
    "concurrent_internal_actions_flat": concurrent_internal_actions_flat_forward,
    "concurrent_internal_joint_actions": concurrent_internal_joint_actions_forward,
}


# ---------------
# IW Model class.
# ---------------


class IWModel(rl.ActionScalarModel, torch.nn.Module):
    def __init__(
        self,
        encoding: str,
        modules: dict[str, torch.nn.Module],
        forward_function,
        flat: bool = False,
        verbose: bool = False,
        projective_iw: bool = False,
        typed_projection: bool = False,
        abstracted_iw: bool = False,
        abstracted_iw_width: int = 1,
        base_abstracted_iw: bool = False,
        preserve_goal_atoms: bool = True,
        keep_depth_one_novel: bool = True,
        keep_goal_nonunary_atoms: bool = False,
        iw_layer_ordering_strategy: str | None = None,
        iw_max_next_layer_states: int = -1,
        iw_beam_width: int = -1,
        iw_beam_novelty_mode: str = "all_tested",
        iw_max_depth: int = -1,
        iw1_precheck_add_effect_novelty: bool = False,
        iw1_atom_first_mode: bool = False,
        iw1_atom_first_ratio: float = 1.0,
        iw1_incremental_first_applicability: bool = False,
        iw1_incremental_first_applicability_debug_crosscheck: bool = False,
        iw_relaxed_survivors_only_beam: bool = False,
        iw_randomize_equal_score_ties: bool = False,
        iw_equal_score_tie_seed: int | None = None,
        iw_num_threads: int = -1,
        iw_chunk_size: int = -1,
        iw_depth_feature: bool = False,
        iw_depth_mode: str = "none",
        iw_depth_precedes_mode: str = "mlp",
        iw_depth_precedes_scan: bool = False,
        lgan_config: LGANConfig | None = None,
        rw_return_config: RWReturnConfig | None = None,
        iw_depth_order_aux_loss: bool = False,
        iw_depth_order_aux_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoding = encoding
        self.flat = flat
        self.forward_function = forward_function
        self.verbose = verbose
        self.projective_iw = projective_iw
        self.typed_projection = typed_projection
        self.abstracted_iw = abstracted_iw or projective_iw
        self.abstracted_iw_width = abstracted_iw_width
        self.base_abstracted_iw = (
            (not typed_projection)
            if projective_iw and not abstracted_iw
            else base_abstracted_iw
        )
        self.preserve_goal_atoms = preserve_goal_atoms
        self.keep_depth_one_novel = keep_depth_one_novel
        self.keep_goal_nonunary_atoms = keep_goal_nonunary_atoms
        self.iw_layer_ordering_strategy = iw_layer_ordering_strategy
        self.iw_max_next_layer_states = iw_max_next_layer_states
        self.iw_beam_width = iw_beam_width
        self.iw_beam_novelty_mode = iw_beam_novelty_mode
        self.iw_max_depth = iw_max_depth
        self.iw1_precheck_add_effect_novelty = (
            iw1_precheck_add_effect_novelty
        )
        self.iw1_atom_first_mode = iw1_atom_first_mode
        self.iw1_atom_first_ratio = iw1_atom_first_ratio
        self.iw1_incremental_first_applicability = (
            iw1_incremental_first_applicability
        )
        self.iw1_incremental_first_applicability_debug_crosscheck = (
            iw1_incremental_first_applicability_debug_crosscheck
        )
        self.iw_relaxed_survivors_only_beam = iw_relaxed_survivors_only_beam
        self.iw_randomize_equal_score_ties = iw_randomize_equal_score_ties
        self.iw_equal_score_tie_seed = iw_equal_score_tie_seed
        self.iw_num_threads = iw_num_threads
        self.iw_chunk_size = iw_chunk_size
        self.iw_depth_config = resolve_iw_depth_config(
            iw_depth_feature=iw_depth_feature,
            iw_depth_mode=iw_depth_mode,
            iw_depth_precedes_mode=iw_depth_precedes_mode,
            iw_depth_precedes_scan=iw_depth_precedes_scan,
        )
        self.iw_depth_feature = self.iw_depth_config.uses_init_scalar
        self.iw_depth_mode = self.iw_depth_config.mode
        self.iw_depth_precedes_mode = self.iw_depth_config.precedes_mode
        self.iw_depth_precedes_scan = self.iw_depth_config.precedes_scan
        self.lgan_config = _validate_lgan_config(encoding, lgan_config)
        self.include_lgan_edges = self.lgan_config.include_lgan_edges
        self.lgan_anchor_sources = self.lgan_config.lgan_anchor_sources
        self.lgan_tn_edge_pos = self.lgan_config.lgan_tn_edge_pos
        self.lgan_nn_edge_pos = self.lgan_config.lgan_nn_edge_pos
        self.lgan_rr_edge_pos = self.lgan_config.lgan_rr_edge_pos
        self.rw_return_config = _validate_rw_return_config(
            encoding, rw_return_config
        )
        self.rw_return_feature_mode = self.rw_return_config.mode
        self.rw_walk_mode = self.rw_return_config.walk_mode
        self.rw_return_k = self.rw_return_config.k_max
        self.rw_return_include_depth_nodes = (
            self.rw_return_config.include_depth_nodes
        )
        self.rw_return_lazy = self.rw_return_config.lazy_walk
        self.rw_return_lazy_beta = self.rw_return_config.lazy_beta
        self.rw_return_block_size = self.rw_return_config.block_size
        self.rw_return_force_block_threshold = (
            self.rw_return_config.force_block_threshold
        )
        _validate_iw_depth_order_aux(
            self.iw_depth_config,
            iw_depth_order_aux_loss,
            iw_depth_order_aux_weight,
        )
        self.iw_depth_order_aux_loss = iw_depth_order_aux_loss
        self.iw_depth_order_aux_weight = iw_depth_order_aux_weight
        for name, module in modules.items():
            self.add_module(name, module)

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    @staticmethod
    def new(
        domain: mm.Domain,
        embedding_size: int,
        num_layers: int,
        aggregation_name: str,
        binary_embeddings: bool,
        encoding: str,
        flat: bool = False,
        projective_iw: bool = False,
        typed_projection: bool = False,
        abstracted_iw: bool = False,
        abstracted_iw_width: int = 1,
        base_abstracted_iw: bool = False,
        preserve_goal_atoms: bool = True,
        keep_depth_one_novel: bool | None = None,
        keep_goal_nonunary_atoms: bool | None = None,
        iw_layer_ordering_strategy: str | None = None,
        iw_max_next_layer_states: int | None = None,
        iw_beam_width: int | None = None,
        iw_beam_novelty_mode: str | None = None,
        iw_max_depth: int | None = None,
        iw1_precheck_add_effect_novelty: bool | None = None,
        iw1_atom_first_mode: bool | None = None,
        iw1_atom_first_ratio: float | None = None,
        iw1_incremental_first_applicability: bool | None = None,
        iw1_incremental_first_applicability_debug_crosscheck: bool | None = None,
        iw_relaxed_survivors_only_beam: bool | None = None,
        iw_randomize_equal_score_ties: bool | None = None,
        iw_equal_score_tie_seed: int | None = None,
        iw_num_threads: int | None = None,
        iw_chunk_size: int | None = None,
        iw_depth_feature: bool = False,
        iw_depth_mode: str | None = None,
        iw_depth_precedes_mode: str | None = None,
        iw_depth_precedes_scan: bool | None = None,
        lgan_config: LGANConfig | None = None,
        rw_return_config: RWReturnConfig | None = None,
        iw_depth_order_aux_loss: bool = False,
        iw_depth_order_aux_weight: float = 0.0,
    ) -> "IWModel":
        assert encoding in ENCODING_CREATE_FUNCTION, (
            f"Unknown encoding type: {encoding}."
        )
        assert encoding in ENCODING_FORWARD_FUNCTION, (
            f"Unknown encoding type: {encoding}."
        )
        resolved_iw_depth_config = _validate_iw_depth_config(
            encoding,
            resolve_iw_depth_config(
                iw_depth_feature=iw_depth_feature,
                iw_depth_mode=iw_depth_mode,
                iw_depth_precedes_mode=iw_depth_precedes_mode,
                iw_depth_precedes_scan=iw_depth_precedes_scan,
            ),
            aggregation_name,
        )
        resolved_rw_return_config = _validate_rw_return_config(
            encoding,
            RWReturnConfig()
            if rw_return_config is None
            else rw_return_config,
        )
        resolved_lgan_config = _validate_lgan_config(encoding, lgan_config)
        _validate_iw_depth_order_aux(
            resolved_iw_depth_config,
            iw_depth_order_aux_loss,
            iw_depth_order_aux_weight,
        )
        create_function = ENCODING_CREATE_FUNCTION[encoding]
        forward_function = partial(
            ENCODING_FORWARD_FUNCTION[encoding], flat=flat
        )
        create_kwargs: dict[str, Any] = {
            "iw_depth_feature": resolved_iw_depth_config.uses_init_scalar,
            "iw_depth_config": resolved_iw_depth_config,
            "iw_depth_order_aux_loss": iw_depth_order_aux_loss,
        }
        if encoding in LGAN_SUPPORTED_ENCODINGS:
            create_kwargs["lgan_config"] = resolved_lgan_config
        if encoding in RW_RETURN_SUPPORTED_ENCODINGS:
            create_kwargs["rw_return_config"] = resolved_rw_return_config
        modules = create_function(
            domain,
            embedding_size,
            num_layers,
            aggregation_name,
            binary_embeddings,
            **create_kwargs,
        )
        return IWModel(
            encoding,
            modules,
            forward_function,
            flat=flat,
            projective_iw=projective_iw,
            typed_projection=typed_projection,
            abstracted_iw=abstracted_iw,
            abstracted_iw_width=abstracted_iw_width,
            base_abstracted_iw=base_abstracted_iw,
            preserve_goal_atoms=preserve_goal_atoms,
            keep_depth_one_novel=(
                True if keep_depth_one_novel is None else keep_depth_one_novel
            ),
            keep_goal_nonunary_atoms=(
                False
                if keep_goal_nonunary_atoms is None
                else keep_goal_nonunary_atoms
            ),
            iw_layer_ordering_strategy=iw_layer_ordering_strategy,
            iw_max_next_layer_states=(
                -1
                if iw_max_next_layer_states is None
                else iw_max_next_layer_states
            ),
            iw_beam_width=(-1 if iw_beam_width is None else iw_beam_width),
            iw_beam_novelty_mode=(
                "all_tested"
                if iw_beam_novelty_mode is None
                else iw_beam_novelty_mode
            ),
            iw_max_depth=(-1 if iw_max_depth is None else iw_max_depth),
            iw1_precheck_add_effect_novelty=(
                False
                if iw1_precheck_add_effect_novelty is None
                else iw1_precheck_add_effect_novelty
            ),
            iw1_atom_first_mode=(
                False
                if iw1_atom_first_mode is None
                else iw1_atom_first_mode
            ),
            iw1_atom_first_ratio=(
                1.0 if iw1_atom_first_ratio is None else iw1_atom_first_ratio
            ),
            iw1_incremental_first_applicability=(
                False
                if iw1_incremental_first_applicability is None
                else iw1_incremental_first_applicability
            ),
            iw1_incremental_first_applicability_debug_crosscheck=(
                False
                if iw1_incremental_first_applicability_debug_crosscheck
                is None
                else iw1_incremental_first_applicability_debug_crosscheck
            ),
            iw_relaxed_survivors_only_beam=(
                False
                if iw_relaxed_survivors_only_beam is None
                else iw_relaxed_survivors_only_beam
            ),
            iw_randomize_equal_score_ties=(
                False
                if iw_randomize_equal_score_ties is None
                else iw_randomize_equal_score_ties
            ),
            iw_equal_score_tie_seed=iw_equal_score_tie_seed,
            iw_num_threads=(-1 if iw_num_threads is None else iw_num_threads),
            iw_chunk_size=(-1 if iw_chunk_size is None else iw_chunk_size),
            iw_depth_feature=resolved_iw_depth_config.uses_init_scalar,
            iw_depth_mode=resolved_iw_depth_config.mode,
            iw_depth_precedes_mode=resolved_iw_depth_config.precedes_mode,
            iw_depth_precedes_scan=resolved_iw_depth_config.precedes_scan,
            lgan_config=resolved_lgan_config,
            rw_return_config=resolved_rw_return_config,
            iw_depth_order_aux_loss=iw_depth_order_aux_loss,
            iw_depth_order_aux_weight=iw_depth_order_aux_weight,
        )

    def forward(
        self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]
    ) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        # Print time taken by IW action generation if verbose is enabled.
        start_time = time.time()
        output = self.forward_function(self, state_goals)
        end_time = time.time()
        if self.verbose:
            batch_size = len(output)
            total_actions = sum(len(actions) for _, actions in output)
            print(
                f"Forward took {end_time - start_time:.4f} seconds; batch size: {batch_size}; total IW actions: {total_actions}; average IW actions: {total_actions / batch_size:.2f}."
            )
        return output

    @staticmethod
    def _flatten_depth_scores(
        depth_score_list: list[torch.Tensor], device: torch.device
    ) -> tuple[torch.Tensor, list[int]]:
        depth_node_sizes = [int(scores.shape[0]) for scores in depth_score_list]
        if not depth_score_list:
            return torch.empty(0, dtype=torch.float, device=device), []
        return torch.cat(depth_score_list, dim=0), depth_node_sizes

    def _forward_depth_order_scores(
        self, out: rgnn.ForwardState
    ) -> tuple[torch.Tensor | None, list[int] | None]:
        if not self.iw_depth_order_aux_loss:
            return None, None
        assert hasattr(self, "internal_model")
        assert isinstance(
            self.internal_model, rgnn.RelationalGraphNeuralNetwork
        )
        depth_score_list: list[torch.Tensor] = out.readout(  # type: ignore[assignment]
            DEPTH_ORDER_SCORE_READOUT
        )
        return self._flatten_depth_scores(
            depth_score_list, self.internal_model.get_device()
        )

    def forward_with_aux(
        self,
        state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
    ) -> dict[str, Any]:
        assert hasattr(self, "internal_model")
        assert isinstance(
            self.internal_model, rgnn.RelationalGraphNeuralNetwork
        )
        if self.encoding in CONCURRENT_INTERNAL_ACTIONS_READOUTS:
            readout_spec = CONCURRENT_INTERNAL_ACTIONS_READOUTS[self.encoding]
            input_list: list[
                tuple[mm.State, Any, mm.GroundConjunctiveCondition]
            ] = []
            actions_list: list[list[mm.GroundAction]] = []
            for state, goal in state_goals:
                expansion = get_model_iw_expansion(self, state, include_start=True)
                input_list.append((state, expansion, goal))
                actions_list.append(list(expansion.actions_py()))  # type: ignore[arg-type]
            out = self.internal_model.forward(
                input_list,
                filter_out_root_action_ids=True,
            )
            scalars_list: list[torch.Tensor] = out.readout(readout_spec.readout_name)  # type: ignore[assignment]
        elif self.encoding == "concurrent_internal":
            input_list = []
            actions_list = []
            for state, goal in state_goals:
                expansion = get_model_iw_expansion(self, state, include_start=False)
                input_list.append((state, expansion, goal))
                actions_list.append(list(expansion.actions_py()))  # type: ignore[arg-type]
            out = self.internal_model.forward(input_list)
            scalars_list = out.readout("action_scalar")  # type: ignore[assignment]
        elif self.encoding in {
            "concurrent_internal_tree",
            "concurrent_internal_action_tree",
            "concurrent_internal_action_hybrid_tree",
            "concurrent_internal_comparison_tree",
        }:
            input_list = []
            actions_list = []
            for state, goal in state_goals:
                expansion = get_model_iw_expansion(self, state, include_start=False)
                input_list.append((state, expansion, goal))
                actions_list.append(list(expansion.actions_py()))  # type: ignore[arg-type]
            out = self.internal_model.forward(input_list)
            scalars_list = out.readout("action_scalar")  # type: ignore[assignment]
        elif self.encoding == "concurrent_internal_tree_rooted":
            input_list = []
            actions_list = []
            for state, goal in state_goals:
                expansion = get_model_iw_expansion(self, state, include_start=True)
                input_list.append((state, expansion, goal))
                actions_list.append(list(expansion.actions_py()))  # type: ignore[arg-type]
            out = self.internal_model.forward(
                input_list,
                filter_out_root_action_ids=True,
            )
            scalars_list = out.readout("action_scalar")  # type: ignore[assignment]
        elif self.encoding == "concurrent_internal_joint_actions":
            input_list = []
            actions_list = []
            for state, goal in state_goals:
                iw_actions, iw_edges = get_model_iw_actions(self, state)
                transition_depths = get_iw_action_depth_feature_values(
                    iw_actions, self.iw_depth_config
                )
                iw_effects = [iw_action.get_effect() for iw_action in iw_actions]
                successor_actions = [
                    iw_action.final_state.generate_applicable_actions()
                    for iw_action in iw_actions
                ]
                input_list.append(
                    (
                        state,
                        (
                            iw_effects,
                            iw_edges,
                            successor_actions,
                            transition_depths,
                        ),
                        goal,
                    )
                )
                actions_list.append(
                    [
                        JointIWGroundAction(iw_action, action)
                        for iw_action, actions in zip(
                            iw_actions, successor_actions
                        )
                        for action in actions
                    ]
                )
            out = self.internal_model.forward(input_list)
            scalars_list = out.readout("action_scalar")  # type: ignore[assignment]
        else:
            raise RuntimeError(
                "forward_with_aux is only supported for IW tree encodings."
            )
        action_outputs = list(zip(scalars_list, actions_list))
        depth_scores, depth_node_sizes = self._forward_depth_order_scores(out)
        return {
            "action_outputs": action_outputs,
            "depth_scores": depth_scores,
            "depth_node_sizes": depth_node_sizes,
        }

    def save(self, path: Path) -> None:
        extras: dict[str, Any] = {
            "encoding": self.encoding,
            "flat": self.flat,
            "abstracted_iw": self.abstracted_iw,
            "abstracted_iw_width": self.abstracted_iw_width,
            "base_abstracted_iw": self.base_abstracted_iw,
            "preserve_goal_atoms": self.preserve_goal_atoms,
            "projective_iw": self.projective_iw,
            "typed_projection": self.typed_projection,
            "keep_depth_one_novel": self.keep_depth_one_novel,
            "keep_goal_nonunary_atoms": self.keep_goal_nonunary_atoms,
            "iw_layer_ordering_strategy": self.iw_layer_ordering_strategy,
            "iw_max_next_layer_states": self.iw_max_next_layer_states,
            "iw_beam_width": self.iw_beam_width,
            "iw_beam_novelty_mode": self.iw_beam_novelty_mode,
            "iw_max_depth": self.iw_max_depth,
            "iw1_precheck_add_effect_novelty": self.iw1_precheck_add_effect_novelty,
            "iw1_atom_first_mode": self.iw1_atom_first_mode,
            "iw1_atom_first_ratio": self.iw1_atom_first_ratio,
            "iw1_incremental_first_applicability": self.iw1_incremental_first_applicability,
            "iw1_incremental_first_applicability_debug_crosscheck": self.iw1_incremental_first_applicability_debug_crosscheck,
            "iw_relaxed_survivors_only_beam": self.iw_relaxed_survivors_only_beam,
            "iw_randomize_equal_score_ties": self.iw_randomize_equal_score_ties,
            "iw_equal_score_tie_seed": self.iw_equal_score_tie_seed,
            "iw_num_threads": self.iw_num_threads,
            "iw_chunk_size": self.iw_chunk_size,
            "iw_depth_order_aux_loss": self.iw_depth_order_aux_loss,
            "iw_depth_order_aux_weight": self.iw_depth_order_aux_weight,
        }
        extras.update(self.lgan_config.to_extras())
        extras.update(self.iw_depth_config.to_extras())
        extras.update(self.rw_return_config.to_extras())
        if hasattr(self, "external_mlp"):
            assert isinstance(self.external_mlp, rgnn.MLP)
            extras["external_mlp"] = self.external_mlp.state_dict()
        if hasattr(self, "external_readout"):
            assert isinstance(self.external_readout, rgnn.SumReadout)
            extras["external_readout"] = self.external_readout.state_dict()
        assert hasattr(self, "internal_model")
        assert isinstance(
            self.internal_model, rgnn.RelationalGraphNeuralNetwork
        )
        self.internal_model.save(path, extras=extras)

    @staticmethod
    def load(
        domain: mm.Domain,
        path: Path,
        device: torch.device,
        flat: bool = False,
        projective_iw: bool | None = None,
        typed_projection: bool | None = None,
        abstracted_iw: bool | None = None,
        abstracted_iw_width: int | None = None,
        base_abstracted_iw: bool | None = None,
        preserve_goal_atoms: bool | None = None,
        keep_depth_one_novel: bool | None = None,
        keep_goal_nonunary_atoms: bool | None = None,
        iw_layer_ordering_strategy: str | None = None,
        iw_max_next_layer_states: int | None = None,
        iw_beam_width: int | None = None,
        iw_beam_novelty_mode: str | None = None,
        iw_max_depth: int | None = None,
        iw1_precheck_add_effect_novelty: bool | None = None,
        iw1_atom_first_mode: bool | None = None,
        iw1_atom_first_ratio: float | None = None,
        iw1_incremental_first_applicability: bool | None = None,
        iw1_incremental_first_applicability_debug_crosscheck: bool | None = None,
        iw_relaxed_survivors_only_beam: bool | None = None,
        iw_randomize_equal_score_ties: bool | None = None,
        iw_equal_score_tie_seed: int | None = None,
        iw_num_threads: int | None = None,
        iw_chunk_size: int | None = None,
        iw_depth_order_aux_loss: bool | None = None,
        iw_depth_order_aux_weight: float | None = None,
    ) -> "IWModel":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        extras = checkpoint["extras"]
        encoding = extras["encoding"]
        module_config = rgnn.ModuleConfig(**checkpoint["module_config"])
        resolved_iw_depth_config = _validate_iw_depth_config(
            encoding,
            IWDepthConfig.from_extras(extras),
            _aggregation_name_from_function(
                module_config.aggregation_function
            ),
        )
        resolved_lgan_config = _validate_lgan_config(
            encoding,
            LGANConfig.from_extras(extras),
        )
        resolved_rw_return_config = _validate_rw_return_config(
            encoding,
            RWReturnConfig.from_extras(extras),
        )
        resolved_iw_depth_order_aux_loss = (
            bool(extras.get("iw_depth_order_aux_loss", False))
            if iw_depth_order_aux_loss is None
            else iw_depth_order_aux_loss
        )
        resolved_iw_depth_order_aux_weight = (
            float(extras.get("iw_depth_order_aux_weight", 0.0))
            if iw_depth_order_aux_weight is None
            else iw_depth_order_aux_weight
        )
        _validate_iw_depth_order_aux(
            resolved_iw_depth_config,
            resolved_iw_depth_order_aux_loss,
            resolved_iw_depth_order_aux_weight,
        )
        resolved_flat = (
            flat
            or bool(extras.get("flat", False))
            or encoding.endswith("_flat")
        )
        hparam_config = rgnn.HyperparameterConfig(
            domain=domain, **checkpoint["hparam_config"]
        )
        input_spec = _hydrate_legacy_transition_tree_input_spec(
            checkpoint["input_spec"],
            resolved_iw_depth_config,
            resolved_rw_return_config,
            resolved_lgan_config,
        )
        output_spec = checkpoint["output_spec"]
        internal_model_class = _tree_model_class(
            encoding,
            resolved_iw_depth_config,
            resolved_rw_return_config,
            resolved_lgan_config,
        )
        if (
            encoding in EXTENDED_ENCODINGS
            and resolved_iw_depth_config.mode == "none"
            and not resolved_rw_return_config.enabled
            and not resolved_lgan_config.enabled
        ):
            internal_model_class = ExtendedRGNN
        if internal_model_class is DepthAwareExtendedRGNN:
            internal_model = internal_model_class(
                hparam_config,
                module_config,
                input_spec,
                output_spec,
                resolved_iw_depth_config,
                resolved_rw_return_config,
                resolved_lgan_config,
            )
        else:
            internal_model = internal_model_class(
                hparam_config,
                module_config,
                input_spec,
                output_spec,
            )
        internal_model.load_state_dict(checkpoint["model"])
        internal_model = internal_model.to(device)
        forward_function = partial(
            ENCODING_FORWARD_FUNCTION[encoding], flat=resolved_flat
        )
        modules: dict[str, torch.nn.Module] = {"internal_model": internal_model}
        if "external_mlp" in extras:
            external_mlp = rgnn.MLP(
                2 * internal_model.get_hparam_config().embedding_size,
                internal_model.get_hparam_config().embedding_size,
            )
            external_mlp.load_state_dict(extras["external_mlp"])
            modules["external_mlp"] = external_mlp
        if "external_readout" in extras:
            external_readout = rgnn.SumReadout(
                internal_model.get_hparam_config().embedding_size, 1
            )
            external_readout.load_state_dict(extras["external_readout"])
            modules["external_readout"] = external_readout
        resolved_projective_iw = (
            extras.get("projective_iw", False)
            if projective_iw is None
            else projective_iw
        )
        resolved_typed_projection = (
            extras.get("typed_projection", False)
            if typed_projection is None
            else typed_projection
        )
        resolved_abstracted_iw = (
            extras.get("abstracted_iw", resolved_projective_iw)
            if abstracted_iw is None
            else abstracted_iw
        )
        resolved_abstracted_iw_width = (
            extras.get("abstracted_iw_width", 1)
            if abstracted_iw_width is None
            else abstracted_iw_width
        )
        resolved_base_abstracted_iw = (
            extras.get(
                "base_abstracted_iw",
                resolved_projective_iw and not resolved_typed_projection,
            )
            if base_abstracted_iw is None
            else base_abstracted_iw
        )
        resolved_preserve_goal_atoms = (
            extras.get("preserve_goal_atoms", True)
            if preserve_goal_atoms is None
            else preserve_goal_atoms
        )
        resolved_keep_depth_one_novel = (
            extras.get("keep_depth_one_novel", True)
            if keep_depth_one_novel is None
            else keep_depth_one_novel
        )
        resolved_keep_goal_nonunary_atoms = (
            extras.get("keep_goal_nonunary_atoms", False)
            if keep_goal_nonunary_atoms is None
            else keep_goal_nonunary_atoms
        )
        resolved_iw_layer_ordering_strategy = (
            extras.get("iw_layer_ordering_strategy")
            if iw_layer_ordering_strategy is None
            else iw_layer_ordering_strategy
        )
        resolved_iw_max_next_layer_states = (
            extras.get("iw_max_next_layer_states", -1)
            if iw_max_next_layer_states is None
            else iw_max_next_layer_states
        )
        resolved_iw_beam_width = (
            extras.get("iw_beam_width", -1)
            if iw_beam_width is None
            else iw_beam_width
        )
        resolved_iw_beam_novelty_mode = (
            extras.get("iw_beam_novelty_mode", "all_tested")
            if iw_beam_novelty_mode is None
            else iw_beam_novelty_mode
        )
        resolved_iw_max_depth = (
            extras.get("iw_max_depth", -1)
            if iw_max_depth is None
            else iw_max_depth
        )
        resolved_iw1_precheck_add_effect_novelty = (
            extras.get("iw1_precheck_add_effect_novelty", False)
            if iw1_precheck_add_effect_novelty is None
            else iw1_precheck_add_effect_novelty
        )
        resolved_iw1_atom_first_mode = (
            extras.get("iw1_atom_first_mode", False)
            if iw1_atom_first_mode is None
            else iw1_atom_first_mode
        )
        resolved_iw1_atom_first_ratio = (
            extras.get("iw1_atom_first_ratio", 1.0)
            if iw1_atom_first_ratio is None
            else iw1_atom_first_ratio
        )
        resolved_iw1_incremental_first_applicability = (
            extras.get("iw1_incremental_first_applicability", False)
            if iw1_incremental_first_applicability is None
            else iw1_incremental_first_applicability
        )
        resolved_iw1_incremental_first_applicability_debug_crosscheck = (
            extras.get(
                "iw1_incremental_first_applicability_debug_crosscheck",
                False,
            )
            if iw1_incremental_first_applicability_debug_crosscheck is None
            else iw1_incremental_first_applicability_debug_crosscheck
        )
        resolved_iw_relaxed_survivors_only_beam = (
            extras.get("iw_relaxed_survivors_only_beam", False)
            if iw_relaxed_survivors_only_beam is None
            else iw_relaxed_survivors_only_beam
        )
        resolved_iw_randomize_equal_score_ties = (
            extras.get("iw_randomize_equal_score_ties", False)
            if iw_randomize_equal_score_ties is None
            else iw_randomize_equal_score_ties
        )
        resolved_iw_equal_score_tie_seed = (
            extras.get("iw_equal_score_tie_seed")
            if iw_equal_score_tie_seed is None
            else iw_equal_score_tie_seed
        )
        resolved_iw_num_threads = (
            extras.get("iw_num_threads", -1)
            if iw_num_threads is None
            else iw_num_threads
        )
        resolved_iw_chunk_size = (
            extras.get("iw_chunk_size", -1)
            if iw_chunk_size is None
            else iw_chunk_size
        )
        return IWModel(
            encoding,
            modules,
            forward_function,
            flat=resolved_flat,
            projective_iw=resolved_projective_iw,
            typed_projection=resolved_typed_projection,
            abstracted_iw=resolved_abstracted_iw,
            abstracted_iw_width=resolved_abstracted_iw_width,
            base_abstracted_iw=resolved_base_abstracted_iw,
            preserve_goal_atoms=resolved_preserve_goal_atoms,
            keep_depth_one_novel=resolved_keep_depth_one_novel,
            keep_goal_nonunary_atoms=resolved_keep_goal_nonunary_atoms,
            iw_layer_ordering_strategy=resolved_iw_layer_ordering_strategy,
            iw_max_next_layer_states=resolved_iw_max_next_layer_states,
            iw_beam_width=resolved_iw_beam_width,
            iw_beam_novelty_mode=resolved_iw_beam_novelty_mode,
            iw_max_depth=resolved_iw_max_depth,
            iw1_precheck_add_effect_novelty=resolved_iw1_precheck_add_effect_novelty,
            iw1_atom_first_mode=resolved_iw1_atom_first_mode,
            iw1_atom_first_ratio=resolved_iw1_atom_first_ratio,
            iw1_incremental_first_applicability=resolved_iw1_incremental_first_applicability,
            iw1_incremental_first_applicability_debug_crosscheck=resolved_iw1_incremental_first_applicability_debug_crosscheck,
            iw_relaxed_survivors_only_beam=resolved_iw_relaxed_survivors_only_beam,
            iw_randomize_equal_score_ties=resolved_iw_randomize_equal_score_ties,
            iw_equal_score_tie_seed=resolved_iw_equal_score_tie_seed,
            iw_num_threads=resolved_iw_num_threads,
            iw_chunk_size=resolved_iw_chunk_size,
            iw_depth_feature=resolved_iw_depth_config.uses_init_scalar,
            iw_depth_mode=resolved_iw_depth_config.mode,
            iw_depth_precedes_mode=resolved_iw_depth_config.precedes_mode,
            iw_depth_precedes_scan=resolved_iw_depth_config.precedes_scan,
            lgan_config=resolved_lgan_config,
            rw_return_config=resolved_rw_return_config,
            iw_depth_order_aux_loss=resolved_iw_depth_order_aux_loss,
            iw_depth_order_aux_weight=resolved_iw_depth_order_aux_weight,
        ).to(device)
