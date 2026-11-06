from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import torch

try:
    from . import _iw_encoding_backend as _native_backend_module
except ImportError:
    _native_backend_module: Any = None

native_backend: Any = _native_backend_module


RW_RETURN_NONE = "none"
RW_RETURN_EXACT_FULL = "exact_full"
RW_RETURN_EXACT_BLOCKED = "exact_blocked"
RW_RETURN_MODES = (
    RW_RETURN_NONE,
    RW_RETURN_EXACT_FULL,
    RW_RETURN_EXACT_BLOCKED,
)

RW_WALK_STANDARD = "standard"
RW_WALK_NON_RETURNING = "non_returning"
RW_WALK_MODES = (
    RW_WALK_STANDARD,
    RW_WALK_NON_RETURNING,
)

RW_RETURN_DEFAULT_FORCE_BLOCK_THRESHOLD = 4096
_SPARSE_CSR_BETA_WARNING = r"Sparse CSR tensor support is in beta state\."
warnings.filterwarnings(
    "ignore",
    message=_SPARSE_CSR_BETA_WARNING,
    category=UserWarning,
)


@dataclass(frozen=True)
class RWReturnConfig:
    mode: str = RW_RETURN_NONE
    walk_mode: str = RW_WALK_STANDARD
    k_max: int = 0
    include_depth_nodes: bool = False
    lazy_walk: bool = False
    lazy_beta: float = 0.0
    exact_blocked: bool = True
    block_size: int = 512
    force_block_threshold: int = RW_RETURN_DEFAULT_FORCE_BLOCK_THRESHOLD

    @property
    def enabled(self) -> bool:
        return self.mode != RW_RETURN_NONE

    def canonicalized(self) -> "RWReturnConfig":
        if self.mode not in RW_RETURN_MODES:
            raise RuntimeError(
                f"Unknown rw_return_feature_mode={self.mode!r}. "
                f"Expected one of {RW_RETURN_MODES!r}."
            )
        if self.walk_mode not in RW_WALK_MODES:
            raise RuntimeError(
                f"Unknown rw_walk_mode={self.walk_mode!r}. "
                f"Expected one of {RW_WALK_MODES!r}."
            )
        if self.mode == RW_RETURN_NONE:
            return RWReturnConfig()
        if self.k_max < 1:
            raise RuntimeError(
                "rw_return_k must be at least 1 when rw-return features are enabled."
            )
        if self.block_size < 1:
            raise RuntimeError("rw_return_block_size must be at least 1.")
        if self.force_block_threshold < 0:
            raise RuntimeError(
                "rw_return_force_block_threshold must be non-negative. Use 0 to disable forced fallback."
            )
        if self.lazy_walk and not 0.0 <= self.lazy_beta < 1.0:
            raise RuntimeError(
                "rw_return_lazy_beta must be in [0, 1) when lazy walk is enabled."
            )
        return self

    def validate(self, *, encoding: str | None = None) -> "RWReturnConfig":
        config = self.canonicalized()
        if config.enabled and encoding is not None and "tree" not in encoding and "actions" not in encoding:
            raise RuntimeError(
                "rw-return features are only supported for IW tree-based encodings."
            )
        return config

    def to_extras(self) -> dict[str, Any]:
        return {
            "rw_return_feature_mode": self.mode,
            "rw_walk_mode": self.walk_mode,
            "rw_return_k": self.k_max,
            "rw_return_include_depth_nodes": self.include_depth_nodes,
            "rw_return_lazy": self.lazy_walk,
            "rw_return_lazy_beta": self.lazy_beta,
            "rw_return_block_size": self.block_size,
            "rw_return_force_block_threshold": self.force_block_threshold,
        }

    @staticmethod
    def from_extras(extras: Mapping[str, Any]) -> "RWReturnConfig":
        return resolve_rw_return_config(
            rw_return_feature_mode=extras.get("rw_return_feature_mode"),
            rw_walk_mode=extras.get("rw_walk_mode"),
            rw_return_k=extras.get("rw_return_k"),
            rw_return_include_depth_nodes=extras.get(
                "rw_return_include_depth_nodes"
            ),
            rw_return_lazy=extras.get("rw_return_lazy"),
            rw_return_lazy_beta=extras.get("rw_return_lazy_beta"),
            rw_return_block_size=extras.get("rw_return_block_size"),
            rw_return_force_block_threshold=extras.get(
                "rw_return_force_block_threshold"
            ),
        )


def resolve_rw_return_config(
    *,
    rw_return_feature_mode: str | None = None,
    rw_walk_mode: str | None = None,
    rw_return_k: int | None = None,
    rw_return_include_depth_nodes: bool | None = None,
    rw_return_lazy: bool | None = None,
    rw_return_lazy_beta: float | None = None,
    rw_return_block_size: int | None = None,
    rw_return_force_block_threshold: int | None = None,
) -> RWReturnConfig:
    mode = (
        RW_RETURN_NONE
        if rw_return_feature_mode is None
        else rw_return_feature_mode
    )
    config = RWReturnConfig(
        mode=mode,
        walk_mode=(
            RW_WALK_STANDARD if rw_walk_mode is None else rw_walk_mode
        ),
        k_max=0 if rw_return_k is None else int(rw_return_k),
        include_depth_nodes=(
            False
            if rw_return_include_depth_nodes is None
            else bool(rw_return_include_depth_nodes)
        ),
        lazy_walk=False if rw_return_lazy is None else bool(rw_return_lazy),
        lazy_beta=(
            0.0 if rw_return_lazy_beta is None else float(rw_return_lazy_beta)
        ),
        exact_blocked=mode == RW_RETURN_EXACT_BLOCKED,
        block_size=512 if rw_return_block_size is None else int(rw_return_block_size),
        force_block_threshold=(
            RW_RETURN_DEFAULT_FORCE_BLOCK_THRESHOLD
            if rw_return_force_block_threshold is None
            else int(rw_return_force_block_threshold)
        ),
    )
    return config.canonicalized()


def cli_requested_rw_return_override(args: Any) -> bool:
    return any(
        getattr(args, attr, None) is not None
        for attr in (
            "rw_return_feature_mode",
            "rw_walk_mode",
            "rw_return_k",
            "rw_return_include_depth_nodes",
            "rw_return_lazy",
            "rw_return_lazy_beta",
            "rw_return_block_size",
            "rw_return_force_block_threshold",
        )
    )


def relation_arities_from_input_spec(
    input_specification: Sequence[Any],
    domain: Any,
) -> dict[str, int]:
    relation_arities: dict[str, int] = {}
    for encoder in input_specification:
        get_relations = getattr(encoder, "get_relations", None)
        if not callable(get_relations):
            continue
        for relation_name, arity in get_relations(domain):
            relation_arities[relation_name] = arity
    return relation_arities


def build_kept_node_mapping(
    *,
    num_total_nodes: int,
    object_indices: torch.Tensor,
    action_indices: torch.Tensor,
    root_action_indices: torch.Tensor,
    auxiliary_indices: torch.Tensor,
    depth_node_indices: torch.Tensor,
    separate_action_nodes: bool,
    include_depth_nodes: bool,
) -> tuple[list[int], list[int]]:
    global_to_kept = [-1] * num_total_nodes
    kept_to_global: list[int] = []

    def add_indices(indices: torch.Tensor) -> None:
        for node_id in indices.tolist():
            idx = int(node_id)
            if idx < 0 or idx >= num_total_nodes:
                raise RuntimeError(
                    f"RW-return node id {idx} is outside [0, {num_total_nodes})."
                )
            if global_to_kept[idx] != -1:
                continue
            global_to_kept[idx] = len(kept_to_global)
            kept_to_global.append(idx)

    add_indices(object_indices)
    add_indices(root_action_indices)
    add_indices(auxiliary_indices)
    if not separate_action_nodes:
        add_indices(action_indices)
    if include_depth_nodes:
        add_indices(depth_node_indices)
    return global_to_kept, kept_to_global


def build_rw_return_operator_from_relations(
    *,
    num_total_nodes: int,
    flattened_relations: Mapping[str, torch.Tensor],
    relation_arities: Mapping[str, int],
    object_indices: torch.Tensor,
    action_indices: torch.Tensor,
    root_action_indices: torch.Tensor,
    auxiliary_indices: torch.Tensor,
    depth_node_indices: torch.Tensor,
    separate_action_nodes: bool,
    config: RWReturnConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config = config.canonicalized()
    global_to_kept, kept_to_global = build_kept_node_mapping(
        num_total_nodes=num_total_nodes,
        object_indices=object_indices,
        action_indices=action_indices,
        root_action_indices=root_action_indices,
        auxiliary_indices=auxiliary_indices,
        depth_node_indices=depth_node_indices,
        separate_action_nodes=separate_action_nodes,
        include_depth_nodes=config.include_depth_nodes,
    )
    num_kept = len(kept_to_global)
    incident_counts = [0] * num_kept

    for relation_name, relation_terms in flattened_relations.items():
        arity = relation_arities.get(relation_name)
        if arity is None:
            continue
        if arity < 2:
            continue
        flat_terms = relation_terms.tolist()
        if len(flat_terms) % arity != 0:
            raise RuntimeError(
                f"Relation {relation_name!r} has {len(flat_terms)} terms for arity {arity}."
            )
        chunk_kept: list[int] = []
        for start in range(0, len(flat_terms), arity):
            chunk_kept.clear()
            for node_id in flat_terms[start : start + arity]:
                kept_idx = global_to_kept[int(node_id)]
                if kept_idx != -1:
                    chunk_kept.append(kept_idx)
            if len(chunk_kept) < 2:
                continue
            for kept_idx in chunk_kept:
                incident_counts[kept_idx] += 1

    edge_weights: dict[tuple[int, int], float] = {}
    row_sums = [0.0] * num_kept
    for relation_name, relation_terms in flattened_relations.items():
        arity = relation_arities.get(relation_name)
        if arity is None:
            continue
        if arity < 2:
            continue
        flat_terms = relation_terms.tolist()
        chunk_kept = []
        for start in range(0, len(flat_terms), arity):
            chunk_kept.clear()
            for node_id in flat_terms[start : start + arity]:
                kept_idx = global_to_kept[int(node_id)]
                if kept_idx != -1:
                    chunk_kept.append(kept_idx)
            chunk_size = len(chunk_kept)
            if chunk_size < 2:
                continue
            dst_weight = 1.0 / float(chunk_size - 1)
            for src_pos, src in enumerate(chunk_kept):
                occ_weight = 1.0 / float(incident_counts[src])
                weight = occ_weight * dst_weight
                for dst_pos, dst in enumerate(chunk_kept):
                    if src_pos == dst_pos:
                        continue
                    edge_weights[(dst, src)] = edge_weights.get((dst, src), 0.0) + weight
                    row_sums[src] += weight

    for kept_idx, count in enumerate(incident_counts):
        if count != 0:
            continue
        edge_weights[(kept_idx, kept_idx)] = edge_weights.get(
            (kept_idx, kept_idx), 0.0
        ) + 1.0
        row_sums[kept_idx] += 1.0

    if config.lazy_walk:
        scale = 1.0 - config.lazy_beta
        edge_weights = {
            key: value * scale for key, value in edge_weights.items()
        }
        row_sums = [value * scale for value in row_sums]
        for kept_idx in range(num_kept):
            edge_weights[(kept_idx, kept_idx)] = edge_weights.get(
                (kept_idx, kept_idx), 0.0
            ) + config.lazy_beta
            row_sums[kept_idx] += config.lazy_beta

    for kept_idx, row_sum in enumerate(row_sums):
        if abs(row_sum - 1.0) > 1e-5:
            raise RuntimeError(
                f"RW-return row {kept_idx} is not stochastic: sum={row_sum}."
            )

    row_ptr = [0]
    col_ind: list[int] = []
    values: list[float] = []
    offset = 0
    for row in range(num_kept):
        row_entries = sorted(
            (key[1], value)
            for key, value in edge_weights.items()
            if key[0] == row
        )
        offset += len(row_entries)
        row_ptr.append(offset)
        for col, value in row_entries:
            col_ind.append(col)
            values.append(value)

    return (
        torch.tensor(row_ptr, dtype=torch.int),
        torch.tensor(col_ind, dtype=torch.int),
        torch.tensor(values, dtype=torch.float),
        torch.tensor(kept_to_global, dtype=torch.int),
    )


def _compute_exact_blocked(
    matrix_t: torch.Tensor,
    num_kept: int,
    config: RWReturnConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ret = torch.zeros((num_kept, config.k_max), dtype=dtype, device=device)
    block_size = min(config.block_size, num_kept)
    diagonal_rows = torch.arange(num_kept, device=device)
    for start in range(0, num_kept, block_size):
        stop = min(start + block_size, num_kept)
        local_rows = diagonal_rows[start:stop]
        block_width = int(stop - start)
        current = torch.zeros((num_kept, block_width), dtype=dtype, device=device)
        current[local_rows, torch.arange(block_width, device=device)] = 1.0
        for step in range(config.k_max):
            current = torch.sparse.mm(matrix_t, current)
            ret[local_rows, step] = current[
                local_rows, torch.arange(block_width, device=device)
            ]
    return ret


def _compute_exact_non_returning_native(
    *,
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    values: torch.Tensor,
    num_kept: int,
    config: RWReturnConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if native_backend is None or not hasattr(
        native_backend, "compute_non_returning_features_flat"
    ):
        raise RuntimeError(
            "non_returning RW propagation requires the native striplan extension. "
            "Rebuild and reinstall the package so _iw_encoding_backend exposes "
            "compute_non_returning_features_flat()."
        )

    row_ptr_cpu = (
        row_ptr
        if row_ptr.device.type == "cpu" and row_ptr.dtype == torch.int32
        else row_ptr.to(device="cpu", dtype=torch.int32)
    )
    col_ind_cpu = (
        col_ind
        if col_ind.device.type == "cpu" and col_ind.dtype == torch.int32
        else col_ind.to(device="cpu", dtype=torch.int32)
    )
    values_cpu = (
        values
        if values.device.type == "cpu" and values.dtype == torch.float32
        else values.to(device="cpu", dtype=torch.float32)
    )
    flat = torch.utils.dlpack.from_dlpack(
        native_backend.compute_non_returning_features_flat(
            row_ptr_cpu,
            col_ind_cpu,
            values_cpu,
            num_kept=num_kept,
            k_max=config.k_max,
            block_size=min(config.block_size, num_kept),
        )
    )
    return flat.view(num_kept, config.k_max).to(device=device, dtype=dtype)


def compute_rw_return_features_from_csr(
    *,
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    values: torch.Tensor,
    kept_to_global: torch.Tensor,
    num_total_nodes: int,
    config: RWReturnConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    config = config.canonicalized()
    if not config.enabled:
        return torch.empty((num_total_nodes, 0), dtype=dtype, device=device)
    if kept_to_global.numel() == 0:
        return torch.zeros((num_total_nodes, config.k_max), dtype=dtype, device=device)

    num_kept = int(kept_to_global.numel())

    if config.walk_mode == RW_WALK_NON_RETURNING:
        cpu_device = torch.device("cpu")
        kept_ret = _compute_exact_non_returning_native(
            row_ptr=row_ptr.to(device=cpu_device, dtype=torch.int32),
            col_ind=col_ind.to(device=cpu_device, dtype=torch.int32),
            values=values.to(device=cpu_device, dtype=torch.float32),
            num_kept=num_kept,
            config=config,
            dtype=torch.float32,
            device=cpu_device,
        )
        kept_ret = kept_ret.to(device=device, dtype=dtype)
        full_ret = torch.zeros(
            (num_total_nodes, config.k_max),
            dtype=dtype,
            device=device,
        )
        full_ret.index_copy_(0, kept_to_global.to(device=device, dtype=torch.int64), kept_ret)
        return full_ret

    row_ptr = row_ptr.to(device=device, dtype=torch.int64)
    col_ind = col_ind.to(device=device, dtype=torch.int64)
    values = values.to(device=device, dtype=dtype)
    kept_to_global = kept_to_global.to(device=device, dtype=torch.int64)

    effective_mode = config.mode
    if (
        config.mode == RW_RETURN_EXACT_FULL
        and config.force_block_threshold > 0
        and num_kept > config.force_block_threshold
    ):
        warnings.warn(
            "Falling back to exact_blocked for RW-return features because the kept-node count exceeds rw_return_force_block_threshold.",
            stacklevel=2,
        )
        effective_mode = RW_RETURN_EXACT_BLOCKED
    matrix_t = torch.sparse_csr_tensor(
        row_ptr,
        col_ind,
        values,
        size=(num_kept, num_kept),
        dtype=dtype,
        device=device,
        check_invariants=False,
    )
    if effective_mode == RW_RETURN_EXACT_FULL:
        current = torch.eye(num_kept, dtype=dtype, device=device)
        kept_ret = torch.zeros((num_kept, config.k_max), dtype=dtype, device=device)
        for step in range(config.k_max):
            current = torch.sparse.mm(matrix_t, current)
            kept_ret[:, step] = torch.diagonal(current)
    else:
        kept_ret = _compute_exact_blocked(
            matrix_t,
            num_kept,
            config=config,
            dtype=dtype,
            device=device,
        )

    full_ret = torch.zeros((num_total_nodes, config.k_max), dtype=dtype, device=device)
    full_ret.index_copy_(0, kept_to_global, kept_ret)
    return full_ret