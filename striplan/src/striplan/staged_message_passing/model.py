from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, Callable

import pymimir as mm
import pymimir_rl as rl
import pymimir_rgnn as rgnn
import pymimir_rgnn.model as rgnn_model
import torch

from ..encoders import (
    DepthAwareExtendedRGNN,
    EnhancedStateEncoder,
    ExtendedEncodedTensors,
    ExtendedRGNN,
    TransitionTreeEncoder,
)
from ..model import (
    CONCURRENT_INTERNAL_ACTIONS_READOUTS,
    IWModel as BaseIWModel,
    TransitionPairScalarDecoder,
    concurrent_internal_actions_forward,
    create_aggregation_function,
)


SUPPORTED_ENCODINGS = tuple(CONCURRENT_INTERNAL_ACTIONS_READOUTS.keys())


@dataclass(frozen=True)
class StageLayout:
    world_layers: int
    tree_layers: int
    joint_layers: int
    residual_fusion: bool

    @property
    def total_layers(self) -> int:
        return self.world_layers + self.tree_layers + self.joint_layers


def _filter_relations(
    relations: dict[str, torch.Tensor], allowed_relation_names: set[str]
) -> dict[str, torch.Tensor]:
    return {
        relation_name: relation_tensor
        for relation_name, relation_tensor in relations.items()
        if relation_name in allowed_relation_names
    }


def _has_active_relations(relations: dict[str, torch.Tensor]) -> bool:
    return any(relation_tensor.numel() > 0 for relation_tensor in relations.values())


class _StageTower(torch.nn.Module):
    def __init__(
        self,
        hparam_config: rgnn.HyperparameterConfig,
        input_spec: tuple[Any, ...],
        aggregation_name: str,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self._config = hparam_config
        module_config = rgnn.ModuleConfig(
            aggregation_function=create_aggregation_function(aggregation_name),
            message_function=rgnn.PredicateMLPMessages(
                hparam_config, input_spec
            ),
            update_function=rgnn.MLPUpdates(hparam_config),
        )
        self._relation_network = rgnn_model.RelationalLayerModule(
            hparam_config, module_config
        )
        self._message = module_config.message_function
        if hparam_config.global_readout:
            self._global_readout = rgnn.SumReadout(
                hparam_config.embedding_size,
                hparam_config.embedding_size,
            )
            self._global_update = rgnn.MLP(
                2 * hparam_config.embedding_size,
                hparam_config.embedding_size,
            )
        if hparam_config.normalize_updates:
            self._update_normalization = torch.nn.LayerNorm(
                hparam_config.embedding_size
            )

    def run(
        self,
        node_embeddings: torch.Tensor,
        relations: dict[str, torch.Tensor],
        node_sizes: torch.Tensor,
        start_iteration: int,
        notify_hook: Callable[[int, torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, int]:
        if self.num_layers <= 0 or not _has_active_relations(relations):
            return node_embeddings, start_iteration
        self._message.setup(relations)
        try:
            next_iteration = start_iteration
            for _ in range(self.num_layers):
                next_node_embeddings = self._relation_network(
                    node_embeddings, relations
                )
                if self._config.normalize_updates:
                    next_node_embeddings = self._update_normalization(
                        next_node_embeddings
                    )
                if self._config.global_readout:
                    global_embedding = self._global_readout(
                        node_embeddings, node_sizes
                    )
                    global_messages = self._global_update(
                        torch.cat(
                            (
                                node_embeddings,
                                global_embedding.repeat_interleave(
                                    node_sizes, dim=0
                                ),
                            ),
                            dim=1,
                        )
                    )
                    if self._config.normalize_updates:
                        global_messages = self._update_normalization(
                            global_messages
                        )
                    next_node_embeddings = next_node_embeddings + global_messages
                if self._config.binarize_updates:
                    next_node_embeddings = rgnn.utils.gumbel_sigmoid(
                        next_node_embeddings, hard=True
                    )
                if self._config.residual_updates:
                    next_node_embeddings = node_embeddings + next_node_embeddings
                node_embeddings = next_node_embeddings
                if notify_hook is not None:
                    notify_hook(next_iteration, node_embeddings)
                next_iteration += 1
        finally:
            self._message.cleanup()
        return node_embeddings, next_iteration


class StagedRelationalLayerStackModule(torch.nn.Module):
    def __init__(
        self,
        hparam_config: rgnn.HyperparameterConfig,
        input_spec: tuple[Any, ...],
        aggregation_name: str,
        stage_layout: StageLayout,
    ) -> None:
        super().__init__()
        self._config = hparam_config
        self._stage_layout = stage_layout
        self._world_tower = _StageTower(
            hparam_config,
            input_spec,
            aggregation_name,
            stage_layout.world_layers,
        )
        self._tree_tower = _StageTower(
            hparam_config,
            input_spec,
            aggregation_name,
            stage_layout.tree_layers,
        )
        self._joint_tower = _StageTower(
            hparam_config,
            input_spec,
            aggregation_name,
            stage_layout.joint_layers,
        )
        self._embedding_size = hparam_config.embedding_size
        self._joint_fusion = (
            torch.nn.Linear(3 * self._embedding_size, self._embedding_size)
            if stage_layout.residual_fusion
            else None
        )
        self._hooks: list[Callable[[int, torch.Tensor], None]] = []
        state_encoder, tree_encoder, goal_encoder = input_spec
        domain = hparam_config.domain
        self._world_relation_names = {
            relation_name
            for relation_name, _ in state_encoder.get_relations(domain)
        } | {
            relation_name
            for relation_name, _ in goal_encoder.get_relations(domain)
        }
        self._tree_relation_names = {
            relation_name
            for relation_name, _ in tree_encoder.get_relations(domain)
        }

    def add_hook(self, hook_func: Callable[[int, torch.Tensor], None]) -> None:
        self._hooks.append(hook_func)

    def clear_hooks(self) -> None:
        self._hooks.clear()

    def _notify_hooks(self, iteration: int, embeddings: torch.Tensor) -> None:
        for hook in self._hooks:
            hook(iteration, embeddings)

    def _initialize_node_embeddings(
        self, input_tensors: ExtendedEncodedTensors
    ) -> torch.Tensor:
        return torch.zeros(
            [int(input_tensors.node_sizes.sum()), self._embedding_size],
            dtype=torch.float,
            requires_grad=True,
            device=input_tensors.node_sizes.device,
        )

    def _joint_input(
        self,
        base_embeddings: torch.Tensor,
        world_embeddings: torch.Tensor,
        tree_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if self._joint_fusion is None:
            return tree_embeddings
        fused_embeddings = self._joint_fusion(
            torch.cat(
                (base_embeddings, world_embeddings, tree_embeddings), dim=1
            )
        )
        return tree_embeddings + fused_embeddings

    def forward(self, input_tensors: ExtendedEncodedTensors) -> torch.Tensor:
        base_embeddings = self._initialize_node_embeddings(input_tensors)
        world_relations = _filter_relations(
            input_tensors.flattened_relations, self._world_relation_names
        )
        tree_relations = _filter_relations(
            input_tensors.flattened_relations, self._tree_relation_names
        )
        world_embeddings, next_iteration = self._world_tower.run(
            base_embeddings,
            world_relations,
            input_tensors.node_sizes,
            0,
            self._notify_hooks,
        )
        tree_embeddings, next_iteration = self._tree_tower.run(
            world_embeddings,
            tree_relations,
            input_tensors.node_sizes,
            next_iteration,
            self._notify_hooks,
        )
        joint_embeddings, _ = self._joint_tower.run(
            self._joint_input(
                base_embeddings, world_embeddings, tree_embeddings
            ),
            input_tensors.flattened_relations,
            input_tensors.node_sizes,
            next_iteration,
            self._notify_hooks,
        )
        return joint_embeddings


class DepthAwareStagedRelationalLayerStackModule(
    StagedRelationalLayerStackModule
):
    def __init__(
        self,
        hparam_config: rgnn.HyperparameterConfig,
        input_spec: tuple[Any, ...],
        aggregation_name: str,
        stage_layout: StageLayout,
    ) -> None:
        super().__init__(
            hparam_config,
            input_spec,
            aggregation_name,
            stage_layout,
        )
        self._depth_projection = torch.nn.Linear(
            1, hparam_config.embedding_size, bias=False
        )

    def _initialize_node_embeddings(
        self, input_tensors: ExtendedEncodedTensors
    ) -> torch.Tensor:
        node_embeddings = super()._initialize_node_embeddings(input_tensors)
        if input_tensors.action_depths.numel() == 0:
            return node_embeddings
        assert input_tensors.action_depths.shape == input_tensors.action_depth_mask.shape
        assert input_tensors.action_depths.shape == input_tensors.action_indices.shape
        masked_action_indices = input_tensors.action_indices[
            input_tensors.action_depth_mask
        ].long()
        if masked_action_indices.numel() == 0:
            return node_embeddings
        depth_embeddings = self._depth_projection(
            input_tensors.action_depths[
                input_tensors.action_depth_mask
            ].view(-1, 1)
        )
        return node_embeddings.index_copy(
            0, masked_action_indices, depth_embeddings
        )


class StagedExtendedRGNN(ExtendedRGNN):
    def __init__(
        self,
        hparam_config: rgnn.HyperparameterConfig,
        input_spec: tuple[Any, ...],
        output_spec: list[tuple[str, rgnn.Decoder]],
        *,
        aggregation_name: str,
        stage_layout: StageLayout,
    ) -> None:
        module_config = rgnn.ModuleConfig(
            aggregation_function=create_aggregation_function(aggregation_name),
            message_function=rgnn.PredicateMLPMessages(
                hparam_config, input_spec
            ),
            update_function=rgnn.MLPUpdates(hparam_config),
        )
        super().__init__(hparam_config, module_config, input_spec, output_spec)
        self._aggregation_name = aggregation_name
        self._stage_layout = stage_layout
        self._mpnn_module = StagedRelationalLayerStackModule(
            hparam_config,
            input_spec,
            aggregation_name,
            stage_layout,
        )

    def clone(self) -> "StagedExtendedRGNN":
        model_clone = StagedExtendedRGNN(
            self._hparam_config,
            self._input_spec,
            self._output_spec,
            aggregation_name=self._aggregation_name,
            stage_layout=self._stage_layout,
        )
        model_clone.load_state_dict(self.state_dict())
        return model_clone.to(self.get_device())


class DepthAwareStagedExtendedRGNN(DepthAwareExtendedRGNN):
    def __init__(
        self,
        hparam_config: rgnn.HyperparameterConfig,
        input_spec: tuple[Any, ...],
        output_spec: list[tuple[str, rgnn.Decoder]],
        *,
        aggregation_name: str,
        stage_layout: StageLayout,
    ) -> None:
        module_config = rgnn.ModuleConfig(
            aggregation_function=create_aggregation_function(aggregation_name),
            message_function=rgnn.PredicateMLPMessages(
                hparam_config, input_spec
            ),
            update_function=rgnn.MLPUpdates(hparam_config),
        )
        super().__init__(hparam_config, module_config, input_spec, output_spec)
        self._aggregation_name = aggregation_name
        self._stage_layout = stage_layout
        self._mpnn_module = DepthAwareStagedRelationalLayerStackModule(
            hparam_config,
            input_spec,
            aggregation_name,
            stage_layout,
        )

    def clone(self) -> "DepthAwareStagedExtendedRGNN":
        model_clone = DepthAwareStagedExtendedRGNN(
            self._hparam_config,
            self._input_spec,
            self._output_spec,
            aggregation_name=self._aggregation_name,
            stage_layout=self._stage_layout,
        )
        model_clone.load_state_dict(self.state_dict())
        return model_clone.to(self.get_device())


def _create_input_spec() -> tuple[Any, ...]:
    return (
        EnhancedStateEncoder(add_source_state_handle=True),
        TransitionTreeEncoder(
            connect_to_root_handle=True,
            encode_actions=True,
            encode_effects=True,
            separate_action_nodes=True,
        ),
        rgnn.GoalEncoder(),
    )


def _create_output_spec(
    hparam_config: rgnn.HyperparameterConfig, encoding: str
) -> list[tuple[str, rgnn.Decoder]]:
    readout_spec = CONCURRENT_INTERNAL_ACTIONS_READOUTS.get(encoding)
    if readout_spec is None:
        raise RuntimeError(f"Unsupported staged encoding: {encoding!r}.")
    if readout_spec.node_roles is None:
        return [(readout_spec.readout_name, rgnn.ActionScalarDecoder(hparam_config))]
    return [
        (
            readout_spec.readout_name,
            TransitionPairScalarDecoder(
                hparam_config,
                node_roles=readout_spec.node_roles,
                include_root_state=readout_spec.include_root_state,
            ),
        )
    ]


def _build_internal_model(
    domain: mm.Domain,
    embedding_size: int,
    aggregation_name: str,
    binary_embeddings: bool,
    encoding: str,
    stage_layout: StageLayout,
    iw_depth_feature: bool,
    input_spec: tuple[Any, ...] | None = None,
    output_spec: list[tuple[str, rgnn.Decoder]] | None = None,
) -> rgnn.RelationalGraphNeuralNetwork:
    input_spec = _create_input_spec() if input_spec is None else input_spec
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=stage_layout.total_layers,
        binarize_updates=binary_embeddings,
        residual_updates=not binary_embeddings,
        normalize_updates=not binary_embeddings,
    )
    output_spec = (
        _create_output_spec(hparam_config, encoding)
        if output_spec is None
        else output_spec
    )
    model_class: type[rgnn.RelationalGraphNeuralNetwork]
    model_class = (
        DepthAwareStagedExtendedRGNN
        if iw_depth_feature
        else StagedExtendedRGNN
    )
    return model_class(
        hparam_config,
        input_spec,
        output_spec,
        aggregation_name=aggregation_name,
        stage_layout=stage_layout,
    )


class StagedIWModel(BaseIWModel):
    def __init__(
        self,
        encoding: str,
        modules: dict[str, torch.nn.Module],
        forward_function: Callable[..., Any],
        *,
        aggregation_name: str,
        binary_embeddings: bool,
        stage_layout: StageLayout,
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
    ) -> None:
        super().__init__(
            encoding,
            modules,
            forward_function,
            verbose=verbose,
            projective_iw=projective_iw,
            typed_projection=typed_projection,
            abstracted_iw=abstracted_iw,
            abstracted_iw_width=abstracted_iw_width,
            base_abstracted_iw=base_abstracted_iw,
            preserve_goal_atoms=preserve_goal_atoms,
            keep_depth_one_novel=keep_depth_one_novel,
            keep_goal_nonunary_atoms=keep_goal_nonunary_atoms,
            iw_layer_ordering_strategy=iw_layer_ordering_strategy,
            iw_max_next_layer_states=iw_max_next_layer_states,
            iw_beam_width=iw_beam_width,
            iw_beam_novelty_mode=iw_beam_novelty_mode,
            iw_max_depth=iw_max_depth,
            iw1_precheck_add_effect_novelty=iw1_precheck_add_effect_novelty,
            iw1_atom_first_mode=iw1_atom_first_mode,
            iw1_atom_first_ratio=iw1_atom_first_ratio,
            iw1_incremental_first_applicability=iw1_incremental_first_applicability,
            iw1_incremental_first_applicability_debug_crosscheck=iw1_incremental_first_applicability_debug_crosscheck,
            iw_relaxed_survivors_only_beam=iw_relaxed_survivors_only_beam,
            iw_randomize_equal_score_ties=iw_randomize_equal_score_ties,
            iw_equal_score_tie_seed=iw_equal_score_tie_seed,
            iw_num_threads=iw_num_threads,
            iw_chunk_size=iw_chunk_size,
            iw_depth_feature=iw_depth_feature,
        )
        self.aggregation_name = aggregation_name
        self.binary_embeddings = binary_embeddings
        self.stage_layout = stage_layout

    @staticmethod
    def new(
        domain: mm.Domain,
        embedding_size: int,
        aggregation_name: str,
        binary_embeddings: bool,
        encoding: str,
        *,
        world_layers: int,
        tree_layers: int,
        joint_layers: int,
        residual_fusion: bool,
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
    ) -> "StagedIWModel":
        if flat:
            raise RuntimeError(
                "Staged message passing is only implemented for IW tree encodings."
            )
        if encoding not in SUPPORTED_ENCODINGS:
            raise RuntimeError(
                f"Unsupported staged encoding {encoding!r}; expected one of {SUPPORTED_ENCODINGS!r}."
            )
        stage_layout = StageLayout(
            world_layers=world_layers,
            tree_layers=tree_layers,
            joint_layers=joint_layers,
            residual_fusion=residual_fusion,
        )
        internal_model = _build_internal_model(
            domain,
            embedding_size,
            aggregation_name,
            binary_embeddings,
            encoding,
            stage_layout,
            iw_depth_feature,
        )
        modules: dict[str, torch.nn.Module] = {"internal_model": internal_model}
        return StagedIWModel(
            encoding,
            modules,
            partial(concurrent_internal_actions_forward, flat=False),
            aggregation_name=aggregation_name,
            binary_embeddings=binary_embeddings,
            stage_layout=stage_layout,
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
            iw_depth_feature=iw_depth_feature,
        )

    def save(self, path: Path) -> None:
        assert hasattr(self, "internal_model")
        assert isinstance(
            self.internal_model, rgnn.RelationalGraphNeuralNetwork
        )
        hparam_dict = {
            field.name: getattr(self.internal_model.get_hparam_config(), field.name)
            for field in fields(self.internal_model.get_hparam_config())
        }
        del hparam_dict["domain"]
        extras: dict[str, Any] = {
            "staged_message_passing": True,
            "encoding": self.encoding,
            "aggregation_name": self.aggregation_name,
            "binary_embeddings": self.binary_embeddings,
            "world_layers": self.stage_layout.world_layers,
            "tree_layers": self.stage_layout.tree_layers,
            "joint_layers": self.stage_layout.joint_layers,
            "residual_fusion": self.stage_layout.residual_fusion,
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
            "iw_depth_feature": self.iw_depth_feature,
        }
        checkpoint = {
            "model": self.internal_model.state_dict(),
            "hparam_config": hparam_dict,
            "input_spec": self.internal_model._input_spec,
            "output_spec": self.internal_model._output_spec,
            "extras": extras,
        }
        torch.save(checkpoint, path)

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
    ) -> "StagedIWModel":
        if flat:
            raise RuntimeError("Flat mode is not supported for staged checkpoints.")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        extras = checkpoint["extras"]
        if not extras.get("staged_message_passing", False):
            raise RuntimeError(
                f"Checkpoint {path} is not a staged-message-passing checkpoint."
            )
        encoding = extras["encoding"]
        stage_layout = StageLayout(
            world_layers=int(extras["world_layers"]),
            tree_layers=int(extras["tree_layers"]),
            joint_layers=int(extras["joint_layers"]),
            residual_fusion=bool(extras.get("residual_fusion", True)),
        )
        hparam_config = rgnn.HyperparameterConfig(
            domain=domain, **checkpoint["hparam_config"]
        )
        internal_model = _build_internal_model(
            domain,
            hparam_config.embedding_size,
            extras["aggregation_name"],
            bool(extras["binary_embeddings"]),
            encoding,
            stage_layout,
            bool(extras.get("iw_depth_feature", False)),
            input_spec=checkpoint["input_spec"],
            output_spec=checkpoint["output_spec"],
        )
        internal_model.load_state_dict(checkpoint["model"])
        internal_model = internal_model.to(device)
        modules: dict[str, torch.nn.Module] = {"internal_model": internal_model}
        return StagedIWModel(
            encoding,
            modules,
            partial(concurrent_internal_actions_forward, flat=False),
            aggregation_name=extras["aggregation_name"],
            binary_embeddings=bool(extras["binary_embeddings"]),
            stage_layout=stage_layout,
            projective_iw=(
                extras.get("projective_iw", False)
                if projective_iw is None
                else projective_iw
            ),
            typed_projection=(
                extras.get("typed_projection", False)
                if typed_projection is None
                else typed_projection
            ),
            abstracted_iw=(
                extras.get("abstracted_iw", extras.get("projective_iw", False))
                if abstracted_iw is None
                else abstracted_iw
            ),
            abstracted_iw_width=(
                extras.get("abstracted_iw_width", 1)
                if abstracted_iw_width is None
                else abstracted_iw_width
            ),
            base_abstracted_iw=(
                extras.get(
                    "base_abstracted_iw",
                    extras.get("projective_iw", False)
                    and not extras.get("typed_projection", False),
                )
                if base_abstracted_iw is None
                else base_abstracted_iw
            ),
            preserve_goal_atoms=(
                extras.get("preserve_goal_atoms", True)
                if preserve_goal_atoms is None
                else preserve_goal_atoms
            ),
            keep_depth_one_novel=(
                extras.get("keep_depth_one_novel", True)
                if keep_depth_one_novel is None
                else keep_depth_one_novel
            ),
            keep_goal_nonunary_atoms=(
                extras.get("keep_goal_nonunary_atoms", False)
                if keep_goal_nonunary_atoms is None
                else keep_goal_nonunary_atoms
            ),
            iw_layer_ordering_strategy=(
                extras.get("iw_layer_ordering_strategy")
                if iw_layer_ordering_strategy is None
                else iw_layer_ordering_strategy
            ),
            iw_max_next_layer_states=(
                extras.get("iw_max_next_layer_states", -1)
                if iw_max_next_layer_states is None
                else iw_max_next_layer_states
            ),
            iw_beam_width=(
                extras.get("iw_beam_width", -1)
                if iw_beam_width is None
                else iw_beam_width
            ),
            iw_beam_novelty_mode=(
                extras.get("iw_beam_novelty_mode", "all_tested")
                if iw_beam_novelty_mode is None
                else iw_beam_novelty_mode
            ),
            iw_max_depth=(
                extras.get("iw_max_depth", -1)
                if iw_max_depth is None
                else iw_max_depth
            ),
            iw1_precheck_add_effect_novelty=(
                extras.get("iw1_precheck_add_effect_novelty", False)
                if iw1_precheck_add_effect_novelty is None
                else iw1_precheck_add_effect_novelty
            ),
            iw1_atom_first_mode=(
                extras.get("iw1_atom_first_mode", False)
                if iw1_atom_first_mode is None
                else iw1_atom_first_mode
            ),
            iw1_atom_first_ratio=(
                extras.get("iw1_atom_first_ratio", 1.0)
                if iw1_atom_first_ratio is None
                else iw1_atom_first_ratio
            ),
            iw1_incremental_first_applicability=(
                extras.get("iw1_incremental_first_applicability", False)
                if iw1_incremental_first_applicability is None
                else iw1_incremental_first_applicability
            ),
            iw1_incremental_first_applicability_debug_crosscheck=(
                extras.get(
                    "iw1_incremental_first_applicability_debug_crosscheck",
                    False,
                )
                if iw1_incremental_first_applicability_debug_crosscheck
                is None
                else iw1_incremental_first_applicability_debug_crosscheck
            ),
            iw_relaxed_survivors_only_beam=(
                extras.get("iw_relaxed_survivors_only_beam", False)
                if iw_relaxed_survivors_only_beam is None
                else iw_relaxed_survivors_only_beam
            ),
            iw_randomize_equal_score_ties=(
                extras.get("iw_randomize_equal_score_ties", False)
                if iw_randomize_equal_score_ties is None
                else iw_randomize_equal_score_ties
            ),
            iw_equal_score_tie_seed=(
                extras.get("iw_equal_score_tie_seed")
                if iw_equal_score_tie_seed is None
                else iw_equal_score_tie_seed
            ),
            iw_num_threads=(
                extras.get("iw_num_threads", -1)
                if iw_num_threads is None
                else iw_num_threads
            ),
            iw_chunk_size=(
                extras.get("iw_chunk_size", -1)
                if iw_chunk_size is None
                else iw_chunk_size
            ),
            iw_depth_feature=bool(extras.get("iw_depth_feature", False)),
        ).to(device)
