from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


IW_DEPTH_NONE = "none"
IW_DEPTH_INIT_SCALAR = "init_scalar"
IW_DEPTH_NODES = "depth_nodes"
IW_DEPTH_MODES = (
    IW_DEPTH_NONE,
    IW_DEPTH_INIT_SCALAR,
    IW_DEPTH_NODES,
)

IW_DEPTH_PRECEDES_MLP = "mlp"
IW_DEPTH_PRECEDES_FACTORIZED = "factorized"
IW_DEPTH_PRECEDES_MODES = (
    IW_DEPTH_PRECEDES_MLP,
    IW_DEPTH_PRECEDES_FACTORIZED,
)

IW_IN_DEPTH_RELATION = "iw_in_depth"
IW_DEPTH_PRECEDES_RELATION = "iw_depth_precedes"


@dataclass(frozen=True)
class IWDepthConfig:
    mode: str = IW_DEPTH_NONE
    precedes_mode: str = IW_DEPTH_PRECEDES_MLP
    precedes_scan: bool = False

    @property
    def uses_init_scalar(self) -> bool:
        return self.mode == IW_DEPTH_INIT_SCALAR

    @property
    def uses_depth_nodes(self) -> bool:
        return self.mode == IW_DEPTH_NODES

    def canonicalized(self) -> "IWDepthConfig":
        if self.mode not in IW_DEPTH_MODES:
            raise RuntimeError(
                f"Unknown iw_depth_mode={self.mode!r}. "
                f"Expected one of {IW_DEPTH_MODES!r}."
            )
        if self.precedes_mode not in IW_DEPTH_PRECEDES_MODES:
            raise RuntimeError(
                f"Unknown iw_depth_precedes_mode={self.precedes_mode!r}. "
                f"Expected one of {IW_DEPTH_PRECEDES_MODES!r}."
            )
        if self.mode != IW_DEPTH_NODES:
            return IWDepthConfig(mode=self.mode)
        return self

    def validate(
        self,
        *,
        encoding: str | None = None,
        tree_encodings: set[str] | None = None,
        aggregation_name: str | None = None,
    ) -> "IWDepthConfig":
        config = self.canonicalized()
        if (
            config.mode != IW_DEPTH_NONE
            and encoding is not None
            and tree_encodings is not None
            and encoding not in tree_encodings
        ):
            raise RuntimeError(
                f"iw_depth_mode={config.mode!r} is only supported for IW tree encodings, "
                f"got encoding={encoding!r}."
            )
        if (
            config.uses_depth_nodes
            and config.precedes_mode == IW_DEPTH_PRECEDES_MLP
            and config.precedes_scan
        ):
            raise RuntimeError(
                "iw_depth_precedes_scan=True is incompatible with "
                "iw_depth_precedes_mode='mlp'."
            )
        if (
            config.uses_depth_nodes
            and config.precedes_scan
            and aggregation_name is not None
            and aggregation_name != "smax"
        ):
            raise RuntimeError(
                "iw_depth_precedes_scan=True is only supported with "
                "aggregation='smax'."
            )
        return config

    def to_extras(self) -> dict[str, Any]:
        return {
            "iw_depth_mode": self.mode,
            "iw_depth_precedes_mode": self.precedes_mode,
            "iw_depth_precedes_scan": self.precedes_scan,
            "iw_depth_feature": self.uses_init_scalar,
        }

    @staticmethod
    def from_extras(extras: Mapping[str, Any]) -> "IWDepthConfig":
        return resolve_iw_depth_config(
            iw_depth_feature=bool(extras.get("iw_depth_feature", False)),
            iw_depth_mode=extras.get("iw_depth_mode"),
            iw_depth_precedes_mode=extras.get("iw_depth_precedes_mode"),
            iw_depth_precedes_scan=extras.get("iw_depth_precedes_scan"),
        )


def resolve_iw_depth_config(
    *,
    iw_depth_feature: bool = False,
    iw_depth_mode: str | None = None,
    iw_depth_precedes_mode: str | None = None,
    iw_depth_precedes_scan: bool | None = None,
) -> IWDepthConfig:
    mode = (
        IW_DEPTH_INIT_SCALAR
        if iw_depth_feature and iw_depth_mode is None
        else (IW_DEPTH_NONE if iw_depth_mode is None else iw_depth_mode)
    )
    precedes_mode = (
        IW_DEPTH_PRECEDES_MLP
        if iw_depth_precedes_mode is None
        else iw_depth_precedes_mode
    )
    precedes_scan = (
        False if iw_depth_precedes_scan is None else iw_depth_precedes_scan
    )
    return IWDepthConfig(
        mode=mode,
        precedes_mode=precedes_mode,
        precedes_scan=precedes_scan,
    ).canonicalized()


def cli_requested_iw_depth_override(args: Any) -> bool:
    return bool(
        getattr(args, "iw_depth_feature", False)
        or getattr(args, "iw_depth_mode", None) is not None
        or getattr(args, "iw_depth_precedes_mode", None) is not None
        or getattr(args, "iw_depth_precedes_scan", None) is not None
    )
