from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


LGAN_ANCHOR_ACTIONS = "actions"
LGAN_ANCHOR_GOALS = "goals"
LGAN_ANCHOR_STATES = "states"
LGAN_ANCHOR_SOURCES = (
    LGAN_ANCHOR_ACTIONS,
    LGAN_ANCHOR_GOALS,
    LGAN_ANCHOR_STATES,
)
LGAN_DEFAULT_ANCHOR_SOURCES = (
    LGAN_ANCHOR_ACTIONS,
    LGAN_ANCHOR_GOALS,
)

DEFAULT_LGAN_TN_EDGE_POS = "_lgan_tn_"
DEFAULT_LGAN_NN_EDGE_POS = "_lgan_nn_"
DEFAULT_LGAN_RR_EDGE_POS = "_lgan_rr_"


def _normalize_anchor_sources(
    anchor_sources: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if anchor_sources is None:
        return ()
    if isinstance(anchor_sources, str):
        raw_sources = [part.strip() for part in anchor_sources.split(",")]
    else:
        raw_sources = [str(part).strip() for part in anchor_sources]
    normalized: list[str] = []
    seen: set[str] = set()
    for source in raw_sources:
        if not source:
            continue
        if source not in LGAN_ANCHOR_SOURCES:
            raise RuntimeError(
                f"Unknown lgan_anchor_source={source!r}. "
                f"Expected one of {LGAN_ANCHOR_SOURCES!r}."
            )
        if source in seen:
            continue
        normalized.append(source)
        seen.add(source)
    return tuple(normalized)


@dataclass(frozen=True)
class LGANConfig:
    include_lgan_edges: bool = False
    lgan_anchor_sources: tuple[str, ...] = ()
    lgan_tn_edge_pos: str = DEFAULT_LGAN_TN_EDGE_POS
    lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS
    lgan_rr_edge_pos: str = DEFAULT_LGAN_RR_EDGE_POS

    @property
    def enabled(self) -> bool:
        return self.include_lgan_edges

    def canonicalized(self) -> "LGANConfig":
        lgan_anchor_sources = _normalize_anchor_sources(self.lgan_anchor_sources)
        if not self.include_lgan_edges:
            return LGANConfig(
                include_lgan_edges=False,
                lgan_anchor_sources=(),
                lgan_tn_edge_pos=self.lgan_tn_edge_pos,
                lgan_nn_edge_pos=self.lgan_nn_edge_pos,
                lgan_rr_edge_pos=self.lgan_rr_edge_pos,
            )
        if not lgan_anchor_sources:
            lgan_anchor_sources = LGAN_DEFAULT_ANCHOR_SOURCES
        if not self.lgan_tn_edge_pos:
            raise RuntimeError("lgan_tn_edge_pos must not be empty.")
        if not self.lgan_nn_edge_pos:
            raise RuntimeError("lgan_nn_edge_pos must not be empty.")
        if not self.lgan_rr_edge_pos:
            raise RuntimeError("lgan_rr_edge_pos must not be empty.")
        return LGANConfig(
            include_lgan_edges=True,
            lgan_anchor_sources=lgan_anchor_sources,
            lgan_tn_edge_pos=self.lgan_tn_edge_pos,
            lgan_nn_edge_pos=self.lgan_nn_edge_pos,
            lgan_rr_edge_pos=self.lgan_rr_edge_pos,
        )

    def to_extras(self) -> dict[str, Any]:
        return {
            "include_lgan_edges": self.include_lgan_edges,
            "lgan_anchor_sources": list(self.lgan_anchor_sources),
            "lgan_tn_edge_pos": self.lgan_tn_edge_pos,
            "lgan_nn_edge_pos": self.lgan_nn_edge_pos,
            "lgan_rr_edge_pos": self.lgan_rr_edge_pos,
        }

    @staticmethod
    def from_extras(extras: Mapping[str, Any]) -> "LGANConfig":
        return resolve_lgan_config(
            include_lgan_edges=extras.get("include_lgan_edges"),
            lgan_anchor_sources=extras.get("lgan_anchor_sources"),
            lgan_tn_edge_pos=extras.get("lgan_tn_edge_pos"),
            lgan_nn_edge_pos=extras.get("lgan_nn_edge_pos"),
            lgan_rr_edge_pos=extras.get("lgan_rr_edge_pos"),
        )


def resolve_lgan_config(
    *,
    include_lgan_edges: bool | None = None,
    lgan_anchor_sources: Sequence[str] | str | None = None,
    lgan_tn_edge_pos: str | None = None,
    lgan_nn_edge_pos: str | None = None,
    lgan_rr_edge_pos: str | None = None,
) -> LGANConfig:
    enabled = bool(include_lgan_edges) if include_lgan_edges is not None else False
    anchor_sources = _normalize_anchor_sources(lgan_anchor_sources)
    if enabled and not anchor_sources:
        anchor_sources = LGAN_DEFAULT_ANCHOR_SOURCES
    return LGANConfig(
        include_lgan_edges=enabled,
        lgan_anchor_sources=anchor_sources,
        lgan_tn_edge_pos=(
            DEFAULT_LGAN_TN_EDGE_POS
            if lgan_tn_edge_pos is None
            else str(lgan_tn_edge_pos)
        ),
        lgan_nn_edge_pos=(
            DEFAULT_LGAN_NN_EDGE_POS
            if lgan_nn_edge_pos is None
            else str(lgan_nn_edge_pos)
        ),
        lgan_rr_edge_pos=(
            DEFAULT_LGAN_RR_EDGE_POS
            if lgan_rr_edge_pos is None
            else str(lgan_rr_edge_pos)
        ),
    ).canonicalized()


def cli_requested_lgan_override(args: Any) -> bool:
    return bool(
        getattr(args, "include_lgan_edges", None) is not None
        or getattr(args, "lgan_anchor_sources", None) is not None
        or getattr(args, "lgan_tn_edge_pos", None) is not None
        or getattr(args, "lgan_nn_edge_pos", None) is not None
        or getattr(args, "lgan_rr_edge_pos", None) is not None
    )