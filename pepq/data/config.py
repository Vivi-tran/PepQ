"""
data.config
-----------

Configuration dataclasses for the preprocessor pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from .steps import RemoveDuplicatesStep, VarianceFilterStep, ScalerStep


@dataclass
class DedupConfig:
    """
    Configuration for the deduplication step.

    :param remove_rows: Drop duplicated rows (keep first).
    :param remove_columns: Drop duplicated columns by identical content.
    """

    remove_rows: bool = True
    remove_columns: bool = False

    def set_remove_rows(self, v: bool) -> "DedupConfig":
        self.remove_rows = bool(v)
        return self

    def set_remove_columns(self, v: bool) -> "DedupConfig":
        self.remove_columns = bool(v)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VarianceConfig:
    """
    Configuration for the variance filter step.

    :param threshold: population variance threshold (ddof=0).
    """

    threshold: float = 0.00

    def set_threshold(self, t: float) -> "VarianceConfig":
        self.threshold = float(t)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalerConfig:
    """
    Configuration for the scaler step.

    :param scaler: 'standard' | 'minmax' | 'robust' | None
    """

    scaler: Optional[str] = "standard"

    def set_scaler(self, s: Optional[str]) -> "ScalerConfig":
        self.scaler = None if s is None else str(s)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessorConfig:
    """
    Top-level configuration for the full preprocessor.

    :param dedup: Dedup configuration (DedupConfig).
    :param variance: Variance filter configuration (VarianceConfig).
    :param scaler: Scaler configuration (ScalerConfig).
    :param return_dataframe: Whether transformed output should be DataFrame.
    """

    dedup: DedupConfig = field(default_factory=DedupConfig)
    variance: VarianceConfig = field(default_factory=VarianceConfig)
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    return_dataframe: bool = False

    def set_dedup(
        self, remove_rows: Optional[bool] = None, remove_columns: Optional[bool] = None
    ) -> "PreprocessorConfig":
        if remove_rows is not None:
            self.dedup.remove_rows = bool(remove_rows)
        if remove_columns is not None:
            self.dedup.remove_columns = bool(remove_columns)
        return self

    def set_variance(self, threshold: Optional[float] = None) -> "PreprocessorConfig":
        if threshold is not None:
            self.variance.threshold = float(threshold)
        return self

    def set_scaler(self, scaler: Optional[str]) -> "PreprocessorConfig":
        self.scaler.scaler = None if scaler is None else str(scaler)
        return self

    def set_return_dataframe(self, flag: bool) -> "PreprocessorConfig":
        self.return_dataframe = bool(flag)
        return self

    def validate(self) -> None:
        if self.variance.threshold < 0:
            raise ValueError("variance.threshold must be >= 0")
        if self.scaler.scaler is not None and self.scaler.scaler.lower() not in {
            "standard",
            "minmax",
            "robust",
        }:
            raise ValueError(
                "scaler must be one of 'standard','minmax','robust', or None"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dedup": self.dedup.to_dict(),
            "variance": self.variance.to_dict(),
            "scaler": self.scaler.to_dict(),
            "return_dataframe": self.return_dataframe,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessorConfig":
        ded = DedupConfig(**d.get("dedup", {}))
        var = VarianceConfig(**d.get("variance", {}))
        sc = ScalerConfig(**d.get("scaler", {}))
        ret = bool(d.get("return_dataframe", False))
        return cls(dedup=ded, variance=var, scaler=sc, return_dataframe=ret)

    def build_steps(self):
        self.validate()
        ded = RemoveDuplicatesStep(
            remove_rows=self.dedup.remove_rows, remove_columns=self.dedup.remove_columns
        )
        var = VarianceFilterStep(threshold=self.variance.threshold)
        scl = ScalerStep(scaler=self.scaler.scaler)
        return ded, var, scl

    def __repr__(self) -> str:
        return f"PreprocessorConfig(dedup={self.dedup}, variance={self.variance}, scaler={self.scaler}, return_dataframe={self.return_dataframe})"
