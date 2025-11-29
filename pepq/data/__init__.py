"""
data package: lightweight preprocessing utilities
"""

from .build_data import build_data
from .helpers import _ensure_dataframe, ArrayLike
from .steps import RemoveDuplicatesStep, VarianceFilterStep, ScalerStep
from .config import PreprocessorConfig, DedupConfig, VarianceConfig, ScalerConfig
from .preprocessor import DataPreprocessor
from .feature_importance import FeatureImportanceReport

__all__ = [
    "build_data",
    "ArrayLike",
    "_ensure_dataframe",
    "RemoveDuplicatesStep",
    "VarianceFilterStep",
    "ScalerStep",
    "DedupConfig",
    "VarianceConfig",
    "ScalerConfig",
    "PreprocessorConfig",
    "DataPreprocessor",
    "FeatureImportanceReport",
]
