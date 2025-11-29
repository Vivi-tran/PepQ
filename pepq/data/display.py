"""
data.display
------------

Reusable NiceDisplayMixin to provide rich Jupyter card + ASCII repr.

This is a small, dependency-light helper used by DataPreprocessor.
"""

from __future__ import annotations
from typing import Any, Dict, List
from html import escape
import pandas as pd


class NiceDisplayMixin:
    """
    Mixin that provides a compact HTML card for Jupyter and a simple ASCII repr.

    Subclasses can override helpers:
      - _repr_params_dict() -> Dict[str,Any]
      - _repr_feature_sets() -> Dict[str, List[str]]
      - _ascii_diagram_lines() -> List[str]
      - summary() -> pandas.DataFrame
    """

    def _repr_params_dict(self) -> Dict[str, Any]:
        # sensible fallback: try sklearn-like get_params()
        if hasattr(self, "get_params"):
            try:
                return self.get_params(deep=False)
            except Exception:
                pass
        return {}

    def _repr_feature_sets(self) -> Dict[str, List[str]]:
        return {}

    def _ascii_diagram_lines(self) -> List[str]:
        return [f"{type(self).__name__}()"]

    def _repr_html_(self) -> str:
        title = escape(type(self).__name__)
        params = self._repr_params_dict()
        try:
            param_df = pd.DataFrame(
                list(params.items()), columns=["param", "value"]
            ).to_html(index=False, escape=True)
        except Exception:
            param_df = "<div><em>Unable to render parameters</em></div>"

        try:
            summary_html = (
                self.summary().to_html(index=False, escape=True)
                if hasattr(self, "summary")
                else "<div><em>No summary</em></div>"
            )
        except Exception:
            summary_html = "<div><em>Summary not available</em></div>"

        # feature chips
        feature_html = ""
        features = self._repr_feature_sets()
        if features:

            def chips(names, color="#0EA5E9"):
                if not names:
                    return "<span style='color:#6B7280;'>None</span>"
                return " ".join(
                    f"<span style='display:inline-block;padding:2px 6px;margin:2px;border-radius:999px;background:{color}20;color:{color};font-size:11px;'>{escape(str(n))}</span>"
                    for n in names[:12]
                )

            feature_html = "<div style='margin-top:8px'><div style='font-weight:600;margin-bottom:4px;'>Feature overview</div>"
            for i, (k, v) in enumerate(features.items()):
                color = ["#0EA5E9", "#EF4444", "#10B981"][i % 3]
                feature_html += f"<div style='margin-bottom:6px'><div style='font-size:11px;color:#374151'>{escape(k)}</div>{chips(v,color=color)}</div>"
            feature_html += "</div>"

        html = f"""
        <div style="font-family:system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size:13px; border:1px solid #E5E7EB; border-radius:8px; padding:10px;background:#FFF;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <div><div style="font-weight:600">{title}</div><div style="font-size:11px;color:#6B7280">{escape(repr(self))}</div></div>
            <div style="font-size:11px;padding:2px 8px;border-radius:999px;background:#E5E7EB;color:#374151">object</div>
          </div>
          <div style="display:flex;gap:12px">
            <div style="flex:1;min-width:200px"><div style="font-weight:600;margin-bottom:4px">Parameters</div>{param_df}</div>
            <div style="flex:1;min-width:200px"><div style="font-weight:600;margin-bottom:4px">Summary</div>{summary_html}</div>
          </div>
          {feature_html}
        </div>
        """
        return html

    def __repr__(self) -> str:
        try:
            return "\n".join(self._ascii_diagram_lines())
        except Exception:
            return super().__repr__()
