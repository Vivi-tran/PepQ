"""
PLDDT/PTM/PAE helper utilities and get_data().

This module provides robust readers for compact summary arrays produced by
MetricsCalculator.summarize_plddt / summarize_pae / calculate_ptm_values and a
helper get_data() that flattens per-complex rank entries and attaches
derived numeric summaries.

Semantic map
------------
- plddt summary array (from summarize_plddt):
    plddt[0] -> mean pLDDT over all residues (float)
    plddt[1] -> median pLDDT over all residues (float)
    plddt[2] -> peptide mean pLDDT (float)            # peptide-specific
    plddt[3] -> interface overall average pLDDT (float)  # average over interface residues (prot+pep)

- ptm array (from calculate_ptm_values):
    ptm[0] -> global pTM (float or None)
    ptm[1:] -> per-chain pTM values sorted by chain ID (chain order is lexicographic)

- pae summary (from summarize_pae):
    pae[0] -> max_pae (numeric or None)
    pae[1] -> mean of flattened PAE matrix (float)
    pae[2] -> median of flattened PAE matrix (float)
    pae[3] -> coverage (float in [0,1]) as defined by longest consecutive run metric
"""

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Sequence,
)
import numpy as np


# ---------- utility summary selector ----------
def _summary_select(
    values: Iterable[float] | None, select: str = "mean"
) -> Optional[float]:
    """
    Safely summarize an iterable of numbers.

    :param values: iterable of numeric-like (or None)
    :param select: one of "mean","median","min","max","first","last"
    :return: float or None
    """
    if values is None:
        return None
    try:
        arr = np.asarray(list(values), dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None

    if select == "mean":
        return float(np.nanmean(arr))
    if select == "median":
        return float(np.nanmedian(arr))
    if select == "min":
        return float(np.nanmin(arr))
    if select == "max":
        return float(np.nanmax(arr))
    if select == "first":
        return float(arr[0])
    if select == "last":
        return float(arr[-1])

    # fallback
    return float(np.nanmean(arr))


# ---------- pLDDT helpers ----------
def _get_plddt(scores: Any, select: str = "mean") -> Optional[float]:
    """
    Robustly extract a protein PLDDT summary.

    Accepts:
      - None -> returns None
      - dict -> tries keys: "prot_plddt", "prot", "protein", then "plddt" (and uses _summary_select)
      - sequence (list/tuple/np.array) -> uses the compact index convention:
            index 0 -> mean
            index 1 -> median
            index 2 -> peptide mean
            index 3 -> interface overall avg
        If the requested index is not present, falls back to _summary_select over numeric entries.
      - single numeric -> returns float(value)

    :param scores: source (list/dict/number/None)
    :param select: "mean","median","peptide","interface","min","max","first","last"
    :return: Optional[float]
    """
    if scores is None:
        return None

    # dict-like: try canonical keys
    if isinstance(scores, Mapping):
        for k in ("prot_plddt", "prot", "protein"):
            if k in scores:
                return _summary_select(
                    scores[k],
                    select=select if select not in ("peptide", "interface") else "mean",
                )
        if "plddt" in scores:
            return _summary_select(
                scores["plddt"],
                select=select if select not in ("peptide", "interface") else "min",
            )

    # numeric scalar
    if isinstance(scores, (int, float, np.floating, np.integer)):
        try:
            return float(scores)
        except Exception:
            return None

    # sequence-like: try index mapping first
    try:
        seq = list(scores)
    except Exception:
        return None

    # filter numeric/non-nan entries for fallback summaries
    numeric = []
    for v in seq:
        try:
            fv = float(v)
            if not (np.isnan(fv) or fv == float("inf") or fv == float("-inf")):
                numeric.append(fv)
        except Exception:
            continue

    # mapping of select -> index in the compact plddt array
    index_map = {
        "mean": 0,
        "median": 1,
        "peptide": 2,
        "pep": 2,
        "interface": 3,
        "interface_avg": 3,
    }

    sel_lower = select.lower()
    if sel_lower in index_map:
        idx = index_map[sel_lower]
        if idx < len(seq):
            try:
                val = float(seq[idx])
                if not (np.isnan(val) or val == float("inf") or val == float("-inf")):
                    return val
            except Exception:
                pass
        # fallback to summarizing numeric values if requested index missing
        return _summary_select(
            numeric,
            select="mean" if sel_lower in ("peptide", "interface") else sel_lower,
        )

    # handle min/max/first/last
    if sel_lower in ("min", "max", "first", "last", "median"):
        return _summary_select(numeric, select=sel_lower)

    # default fallback
    return _summary_select(numeric, select="mean")


def _get_pep_plddt(scores: Any) -> Optional[float]:
    """
    Extract peptide pLDDT.

    Preferred behavior:
      - If a dict contains keys 'pep_plddt','pep','peptide', return the mean of that list.
      - If input is a compact plddt sequence, return index 2 when present.
      - Otherwise, attempt to return the minimum pLDDT among numeric entries (conservative).
    """
    if scores is None:
        return None

    if isinstance(scores, Mapping):
        for k in ("pep_plddt", "pep", "peptide"):
            if k in scores:
                return _summary_select(scores[k], select="mean")
        if "plddt" in scores:
            # peptide may be embedded — choose a conservative summary (min)
            return _summary_select(scores["plddt"], select="min")

    # sequence-like
    try:
        seq = list(scores)
    except Exception:
        # try numeric scalar
        try:
            return float(scores)
        except Exception:
            return None

    if len(seq) > 2:
        try:
            v = float(seq[2])
            if not (np.isnan(v) or v == float("inf") or v == float("-inf")):
                return v
        except Exception:
            pass

    # fallback: conservative -> min of numeric entries
    numeric = []
    for v in seq:
        try:
            fv = float(v)
            if not (np.isnan(fv) or fv == float("inf") or fv == float("-inf")):
                numeric.append(fv)
        except Exception:
            continue
    return _summary_select(numeric, select="min")


# ---------- pTM helpers ----------
def _get_ptml(scores: Any, select: str = "mean") -> Optional[float]:
    """
    Read pTM values.

    Typical input is list: [global_ptm, per_chain_ptm_1, per_chain_ptm_2, ...].

    Behavior:
      - If list-like and select == "mean", return the global value at index 0 (if present).
      - If list-like and select == "median", try index 1 (historical oddity) else compute median over numeric entries.
      - For "min","max","first","last" and other selects, use _summary_select on numeric values.
      - If input is a dict or scalar, fall back to _summary_select behavior.

    Note: This function preserves backward-compatibility with older code that expected
    ptm[0] for "mean" and ptm[1] for "median" when those indices exist.
    """
    if scores is None:
        return None

    if isinstance(scores, Mapping):
        # delegate: try to find explicit "ptm" keys if present
        if "ptm" in scores:
            return _summary_select(scores["ptm"], select=select)
        # else fall through to summary of mapping values
        return _summary_select(list(scores.values()), select=select)

    if isinstance(scores, (int, float, np.floating, np.integer)):
        return float(scores)

    try:
        seq = list(scores)
    except Exception:
        return None

    numeric = []
    for v in seq:
        try:
            fv = float(v)
            if not (np.isnan(fv) or fv == float("inf") or fv == float("-inf")):
                numeric.append(fv)
        except Exception:
            continue

    sel_lower = select.lower()
    if sel_lower == "mean":
        # keep backward compat: index 0 is global pTM if present
        if len(seq) > 0:
            try:
                return float(seq[0])
            except Exception:
                pass
        return _summary_select(numeric, select="mean")
    if sel_lower == "median":
        # legacy behavior: try index 1 if exists
        if len(seq) > 1:
            try:
                return float(seq[1])
            except Exception:
                pass
        return _summary_select(numeric, select="median")

    # other summaries
    return _summary_select(numeric, select=sel_lower)


# ---------- PAE helpers ----------
def _get_pae(scores: Any, select: str = "mean") -> Optional[float]:
    """
    Extract a PAE summary value from the pae summary array.

    Expected compact pae array format:
      pae[0] -> max_pae
      pae[1] -> mean_flattened
      pae[2] -> median_flattened
      pae[3] -> coverage (0..1)

    select options:
      - 'max' -> pae[0]
      - 'mean' -> pae[1]
      - 'median' -> pae[2]
      - 'coverage' or 'converage' -> pae[3]   (typo-compatible)
      - fallback to _summary_select for 'min','max','first','last', etc.
    """
    if scores is None:
        return None

    if isinstance(scores, Mapping):
        # if pae is stored as a mapping, try to find keys
        if "pae" in scores:
            return _summary_select(scores["pae"], select=select)
        # otherwise, summarise numeric values of mapping
        return _summary_select(list(scores.values()), select=select)

    if isinstance(scores, (int, float, np.floating, np.integer)):
        return float(scores)

    try:
        seq = list(scores)
    except Exception:
        return None

    # map selects to indices (allow common typo 'converage')
    idx_map = {
        "max": 0,
        "mean": 1,
        "median": 2,
        "coverage": 3,
        "converage": 3,
    }
    sel_lower = select.lower()
    if sel_lower in idx_map:
        idx = idx_map[sel_lower]
        if idx < len(seq):
            try:
                v = float(seq[idx])
                if not (np.isnan(v) or v == float("inf") or v == float("-inf")):
                    return v
            except Exception:
                pass
        # fallback to summary over numeric entries
        numeric = []
        for v in seq:
            try:
                fv = float(v)
                if not (np.isnan(fv) or fv == float("inf") or fv == float("-inf")):
                    numeric.append(fv)
            except Exception:
                continue
        # If the user requested 'mean' or 'median', do that; else pick mean
        return _summary_select(
            numeric,
            select=(
                "mean"
                if sel_lower not in ("min", "max", "first", "last", "median")
                else sel_lower
            ),
        )

    # generic fallbacks
    numeric = []
    for v in seq:
        try:
            fv = float(v)
            if not (np.isnan(fv) or fv == float("inf") or fv == float("-inf")):
                numeric.append(fv)
        except Exception:
            continue
    return _summary_select(numeric, select=sel_lower)


def get_data(
    data: Mapping[str, Mapping[str, MutableMapping[str, Any]]],
    rank: Optional[str] = None,
    plddt_select: str = "mean",
    ptm_select: str = "mean",
    pae_select: str = "mean",
    select_score: Optional[Iterable[str]] = (
        "prot_plddt",
        "pep_plddt",
        "PTM",
        "PAE",
        "iptm",
        "actifptm",
        "composite_ptm",
    ),
) -> List[Dict[str, Any]]:
    """
    Flatten `data` and attach derived numeric summaries.

    Behavior mirrors the original code:
      - if `rank` is None iterate over keys that start with "rank" (case-insensitive)
      - otherwise only use the explicit rank key where present
      - compute derived fields using the helper getters above

    :param data: mapping complex_id -> mapping rank_key -> record mapping
    :param rank: optional specific rank key to select (e.g. "rank001")
    :param plddt_select: selection passed to _get_plddt ("mean","median","peptide","interface",etc.)
    :param ptm_select: selection for _get_ptml
    :param pae_select: selection for _get_pae
    :param select_score: iterable of keys to keep; if None keep everything
    :return: list of flattened record dicts
    """
    keep_scores: Optional[Set[str]]
    if select_score is None:
        keep_scores = None
    else:
        keep_scores = set(select_score)

    records: List[Dict[str, Any]] = []

    for complex_id, ranks in data.items():
        # choose which rank entries to iterate
        if rank is None:
            rank_iterable = (
                (rk, rv)
                for rk, rv in ranks.items()
                if isinstance(rv, Mapping) and str(rk).lower().startswith("rank")
            )
        else:
            if rank in ranks and isinstance(ranks[rank], Mapping):
                rank_iterable = ((rank, ranks[rank]),)
            else:
                rank_iterable = ()

        for rank_key, rec in rank_iterable:
            # rec is expected to be a mapping (dict-like)
            record: Dict[str, Any] = dict(rec) if rec is not None else {}
            record["complex_id"] = complex_id
            record["rank"] = rank_key

            # derived numeric summaries
            plddt_val = record.get("plddt")
            record["prot_plddt"] = _get_plddt(plddt_val, select=plddt_select)
            record["pep_plddt"] = _get_pep_plddt(plddt_val)

            record["PTM"] = _get_ptml(record.get("ptm"), select=ptm_select)
            record["PAE"] = _get_pae(record.get("pae"), select=pae_select)

            if keep_scores is not None:
                filtered: Dict[str, Any] = {
                    "complex_id": record["complex_id"],
                    "rank": record["rank"],
                }
                for key in keep_scores:
                    if key in record:
                        filtered[key] = record[key]
                # ensure derived fields included if requested
                for dk in ("prot_plddt", "pep_plddt", "PTM", "PAE"):
                    if dk in keep_scores and dk in record:
                        filtered[dk] = record[dk]
                records.append(filtered)
            else:
                records.append(record)

    return records
