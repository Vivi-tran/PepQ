import math
import pytest
import numpy as np

import pepq._get_score as gs


# ---------- _summary_select ----------
def test_summary_select_basic():
    assert gs._summary_select([1, 2, 3], "mean") == pytest.approx(2.0)
    assert gs._summary_select([1, 2, 3, 4], "median") == pytest.approx(2.5)
    assert gs._summary_select([5, 2, 9], "min") == 2.0
    assert gs._summary_select([5, 2, 9], "max") == 9.0
    assert gs._summary_select([7, 8], "first") == 7.0
    assert gs._summary_select([7, 8], "last") == 8.0
    assert gs._summary_select([], "mean") is None
    assert gs._summary_select(None, "mean") is None
    # non-numeric values -> None
    assert gs._summary_select(["a", object()], "mean") is None


# ---------- _get_plddt ----------
def test_get_plddt_scalar_and_sequence():
    seq = [75.3, 76.0, 60.2, 70.5]
    assert gs._get_plddt(seq, "mean") == pytest.approx(75.3)
    assert gs._get_plddt(seq, "median") == pytest.approx(76.0)
    assert gs._get_plddt(seq, "peptide") == pytest.approx(60.2)
    assert gs._get_plddt(seq, "interface") == pytest.approx(70.5)
    # scalar
    assert gs._get_plddt(42, "mean") == pytest.approx(42.0)


def test_get_plddt_dict_keys_priority_and_fallback():
    d = {"prot_plddt": [80, 82, 78]}
    assert gs._get_plddt(d, "mean") == pytest.approx(80.0)
    # if only 'plddt' present, _get_plddt will use it
    d2 = {"plddt": [60, 61, 59]}
    assert gs._get_plddt(d2, "median") == pytest.approx(60.0)


def test_get_plddt_missing_index_fallback_to_mean():
    short_seq = [75.0, 76.0]  # no peptide/index 2
    # asking for peptide -> fallback to mean of entries
    assert gs._get_plddt(short_seq, "peptide") == pytest.approx((75.0 + 76.0) / 2.0)


# ---------- _get_pep_plddt ----------
def test_get_pep_plddt_prefers_explicit_keys():
    d = {"pep_plddt": [60, 70, 55]}
    assert gs._get_pep_plddt(d) == pytest.approx((60 + 70 + 55) / 3.0)


def test_get_pep_plddt_from_compact_array_or_min_fallback():
    arr = [80.0, 81.0, 50.0]  # index 2 present
    assert gs._get_pep_plddt(arr) == pytest.approx(50.0)
    short = [90.0, 91.0]  # no index 2 -> min fallback
    assert gs._get_pep_plddt(short) == pytest.approx(90.0)


# ---------- _get_ptml ----------
def test_get_ptml_sequence_and_scalar():
    ptm = [0.68, 0.71, 0.55]
    # mean -> index 0 fallback
    assert gs._get_ptml(ptm, "mean") == pytest.approx(0.68)
    # median -> legacy try index 1
    assert gs._get_ptml(ptm, "median") == pytest.approx(0.71)
    # scalar
    assert gs._get_ptml(0.5, "mean") == pytest.approx(0.5)


# ---------- _get_pae ----------
def test_get_pae_compact_array_and_coverage_typo_ok():
    pae = [6.5, 3.2, 3.0, 0.25]
    assert gs._get_pae(pae, "max") == pytest.approx(6.5)
    assert gs._get_pae(pae, "mean") == pytest.approx(3.2)
    assert gs._get_pae(pae, "median") == pytest.approx(3.0)
    # accept typo 'converage'
    assert gs._get_pae(pae, "converage") == pytest.approx(0.25)
    # also accept 'coverage'
    assert gs._get_pae(pae, "coverage") == pytest.approx(0.25)


def test_get_pae_fallback_when_short_array():
    short = [1.0]  # only max maybe
    # asking for mean should fallback to summarizing numeric entries -> mean of [1.0]
    assert gs._get_pae(short, "mean") == pytest.approx(1.0)


# ---------- get_data integration tests ----------
def make_sample_data():
    return {
        "complex1": {
            "rank001": {
                "plddt": [75.3, 76.0, 60.2, 70.5],
                "ptm": [0.68, 0.71, 0.55],
                "pae": [6.5, 3.2, 3.0, 0.25],
                "iptm": 0.85,
                "actifptm": 0.9,
                "composite_ptm": 0.8,
            },
            "rank002": {
                "plddt": [60.0, 61.0, 50.0, 55.0],
                "ptm": [0.4],
                "pae": [5.0, 2.0, 2.5, 0.1],
            },
        },
        "complex2": {
            "rank001": {
                "plddt": [80.0, 80.5, 79.0, 81.0],
                "ptm": [0.9, 0.88],
                "pae": [4.0, 2.0, 2.1, 0.5],
            }
        },
    }


def test_get_data_select_score_filters_and_includes_derived():
    data = make_sample_data()
    # only keep prot_plddt and PTM
    out = gs.get_data(data, select_score=("prot_plddt", "PTM"))
    # there are 3 ranks across complexes in sample data (complex1: rank001+rank002, complex2: rank001)
    assert isinstance(out, list) and len(out) == 3
    for rec in out:
        assert "complex_id" in rec and "rank" in rec
        # only requested keys (plus complex_id/rank) should be present
        assert "prot_plddt" in rec
        assert "PTM" in rec
        # PAE should be omitted because not requested
        assert "PAE" not in rec


def test_get_data_rank_parameter_limits_to_single_rank():
    data = make_sample_data()
    out = gs.get_data(data, rank="rank001", select_score=None)  # keep everything
    # each complex contributes exactly one rank001 entry -> 2 complexes -> 2 records
    assert isinstance(out, list) and len(out) == 2
    # ensure derived fields present when select_score is None
    for rec in out:
        assert (
            "prot_plddt" in rec and "pep_plddt" in rec and "PTM" in rec and "PAE" in rec
        )


# ---------- edge cases ----------
def test_nan_and_inf_handling_in_summary_select():
    arr = [1.0, float("nan"), 3.0]
    # nan ignored by nanmean/nanmedian -> mean of [1,3] -> 2.0
    assert gs._summary_select(arr, "mean") == pytest.approx(2.0)
    # inf remains but _summary_select will include it -> but our _summary_select uses nan-aware functions
    arr2 = [1.0, float("inf")]
    # mean may be inf; check function doesn't crash
    m = gs._summary_select(arr2, "mean")
    assert m == pytest.approx(np.mean([1.0, float("inf")])) or math.isinf(m)


if __name__ == "__main__":
    pytest.main([__file__])
