import pandas as pd
from pepq._get_score import get_data
from typing import Dict


def build_data(data: Dict, dockq: pd.DataFrame) -> pd.DataFrame:
    """
    Merge flattened summary data with DockQ ranking information.

    This function normalizes the reaction/complex summary entries returned
    by ``get_data(data)``, constructs a stable merge key for each (complex_id, rank)
    pair, standardizes the corresponding DockQ table, and performs a single,
    consistent inner join.

    :param data:
        Raw input object consumed by ``get_data()``.
        Expected to contain a list/dict/structure describing per-complex
        pLDDT, pTM, and PAE summaries.

    :param dockq: pandas.DataFrame
        DockQ results table containing at least columns:
        - ``id`` : Complex identifier matching ``complex_id``
        - ``Rank`` : Integer rank (e.g. 0, 1, 2, …)

    :raises KeyError:
        If required columns ``complex_id`` or ``rank`` are missing in the
        summary table, or if ``id`` / ``Rank`` are missing in the DockQ table.

    :returns: pandas.DataFrame
        A merged dataframe containing the original summary rows, enriched
        with the DockQ fields for matching ``complex_id``–``rank`` pairs.
        Only rows with matching merge keys are retained (inner join).

    """

    # ------------------------------------------------------------
    # 1) Flatten & normalize the summary arrays
    # ------------------------------------------------------------
    df = pd.DataFrame(get_data(data))
    # Standardise column names
    if "complex_id" not in df or "rank" not in df:
        raise KeyError("Input summaries must contain 'complex_id' and 'rank' columns.")

    # Merge key for summary table
    df["merge"] = df["complex_id"].astype(str) + "_" + df["rank"].astype(str)

    # ------------------------------------------------------------
    # 2) Prepare DockQ table
    # ------------------------------------------------------------
    required_cols = {"id", "Rank"}
    if not required_cols <= set(dockq.columns):
        raise KeyError(f"DockQ table must contain {required_cols}")

    dockq = dockq.copy()

    dockq["rank"] = "rank00" + dockq["Rank"].astype(str)
    dockq["merge"] = dockq["id"].astype(str) + "_" + dockq["rank"].astype(str)

    # ------------------------------------------------------------
    # 3) Perform a single clean merge
    # ------------------------------------------------------------
    merged = df.merge(dockq, on="merge", how="inner", suffixes=("", "_dockq"))
    merged["label"] = [0 if dq < 0.23 else 1 for dq in merged["GlobalDockQ"]]
    merged.rename(columns={"GlobalDockQ": "dockq"}, inplace=True)
    merged = merged[
        [
            "prot_plddt",
            "pep_plddt",
            "PTM",
            "PAE",
            "iptm",
            "composite_ptm",
            "actifptm",
            "label",
            "dockq",
        ]
    ]
    return merged
