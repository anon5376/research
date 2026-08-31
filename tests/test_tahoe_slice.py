from __future__ import annotations

import pandas as pd

from tahoe.de import class_means, iter_signature_rows, signature_by_genotype
from tahoe.genotype import ras_raf_class, summarize_drivers
from tahoe.slice import parse_treatment


def _raw_drivers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_name": "A375",
                "Cell_ID_DepMap": "ACH-1",
                "Cell_ID_Cellosaur": "CVCL_A375",
                "Organ": "Skin",
                "Driver_Gene_Symbol": "BRAF",
                "Driver_ProtEffect_or_CdnaEffect": "p.V600E",
                "Driver_Mech_InferDM": "GoF",
            },
            {
                "cell_name": "A375",
                "Cell_ID_DepMap": "ACH-1",
                "Cell_ID_Cellosaur": "CVCL_A375",
                "Organ": "Skin",
                "Driver_Gene_Symbol": "CDKN2A",
                "Driver_ProtEffect_or_CdnaEffect": "p.M53fs",
                "Driver_Mech_InferDM": "LoF",
            },
            {
                "cell_name": "A549",
                "Cell_ID_DepMap": "ACH-2",
                "Cell_ID_Cellosaur": "CVCL_A549",
                "Organ": "Lung",
                "Driver_Gene_Symbol": "KRAS",
                "Driver_ProtEffect_or_CdnaEffect": "p.G12S",
                "Driver_Mech_InferDM": "GoF",
            },
            {
                "cell_name": "HCC827",
                "Cell_ID_DepMap": "ACH-3",
                "Cell_ID_Cellosaur": "CVCL_WT",
                "Organ": "Lung",
                "Driver_Gene_Symbol": "EGFR",
                "Driver_ProtEffect_or_CdnaEffect": "p.E746_A750del",
                "Driver_Mech_InferDM": "GoF",
            },
            {
                "cell_name": "HCC827",
                "Cell_ID_DepMap": "ACH-3",
                "Cell_ID_Cellosaur": "CVCL_WT",
                "Organ": "Lung",
                "Driver_Gene_Symbol": "TP53",
                "Driver_ProtEffect_or_CdnaEffect": "p.V218del",
                "Driver_Mech_InferDM": "LoF",
            },
        ]
    )


def test_summarize_drivers_and_classes() -> None:
    drivers = summarize_drivers(_raw_drivers())
    by_id = drivers.set_index("cell_id")
    assert by_id.loc["CVCL_A375", "braf_v600e"]
    assert by_id.loc["CVCL_A375", "ras_raf_class"] == "BRAF_V600E"
    assert by_id.loc["CVCL_A549", "kras_g12"]
    assert by_id.loc["CVCL_A549", "ras_raf_class"] == "KRAS_GOF"
    assert by_id.loc["CVCL_WT", "ras_raf_class"] == "RAS_RAF_WT"
    assert by_id.loc["CVCL_WT", "tp53_lof"]


def test_ras_raf_class_other_when_both() -> None:
    row = pd.Series(
        {"braf_v600e": True, "has_braf": True, "has_kras": True, "kras_g12": True}
    )
    assert ras_raf_class(row) == "other"


def test_parse_treatment() -> None:
    name, conc = parse_treatment("[('Dabrafenib', 0.5, 'uM')]")
    assert name == "Dabrafenib"
    assert conc == 0.5


def test_signature_by_genotype_recovers_braf_effect() -> None:
    drivers = summarize_drivers(_raw_drivers())
    records = [
        {
            "Cell_ID_Cellosaur": "CVCL_A375",
            "drug": "Dabrafenib",
            "concentration": 0.5,
            "gene_name": "DUSP6",
            "log2FoldChange": -2.0,
        },
        {
            "Cell_ID_Cellosaur": "CVCL_A375",
            "drug": "Dabrafenib",
            "concentration": 0.5,
            "gene_name": "SPRY2",
            "log2FoldChange": -1.0,
        },
        {
            "Cell_ID_Cellosaur": "CVCL_A549",
            "drug": "Dabrafenib",
            "concentration": 0.5,
            "gene_name": "DUSP6",
            "log2FoldChange": -0.1,
        },
        {
            "Cell_ID_Cellosaur": "CVCL_A549",
            "drug": "Dabrafenib",
            "concentration": 0.5,
            "gene_name": "SPRY2",
            "log2FoldChange": 0.1,
        },
        {
            "Cell_ID_Cellosaur": "CVCL_A549",
            "drug": "DMSO_TF",
            "concentration": 0.0,
            "gene_name": "DUSP6",
            "log2FoldChange": 5.0,
        },
    ]
    rows = list(iter_signature_rows(records, drugs=("Dabrafenib",), concentration=0.5))
    de = pd.DataFrame(rows)
    scored = signature_by_genotype(de, drivers)
    summary = class_means(scored)
    braf = summary.loc[summary["ras_raf_class"] == "BRAF_V600E", "mean_mapk_log2fc"].iloc[0]
    kras = summary.loc[summary["ras_raf_class"] == "KRAS_GOF", "mean_mapk_log2fc"].iloc[0]
    assert braf < kras
    assert braf == -1.5
