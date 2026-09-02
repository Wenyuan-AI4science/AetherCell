import numpy as np
import pandas as pd

from aethercell.benchmarks import run_ac_dr, run_ac_rp, run_cdx, run_synergy, run_tcga


def test_all_downstream_runners():
    rp, grouped = run_ac_rp(pd.DataFrame({"data_type": ["a"] * 4, "pred": [1, 2, 3, 4], "true": [1, 2, 3, 4]}))
    assert np.isclose(rp["pearson"], 1.0) and len(grouped) == 1
    synergy, _ = run_synergy(pd.DataFrame({"y_prob_pred": [0.1, 0.9, 0.2, 0.8], "label_bin": [0, 1, 0, 1]}))
    assert synergy["auroc"] == 1.0
    tcga, _ = run_tcga(pd.DataFrame({"score": [0.1, 0.9, 0.2, 0.8], "label": [0, 1, 0, 1]}))
    assert tcga["auroc"] == 1.0
    cdx, ranked = run_cdx(pd.DataFrame({"ic50": [1, 1], "ic50_post_sh": [0, 2], "ic50_diff": [-1, 1]}), 1)
    assert cdx["max_difference_consistency_error"] == 0.0 and ranked.iloc[0]["ic50_diff"] == -1
    acdr, ranked = run_ac_dr(pd.DataFrame({
        "drugbank_id": ["D1", "D2"], "moe_score": [0.2, 0.8], "te_score": [0.1, 0.7],
        "kg_score": [0.3, 0.4], "te_weight": [0.5, 0.5], "input_status": ["Transcriptome_and_KG"] * 2,
    }), 1)
    assert acdr["top_score"] == 0.8 and ranked.iloc[0]["drugbank_id"] == "D2"
