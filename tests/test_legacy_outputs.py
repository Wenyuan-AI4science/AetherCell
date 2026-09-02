from pathlib import Path

import numpy as np
import pandas as pd

from aethercell.legacy_outputs import write_legacy_delta_z, write_legacy_expression


def test_legacy_output_contract_is_additive(tmp_path: Path):
    prediction = np.asarray([[1.0, 3.0], [2.0, 7.0]], dtype=np.float32)
    delta = np.asarray([[0.5, 1.0], [-1.0, 2.0]], dtype=np.float32)
    latent = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.save(tmp_path / "predicted_expression.npy", prediction)
    np.save(tmp_path / "predicted_delta.npy", delta)
    np.save(tmp_path / "predicted_delta_z.npy", latent)
    pd.DataFrame(
        {
            "row": [0, 1],
            "sample_id": ["S1", "S2"],
            "cell_id": ["A", "B"],
            "pert_id": ["D1", "D2"],
            "control_id": ["C1", "C2"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)

    write_legacy_delta_z(tmp_path, chunk_size=1)
    write_legacy_expression(tmp_path, chunk_size=1)

    np.testing.assert_array_equal(np.load(tmp_path / "delta_z_predictions.npy"), latent)
    np.testing.assert_array_equal(np.load(tmp_path / "perturbed_expression.npy"), prediction)
    np.testing.assert_array_equal(np.load(tmp_path / "delta_expression.npy"), delta)
    np.testing.assert_allclose(np.load(tmp_path / "control_expression.npy"), prediction - delta)
    assert list(pd.read_csv(tmp_path / "delta_z_predictions.csv").columns[-4:]) == [
        "delta_z_0",
        "delta_z_1",
        "delta_z_2",
        "delta_z_3",
    ]
    summary = pd.read_csv(tmp_path / "summary_statistics.csv").iloc[0]
    assert summary["n_samples"] == 2
    assert summary["n_genes"] == 2
    assert np.isclose(summary["delta_abs_mean"], np.abs(delta).mean())
