from pathlib import Path

import numpy as np

from aethercell.data import PerturbationNPZDataset, ShRNAPerturbationNPZDataset
from aethercell.train import main


def test_npz_rejects_missing_arrays(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez(path, control=np.zeros((2, 3), dtype="float32"))
    try:
        PerturbationNPZDataset(path)
    except ValueError as error:
        assert "missing required arrays" in str(error)
    else:
        raise AssertionError("invalid bundle was accepted")


def test_training_smoke(tmp_path: Path):
    output = tmp_path / "training"
    assert main(["--smoke-test", "--output-dir", str(output)]) == 0
    assert (output / "best_model.pt").is_file()
    assert (output / "training_log.csv").is_file()


def test_shrna_npz_dataset(tmp_path: Path):
    path = tmp_path / "shrna.npz"
    np.savez(
        path,
        control=np.zeros((2, 978), dtype="float32"),
        rna=np.zeros((2, 19326), dtype="float32"),
        sh_ppi=np.zeros((2, 256), dtype="float32"),
        sh_seq=np.zeros((2, 1152), dtype="float32"),
        sample_id=np.asarray(["S1", "S2"]),
        pert_id=np.asarray(["ENSG1", "ENSG2"]),
        cell_id=np.asarray(["A", "B"]),
    )
    dataset = ShRNAPerturbationNPZDataset(path)
    assert len(dataset) == 2
    assert tuple(dataset[0]["sh_ppi"].shape) == (256,)
    assert tuple(dataset[0]["sh_seq"].shape) == (1152,)
