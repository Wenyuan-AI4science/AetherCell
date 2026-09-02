"""Portable NPZ datasets for reviewer-facing training and inference."""

from __future__ import annotations

from pathlib import Path
import io
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_TRAIN_KEYS = ("control", "label", "rna", "input_ids", "attention_mask", "cell_id")
REQUIRED_INFERENCE_KEYS = ("control", "rna", "input_ids", "attention_mask")
REQUIRED_SH_INFERENCE_KEYS = ("control", "rna", "sh_ppi", "sh_seq")


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"global object {module}.{name} is forbidden in an index mapping")


def safe_load_mapping(path: str | Path) -> dict:
    """Load a primitive dict pickle while rejecting all executable/global objects.

    The pinned Zenodo release stores three string-to-integer maps as pickle
    files.  A normal ``pickle.load`` can execute code; this restricted loader
    only accepts pickle opcodes that construct primitive containers and values.
    """
    raw = Path(path).read_bytes()
    value = _RestrictedUnpickler(io.BytesIO(raw)).load()
    if not isinstance(value, dict):
        raise ValueError(f"expected a dictionary in {path}")
    result = {}
    for key, index in value.items():
        if not isinstance(key, (str, int)) or not isinstance(index, int):
            raise ValueError(f"mapping {path} contains non-primitive key/value types")
        result[str(key)] = index
    return result


class PerturbationNPZDataset(Dataset):
    """Load a compact, pickle-free AetherCell matrix bundle."""

    def __init__(self, path: str | Path, require_labels: bool = True):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        bundle = np.load(self.path, allow_pickle=False, mmap_mode="r")
        required = REQUIRED_TRAIN_KEYS if require_labels else REQUIRED_INFERENCE_KEYS
        missing = sorted(set(required) - set(bundle.files))
        if missing:
            raise ValueError(f"{self.path} is missing required arrays: {missing}")
        lengths = {key: len(bundle[key]) for key in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"arrays have inconsistent first dimensions: {lengths}")
        self.bundle = bundle
        self.require_labels = require_labels
        self.length = next(iter(lengths.values()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        result: dict[str, object] = {
            "control": torch.from_numpy(np.asarray(self.bundle["control"][index], dtype=np.float32)),
            "rna": torch.from_numpy(np.asarray(self.bundle["rna"][index], dtype=np.float32)),
            "input_ids": torch.from_numpy(np.asarray(self.bundle["input_ids"][index], dtype=np.int64)),
            "attention_mask": torch.from_numpy(np.asarray(self.bundle["attention_mask"][index], dtype=np.int64)),
            "sample_id": str(self.bundle["sample_id"][index]) if "sample_id" in self.bundle else str(index),
            "pert_id": str(self.bundle["pert_id"][index]) if "pert_id" in self.bundle else "",
            "cell_id": str(self.bundle["cell_id"][index]) if "cell_id" in self.bundle else "",
        }
        if self.require_labels:
            result["label"] = torch.from_numpy(np.asarray(self.bundle["label"][index], dtype=np.float32))
        return result


class ShRNAPerturbationNPZDataset(Dataset):
    """Pickle-free custom shRNA inference bundle."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.bundle = np.load(self.path, allow_pickle=False, mmap_mode="r")
        missing = sorted(set(REQUIRED_SH_INFERENCE_KEYS) - set(self.bundle.files))
        if missing:
            raise ValueError(f"{self.path} is missing required shRNA arrays: {missing}")
        lengths = {key: len(self.bundle[key]) for key in REQUIRED_SH_INFERENCE_KEYS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"arrays have inconsistent first dimensions: {lengths}")
        self.length = next(iter(lengths.values()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "control": torch.from_numpy(np.asarray(self.bundle["control"][index], dtype=np.float32)),
            "rna": torch.from_numpy(np.asarray(self.bundle["rna"][index], dtype=np.float32)),
            "sh_ppi": torch.from_numpy(np.asarray(self.bundle["sh_ppi"][index], dtype=np.float32)),
            "sh_seq": torch.from_numpy(np.asarray(self.bundle["sh_seq"][index], dtype=np.float32)),
            "sample_id": str(self.bundle["sample_id"][index]) if "sample_id" in self.bundle else str(index),
            "pert_id": str(self.bundle["pert_id"][index]) if "pert_id" in self.bundle else "",
            "cell_id": str(self.bundle["cell_id"][index]) if "cell_id" in self.bundle else "",
        }


class ZenodoDrugDataset(Dataset):
    """Memory-efficient reader for the pinned ``data4zendo/compound_perturbed`` layout."""

    def __init__(self, root: str | Path, split: str):
        import pandas as pd

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        root = Path(root)
        compound = root / "compound_perturbed"
        if not compound.is_dir() and (root / "data4zendo" / "compound_perturbed").is_dir():
            compound = root / "data4zendo" / "compound_perturbed"
        required = [
            f"df_{split}_drug.csv", "L1000_exp.npy", "L1000_ctrl.npy", "exp_idx_map.pkl",
            "ctrl_idx_map.pkl", "RNAseq.parquet", "drug_input_ids.npy", "drug_attention_mask.npy",
            "drug_idx_map.pkl",
        ]
        missing = [name for name in required if not (compound / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Zenodo compound_perturbed directory is missing: {missing}\n"
                "Run: python scripts/download_data.py --extract --output-dir data/zenodo\n"
                "Source: https://doi.org/10.5281/zenodo.18295255"
            )
        self.meta = pd.read_csv(compound / f"df_{split}_drug.csv")
        required_columns = {"sample_id", "representative_control", "pert_id", "cell_iname"}
        absent = sorted(required_columns - set(self.meta.columns))
        if absent:
            raise ValueError(f"metadata is missing columns: {absent}")
        self.expression = np.load(compound / "L1000_exp.npy", mmap_mode="r")
        self.control = np.load(compound / "L1000_ctrl.npy", mmap_mode="r")
        self.expression_index = safe_load_mapping(compound / "exp_idx_map.pkl")
        self.control_index = safe_load_mapping(compound / "ctrl_idx_map.pkl")
        self.drug_index = safe_load_mapping(compound / "drug_idx_map.pkl")
        self.input_ids = np.load(compound / "drug_input_ids.npy", mmap_mode="r")
        self.attention = np.load(compound / "drug_attention_mask.npy", mmap_mode="r")
        rna = pd.read_parquet(compound / "RNAseq.parquet")
        self.rna = rna.to_numpy(dtype=np.float32, copy=True)
        self.cell_index = {str(cell): index for index, cell in enumerate(rna.columns)}

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.meta.iloc[index]
        sample_id = str(row["sample_id"])
        control_id = str(row["representative_control"])
        pert_id = str(row["pert_id"])
        cell_id = str(row["cell_iname"])
        try:
            expression_row = self.expression_index[sample_id]
            control_row = self.control_index[control_id]
            drug_row = self.drug_index[pert_id]
            cell_column = self.cell_index[cell_id]
        except KeyError as error:
            raise KeyError(f"row {index} references an absent sample/control/drug/cell: {error}") from error
        return {
            "control": torch.from_numpy(np.array(self.control[control_row], dtype=np.float32, copy=True)),
            "label": torch.from_numpy(np.array(self.expression[expression_row], dtype=np.float32, copy=True)),
            "rna": torch.from_numpy(np.asarray(self.rna[:, cell_column], dtype=np.float32)),
            "input_ids": torch.from_numpy(np.array(self.input_ids[drug_row], dtype=np.int64, copy=True)),
            "attention_mask": torch.from_numpy(np.array(self.attention[drug_row], dtype=np.int64, copy=True)),
            "sample_id": sample_id,
            "control_id": control_id,
            "pert_id": pert_id,
            "cell_id": cell_id,
            "det_plate": str(row["det_plate"]) if "det_plate" in row.index else "",
            "pert_type": str(row["pert_type"]) if "pert_type" in row.index else "",
        }


class ZenodoShRNADataset(Dataset):
    """Memory-efficient reader for the pinned ``data4zendo/shRNA_perturbed`` layout."""

    def __init__(self, root: str | Path, split: str):
        import pandas as pd

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        root = Path(root)
        folder = root / "shRNA_perturbed"
        if not folder.is_dir() and (root / "data4zendo" / "shRNA_perturbed").is_dir():
            folder = root / "data4zendo" / "shRNA_perturbed"
        required = [
            f"sh_meta_s1_{split}.csv", "L1000_exp.npy", "L1000_ctrl.npy", "exp_idx_map.pkl",
            "ctrl_idx_map.pkl", "RNAseq.parquet", "ensg_PPI_emb.csv", "emb_tokens_first_all.npy",
            "id2idx_ensg_seq2_all.pkl",
        ]
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Zenodo shRNA_perturbed directory is missing: {missing}\n"
                "Run: python scripts/download_data.py --extract --output-dir data/zenodo"
            )
        self.meta = pd.read_csv(folder / f"sh_meta_s1_{split}.csv", low_memory=False)
        self.expression = np.load(folder / "L1000_exp.npy", mmap_mode="r")
        self.control = np.load(folder / "L1000_ctrl.npy", mmap_mode="r")
        self.expression_index = safe_load_mapping(folder / "exp_idx_map.pkl")
        self.control_index = safe_load_mapping(folder / "ctrl_idx_map.pkl")
        self.sequence_index = safe_load_mapping(folder / "id2idx_ensg_seq2_all.pkl")
        self.sequence = np.load(folder / "emb_tokens_first_all.npy", mmap_mode="r")
        rna = pd.read_parquet(folder / "RNAseq.parquet")
        self.rna = rna.to_numpy(dtype=np.float32, copy=True)
        self.cell_index = {str(cell): index for index, cell in enumerate(rna.columns)}
        ppi = pd.read_csv(folder / "ensg_PPI_emb.csv")
        id_column = "gene" if "gene" in ppi.columns else ("pert_id" if "pert_id" in ppi.columns else ppi.columns[0])
        numeric = [column for column in ppi.columns if column != id_column and pd.api.types.is_numeric_dtype(ppi[column])]
        self.ppi = ppi[numeric].to_numpy(dtype=np.float32, copy=True)
        self.ppi_index = {str(value): index for index, value in enumerate(ppi[id_column])}
        if self.ppi.shape[1] != 256 or self.sequence.shape[1] != 1152:
            raise ValueError(f"unexpected shRNA embedding dimensions: PPI={self.ppi.shape}, sequence={self.sequence.shape}")

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.meta.iloc[index]
        sample_id = str(row["sample_id"])
        control_id = str(row["representative_control"])
        cell_id = str(row["cell_iname"])
        pert_column = "gene_ensg" if "gene_ensg" in row.index else "pert_id"
        pert_id = str(row[pert_column])
        try:
            expression_row = self.expression_index[sample_id]
            control_row = self.control_index[control_id]
            cell_column = self.cell_index[cell_id]
            ppi_row = self.ppi_index[pert_id]
            sequence_row = self.sequence_index[pert_id]
        except KeyError as error:
            raise KeyError(f"row {index} references an absent sample/control/cell/gene: {error}") from error
        return {
            "control": torch.from_numpy(np.array(self.control[control_row], dtype=np.float32, copy=True)),
            "label": torch.from_numpy(np.array(self.expression[expression_row], dtype=np.float32, copy=True)),
            "rna": torch.from_numpy(np.asarray(self.rna[:, cell_column], dtype=np.float32)),
            "sh_ppi": torch.from_numpy(np.asarray(self.ppi[ppi_row], dtype=np.float32)),
            "sh_seq": torch.from_numpy(np.array(self.sequence[sequence_row], dtype=np.float32, copy=True)),
            "sample_id": sample_id,
            "control_id": control_id,
            "pert_id": pert_id,
            "cell_id": cell_id,
            "det_plate": str(row["det_plate"]) if "det_plate" in row.index else "",
            "pert_type": str(row["pert_type"]) if "pert_type" in row.index else "",
        }
