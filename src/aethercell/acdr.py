"""End-to-end AC-DR drug-repurposing inference from the released TorchScript assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


def _decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


class StaticAssets:
    def __init__(self, path: Path):
        with h5py.File(path, "r") as handle:
            self.input_ids = torch.from_numpy(handle["drug_input_ids"][:]).long()
            self.attention = torch.from_numpy(handle["drug_attention_mask"][:]).long()
            self.drug_embeddings = torch.from_numpy(handle["node_emb/drug"][:]).float()
            self.disease_embeddings = torch.from_numpy(handle["node_emb/disease"][:]).float()
            self.gnn_scores = torch.from_numpy(handle["gnn_score/values"][:]).float()
            self.gnn_rows = {int(value): i for i, value in enumerate(json.loads(handle["gnn_score"].attrs["index"]))}
            self.gnn_cols = {str(value): i for i, value in enumerate(json.loads(handle["gnn_score"].attrs["columns"]))}
            self.disease_mapping = json.loads(handle.attrs["disease_mapping"])
            smiles_to_token = json.loads(handle.attrs["smiles2idx"])
            kg_to_db = json.loads(handle.attrs["drug_idx_mapping"])
            dbids = _decode(handle["drug_id_mapping/drugbank_id"][:])
            smiles = _decode(handle["drug_id_mapping/smiles"][:])
        db_to_smiles = dict(zip(dbids, smiles))
        db_to_kg = {str(dbid): int(float(index)) for index, ids in kg_to_db.items() for dbid in ids}
        disease_to_kg = {str(disease): int(float(index)) for index, ids in self.disease_mapping.items() for disease in ids}
        self.disease_to_kg = disease_to_kg
        candidates = []
        for dbid, gnn_col in self.gnn_cols.items():
            smiles_value = db_to_smiles.get(dbid)
            token_row, kg_row = smiles_to_token.get(smiles_value), db_to_kg.get(dbid)
            if smiles_value is not None and token_row is not None and kg_row is not None and kg_row < len(self.drug_embeddings):
                candidates.append((dbid, smiles_value, int(token_row), int(kg_row), int(gnn_col)))
        if not candidates:
            raise ValueError("no aligned AC-DR drug candidates were found in static_data.h5")
        self.candidates = candidates


def _disease_index(assets: StaticAssets, mondo_id: str) -> tuple[int, int]:
    key = str(int(float(mondo_id)))
    if key not in assets.disease_to_kg:
        raise KeyError(f"MONDO ID {mondo_id!r} is absent from disease_mapping")
    kg_index = assets.disease_to_kg[key]
    if kg_index not in assets.gnn_rows:
        raise KeyError(f"MONDO ID {mondo_id!r} has no GNN score row")
    return kg_index, assets.gnn_rows[kg_index]


@torch.inference_mode()
def predict(
    model_path: Path,
    static_h5: Path,
    mondo_id: str,
    disease_expression: np.ndarray | None,
    control_expression: np.ndarray | None,
    batch_size: int,
) -> pd.DataFrame:
    missing = [str(path) for path in (model_path, static_h5) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing AC-DR assets: " + ", ".join(missing) + "\n"
            "Run: python scripts/download_models.py --extract --output-dir models\n"
            "Source: https://huggingface.co/liwenyuan99/AetherCell"
        )
    if (disease_expression is None) != (control_expression is None):
        raise ValueError("disease and control expression must be supplied together")
    assets = StaticAssets(static_h5)
    kg_index, gnn_row = _disease_index(assets, mondo_id)
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    has_expression = disease_expression is not None
    expression_size = int(disease_expression.size) if has_expression else 10085
    disease = torch.from_numpy(disease_expression).float() if has_expression else torch.zeros(expression_size)
    control = torch.from_numpy(control_expression).float() if has_expression else torch.zeros(expression_size)
    rows = []
    for start in range(0, len(assets.candidates), batch_size):
        chunk = assets.candidates[start : start + batch_size]
        token_rows = torch.tensor([item[2] for item in chunk])
        kg_rows = torch.tensor([item[3] for item in chunk])
        gnn_cols = torch.tensor([item[4] for item in chunk])
        n = len(chunk)
        result = model(
            disease.unsqueeze(0).expand(n, -1),
            control.unsqueeze(0).expand(n, -1),
            assets.input_ids[token_rows],
            assets.attention[token_rows].float(),
            assets.disease_embeddings[kg_index].unsqueeze(0).expand(n, -1),
            assets.drug_embeddings[kg_rows],
            assets.gnn_scores[gnn_row, gnn_cols].unsqueeze(1),
        )
        scores, te_scores, kg_scores, weights = [value.detach().cpu().numpy() for value in result]
        for i, (dbid, smiles, token_row, _, _) in enumerate(chunk):
            rows.append({
                "mondo_id": str(mondo_id),
                "drugbank_id": dbid,
                "smiles": smiles,
                "drug_idx": token_row,
                "moe_score": float(scores[i]),
                "te_score": float(te_scores[i]),
                "kg_score": float(kg_scores[i]),
                "te_weight": float(weights[i, 0]),
                "input_status": "Transcriptome_and_KG" if has_expression else "KG_Only",
            })
    score = "moe_score" if has_expression else "kg_score"
    return pd.DataFrame(rows).sort_values(score, ascending=False).reset_index(drop=True)


def _load_expression(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] != 1:
            raise ValueError("expression CSV must contain exactly one numeric expression column")
        value = numeric.iloc[:, 0].to_numpy()
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size != 10085:
        raise ValueError(f"AC-DR expression must contain 10,085 genes; found {value.size}")
    if not np.isfinite(value).all():
        raise ValueError("expression contains NaN or infinite values")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--static-h5", type=Path, required=True)
    parser.add_argument("--mondo-id", required=True)
    parser.add_argument("--disease-expression", type=Path)
    parser.add_argument("--control-expression", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    disease = _load_expression(args.disease_expression)
    control = _load_expression(args.control_expression)
    result = predict(args.model, args.static_h5, args.mondo_id, disease, control, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.head(args.top_n).to_csv(args.output, index=False)
    print(f"wrote top {min(args.top_n, len(result))} AC-DR candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
