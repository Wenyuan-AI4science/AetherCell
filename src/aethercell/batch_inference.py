"""Configurable batch inference for portable AetherCell NPZ bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import (
    PerturbationNPZDataset,
    ShRNAPerturbationNPZDataset,
    ZenodoDrugDataset,
    ZenodoShRNADataset,
)
from .model_io import build_drug_model, build_sh_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["drug", "shrna"], default="drug")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="portable NPZ bundle")
    source.add_argument("--zenodo-dir", type=Path, help="extracted data4zendo directory")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy-src",
        type=Path,
        help="released model source directory; defaults to the directory containing --lincs-vae",
    )
    parser.add_argument("--lincs-vae", type=Path, required=True)
    parser.add_argument("--rna-vae", type=Path, required=True)
    parser.add_argument("--molformer-dir", type=Path, help="required for --mode drug")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


@torch.inference_mode()
def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device, mode: str = "drug"):
    model.eval()
    expressions, deltas, latent_deltas, rows = [], [], [], []
    offset = 0
    for batch in loader:
        control = batch["control"].to(device).float()
        if mode == "drug":
            prediction, _, delta_z, _ = model(
                batch["rna"].to(device).float(),
                control,
                batch["input_ids"].to(device).long(),
                batch["attention_mask"].to(device).long(),
            )
        else:
            prediction, _, delta_z, _ = model(
                batch["rna"].to(device).float(),
                control,
                batch["sh_ppi"].to(device).float(),
                batch["sh_seq"].to(device).float(),
            )
        expressions.append(prediction.cpu().numpy())
        deltas.append((prediction - control).cpu().numpy())
        latent_deltas.append(delta_z.cpu().numpy())
        for i in range(len(batch["sample_id"])):
            row = {
                "row": offset + i,
                "sample_id": batch["sample_id"][i],
                "cell_id": batch["cell_id"][i],
                "pert_id": batch["pert_id"][i],
            }
            for key in ("control_id", "det_plate", "pert_type"):
                if key in batch:
                    row[key] = batch[key][i]
            rows.append(row)
        offset += len(batch["sample_id"])
    if not rows:
        raise ValueError("input dataset is empty")
    return np.concatenate(expressions), np.concatenate(deltas), np.concatenate(latent_deltas), pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.mode == "drug":
        dataset = (
            ZenodoDrugDataset(args.zenodo_dir, args.split)
            if args.zenodo_dir is not None
            else PerturbationNPZDataset(args.data, require_labels=False)
        )
    else:
        dataset = (
            ZenodoShRNADataset(args.zenodo_dir, args.split)
            if args.zenodo_dir is not None
            else ShRNAPerturbationNPZDataset(args.data)
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.mode == "drug":
        if args.molformer_dir is None:
            raise SystemExit("--molformer-dir is required for --mode drug; run aethercell-doctor for asset help")
        model = build_drug_model(
            legacy_src=args.legacy_src or args.lincs_vae.parent,
            lincs_vae_checkpoint=args.lincs_vae,
            rna_vae_checkpoint=args.rna_vae,
            molformer_dir=args.molformer_dir,
            predictor_checkpoint=args.checkpoint,
            device=device,
        )
    else:
        model = build_sh_model(
            legacy_src=args.legacy_src or args.lincs_vae.parent,
            lincs_vae_checkpoint=args.lincs_vae,
            rna_vae_checkpoint=args.rna_vae,
            predictor_checkpoint=args.checkpoint,
            device=device,
        )
    expression, delta, delta_z, metadata = infer(model, loader, device, args.mode)
    np.save(args.output_dir / "predicted_expression.npy", expression)
    np.save(args.output_dir / "predicted_delta.npy", delta)
    np.save(args.output_dir / "predicted_delta_z.npy", delta_z)
    metadata.to_csv(args.output_dir / "metadata.csv", index=False)
    print(f"wrote {len(metadata)} predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
