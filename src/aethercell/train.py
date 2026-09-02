"""Single-GPU/CPU AetherCell training entry point with specificity-aware loss."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .data import PerturbationNPZDataset, ZenodoDrugDataset
from .losses import AetherCellLoss, AetherCellValidationLoss, LossWeights, SpecificityReference
from .model_io import build_drug_model, trainable_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encoder(model: nn.Module) -> nn.Module:
    raw = model.module if hasattr(model, "module") else model
    return raw.L_encoder


def _keep_frozen_backbone_in_eval(model: nn.Module) -> None:
    raw = model.module if hasattr(model, "module") else model
    for name in ("L_encoder", "L_decoder", "RNAencoder"):
        module = getattr(raw, name, None)
        if module is not None:
            module.eval()


@torch.no_grad()
def latent_reference(model: nn.Module, loader: DataLoader, device: torch.device) -> SpecificityReference:
    encoder = _encoder(model)
    encoder.eval()
    batches = []
    for batch in loader:
        control = batch["control"].to(device).float()
        target = batch["label"].to(device).float()
        _, control_mu, _ = encoder(control)
        _, target_mu, _ = encoder(target)
        batches.append(((target_mu - control_mu).cpu(), list(batch["cell_id"])))
    return SpecificityReference.from_batches(batches)


def build_objectives(
    reference: SpecificityReference,
    weights: LossWeights,
    top_k: int,
    specificity_margin: float,
) -> tuple[AetherCellLoss, AetherCellValidationLoss]:
    """Return the specificity-aware training and specificity-free validation objectives."""
    training = AetherCellLoss(
        reference,
        weights,
        top_k,
        specificity_margin,
        leave_one_out=True,
    )
    validation = AetherCellValidationLoss(weights, top_k)
    return training, validation


def _forward_loss(
    model: nn.Module,
    objective: AetherCellLoss | AetherCellValidationLoss,
    batch: dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    control = batch["control"].to(device).float()
    target = batch["label"].to(device).float()
    rna = batch["rna"].to(device).float()
    input_ids = batch["input_ids"].to(device).long()
    attention = batch["attention_mask"].to(device).long()
    prediction, _, delta_z_pred, z_base = model(rna, control, input_ids, attention)
    with torch.no_grad():
        _, target_mu, _ = _encoder(model)(target)
        delta_z_true = (target_mu - z_base).detach()
    arguments = dict(
        prediction=prediction,
        target=target,
        control=control,
        delta_z_pred=delta_z_pred,
        delta_z_true=delta_z_true,
    )
    if isinstance(objective, AetherCellLoss):
        arguments["cell_ids"] = list(batch["cell_id"])
    return objective(**arguments)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    objective: AetherCellLoss | AetherCellValidationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    _keep_frozen_backbone_in_eval(model)
    totals: dict[str, float] = {}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch_size = int(batch["control"].shape[0])
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, parts = _forward_loss(model, objective, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training objective: {float(loss.detach())}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
            values = {"loss": loss, **parts}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            samples += batch_size
    return {name: total / max(samples, 1) for name, total in totals.items()}


class TinyPerturbationModel(nn.Module):
    """Fast architecture-compatible model used only by ``--smoke-test``."""

    class Encoder(nn.Module):
        def __init__(self, genes: int, latent: int):
            super().__init__()
            self.linear = nn.Linear(genes, latent)

        def forward(self, x):
            mu = self.linear(x)
            return mu, mu, torch.zeros_like(mu)

    def __init__(self, genes: int, latent: int = 16, vocab: int = 4096):
        super().__init__()
        self.L_encoder = self.Encoder(genes, latent)
        self.L_decoder = nn.Linear(latent, genes)
        self.token = nn.Embedding(vocab, latent)
        self.delta = nn.Sequential(nn.Linear(latent * 2, latent), nn.Tanh())

    def forward(self, rna, control, input_ids, attention_mask):
        del rna
        _, z_base, _ = self.L_encoder(control)
        mask = attention_mask.unsqueeze(-1).float()
        token = self.token(input_ids.remainder(self.token.num_embeddings))
        token = (token * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        delta_z = self.delta(torch.cat([z_base, token], dim=1))
        z_pred = z_base + delta_z
        return self.L_decoder(z_pred), z_pred, delta_z, z_base


def _smoke_bundle(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, genes, rna_genes, tokens = 16, 32, 48, 12
    control = rng.normal(size=(n, genes)).astype("float32")
    label = control + rng.normal(scale=0.3, size=(n, genes)).astype("float32")
    np.savez_compressed(
        path,
        control=control,
        label=label,
        rna=rng.normal(size=(n, rna_genes)).astype("float32"),
        input_ids=rng.integers(0, 100, size=(n, tokens), dtype=np.int64),
        attention_mask=np.ones((n, tokens), dtype=np.int64),
        cell_id=np.asarray([f"CELL_{i % 4}" for i in range(n)]),
        sample_id=np.asarray([f"S{i}" for i in range(n)]),
        pert_id=np.asarray([f"D{i % 3}" for i in range(n)]),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="pickle-free NPZ training bundle")
    parser.add_argument("--zenodo-dir", type=Path, help="extracted data4zendo directory; uses the official train/test split")
    parser.add_argument("--output-dir", type=Path, default=Path("results/training"))
    parser.add_argument(
        "--legacy-src",
        type=Path,
        help="released model source directory; defaults to the directory containing --lincs-vae",
    )
    parser.add_argument("--lincs-vae", type=Path)
    parser.add_argument("--rna-vae", type=Path)
    parser.add_argument("--molformer-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--specificity-margin", type=float, default=0.1)
    parser.add_argument("--lambda-reconstruction", type=float, default=0.5)
    parser.add_argument("--lambda-directional", type=float, default=2.0)
    parser.add_argument("--lambda-weighted-mse", type=float, default=0.3)
    parser.add_argument("--lambda-latent-alignment", type=float, default=0.2)
    parser.add_argument("--lambda-specificity", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--smoke-test", action="store_true", help="run a deterministic 2-epoch CPU integration test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        args.device, args.epochs, args.batch_size = "cpu", 2, 4
        args.data = args.output_dir / "smoke_training.npz"
        _smoke_bundle(args.data, args.seed)
    if args.data is None and args.zenodo_dir is None:
        raise SystemExit("provide --data or --zenodo-dir unless --smoke-test is used")
    if args.data is not None and args.zenodo_dir is not None:
        raise SystemExit("--data and --zenodo-dir are mutually exclusive")
    device = torch.device(args.device)
    if args.zenodo_dir is not None:
        train_set = ZenodoDrugDataset(args.zenodo_dir, "train")
        val_set = ZenodoDrugDataset(args.zenodo_dir, "test")
        dataset = train_set
    else:
        dataset = PerturbationNPZDataset(args.data, require_labels=True)
        val_n = max(1, round(len(dataset) * args.validation_fraction))
        train_n = len(dataset) - val_n
        if train_n < 2:
            raise ValueError("training split must contain at least two samples")
        train_set, val_set = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    reference_loader = DataLoader(train_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.smoke_test:
        genes = int(dataset.bundle["control"].shape[1])
        model = TinyPerturbationModel(genes).to(device)
    else:
        missing = [name for name in ("lincs_vae", "rna_vae", "molformer_dir") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"missing real-model arguments: {', '.join('--' + x.replace('_', '-') for x in missing)}")
        model = build_drug_model(
            legacy_src=args.legacy_src or args.lincs_vae.parent,
            lincs_vae_checkpoint=args.lincs_vae,
            rna_vae_checkpoint=args.rna_vae,
            molformer_dir=args.molformer_dir,
            device=device,
            predictor_checkpoint=args.resume,
        )
    reference = latent_reference(model, reference_loader, device)
    weights = LossWeights(
        args.lambda_reconstruction,
        args.lambda_directional,
        args.lambda_weighted_mse,
        args.lambda_latent_alignment,
        args.lambda_specificity,
    )
    training_objective, validation_objective = build_objectives(
        reference,
        weights,
        args.top_k,
        args.specificity_margin,
    )
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay)
    log_path = args.output_dir / "training_log.csv"
    best = float("inf")
    fieldnames: list[str] | None = None
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, training_objective, device, optimizer, args.max_grad_norm)
            val_metrics = run_epoch(model, val_loader, validation_objective, device, None, args.max_grad_norm)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            if writer is None:
                fieldnames = list(row)
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, sort_keys=True))
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": best}, args.output_dir / "best_model.pt")
    print(f"completed: best validation loss={best:.6f}; checkpoint={args.output_dir / 'best_model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
