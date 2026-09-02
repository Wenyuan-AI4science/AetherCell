"""Generate the historical inference output contract from portable CLI outputs."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def _require(output_dir: Path, *names: str) -> None:
    missing = [name for name in names if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot create legacy outputs; missing portable outputs: {missing}")


def write_legacy_delta_z(output_dir: str | Path, chunk_size: int = 4096) -> None:
    """Add ``delta_z_predictions.npy/csv`` without removing portable outputs."""
    output_dir = Path(output_dir)
    _require(output_dir, "predicted_delta_z.npy", "metadata.csv")
    source = output_dir / "predicted_delta_z.npy"
    shutil.copyfile(source, output_dir / "delta_z_predictions.npy")
    latent = np.load(source, mmap_mode="r")
    metadata = pd.read_csv(output_dir / "metadata.csv")
    if len(metadata) != len(latent):
        raise ValueError("metadata and predicted_delta_z have different row counts")
    destination = output_dir / "delta_z_predictions.csv"
    columns = [f"delta_z_{index}" for index in range(latent.shape[1])]
    first = True
    for start in range(0, len(latent), chunk_size):
        stop = min(start + chunk_size, len(latent))
        values = pd.DataFrame(np.asarray(latent[start:stop]), columns=columns)
        table = pd.concat((metadata.iloc[start:stop].reset_index(drop=True), values), axis=1)
        table.to_csv(destination, mode="w" if first else "a", header=first, index=False)
        first = False
    print(f"wrote legacy delta-z outputs to {output_dir}")


def write_legacy_expression(output_dir: str | Path, chunk_size: int = 4096) -> None:
    """Add historical expression filenames and summary statistics in chunks."""
    output_dir = Path(output_dir)
    _require(output_dir, "predicted_expression.npy", "predicted_delta.npy", "metadata.csv")
    prediction_source = output_dir / "predicted_expression.npy"
    delta_source = output_dir / "predicted_delta.npy"
    shutil.copyfile(prediction_source, output_dir / "perturbed_expression.npy")
    shutil.copyfile(delta_source, output_dir / "delta_expression.npy")
    prediction = np.load(prediction_source, mmap_mode="r")
    delta = np.load(delta_source, mmap_mode="r")
    if prediction.shape != delta.shape:
        raise ValueError(f"prediction/delta shape mismatch: {prediction.shape} versus {delta.shape}")
    control = np.lib.format.open_memmap(
        output_dir / "control_expression.npy",
        mode="w+",
        dtype=prediction.dtype,
        shape=prediction.shape,
    )
    prediction_sum = prediction_square_sum = 0.0
    delta_sum = delta_square_sum = delta_absolute_sum = 0.0
    for start in range(0, len(prediction), chunk_size):
        stop = min(start + chunk_size, len(prediction))
        prediction_chunk = np.asarray(prediction[start:stop])
        delta_chunk = np.asarray(delta[start:stop])
        control[start:stop] = prediction_chunk - delta_chunk
        prediction64 = prediction_chunk.astype(np.float64, copy=False)
        delta64 = delta_chunk.astype(np.float64, copy=False)
        prediction_sum += float(prediction64.sum())
        prediction_square_sum += float(np.square(prediction64).sum())
        delta_sum += float(delta64.sum())
        delta_square_sum += float(np.square(delta64).sum())
        delta_absolute_sum += float(np.abs(delta64).sum())
    control.flush()
    del control
    count = int(prediction.size)
    prediction_mean = prediction_sum / count
    delta_mean = delta_sum / count
    summary = {
        "n_samples": len(prediction),
        "n_genes": prediction.shape[1],
        "perturbed_mean": prediction_mean,
        "perturbed_std": float(max(0.0, prediction_square_sum / count - prediction_mean**2) ** 0.5),
        "delta_mean": delta_mean,
        "delta_std": float(max(0.0, delta_square_sum / count - delta_mean**2) ** 0.5),
        "delta_abs_mean": delta_absolute_sum / count,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary_statistics.csv", index=False)
    print(f"wrote legacy expression outputs to {output_dir}")
