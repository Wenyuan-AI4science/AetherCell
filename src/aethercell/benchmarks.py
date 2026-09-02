"""Minimum reproducible analyses for AC-RP, synergy, CDx, TCGA, and AC-DR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score


def _finite(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            raise ValueError(f"missing required column {column!r}; found {list(result.columns)}")
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=columns)


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic)


def regression_metrics(frame: pd.DataFrame, pred: str, truth: str) -> dict[str, float | int]:
    clean = _finite(frame, [pred, truth])
    x, y = clean[pred].to_numpy(), clean[truth].to_numpy()
    return {
        "n": len(clean),
        "pearson": _correlation(x, y, "pearson"),
        "spearman": _correlation(x, y, "spearman"),
        "rmse": float(mean_squared_error(y, x) ** 0.5),
        "mae": float(mean_absolute_error(y, x)),
    }


def classification_metrics(frame: pd.DataFrame, score: str, label: str) -> dict[str, float | int]:
    clean = _finite(frame, [score, label])
    y, s = clean[label].astype(int).to_numpy(), clean[score].to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("classification metrics require both positive and negative labels")
    return {"n": len(clean), "auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def run_ac_rp(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    pred = "pred" if "pred" in frame else "score"
    truth = "true" if "true" in frame else "label"
    overall = regression_metrics(frame, pred, truth)
    grouped = None
    if "data_type" in frame:
        records = [{"data_type": key, **regression_metrics(group, pred, truth)} for key, group in frame.groupby("data_type")]
        grouped = pd.DataFrame(records)
    return {"task": "AC-RP", **overall}, grouped


def run_synergy(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    score = "y_prob_pred" if "y_prob_pred" in frame else "logit"
    label = "label_bin" if "label_bin" in frame else "label"
    metrics = {"task": "synergy", **classification_metrics(frame, score, label)}
    if {"y_cont_pred", "label_cont"}.issubset(frame.columns):
        metrics.update({f"continuous_{k}": v for k, v in regression_metrics(frame, "y_cont_pred", "label_cont").items()})
    grouped = None
    cell = "Cell.line" if "Cell.line" in frame else ("cell_name" if "cell_name" in frame else None)
    if cell:
        records = []
        for key, group in frame.groupby(cell):
            try:
                records.append({"cell_line": key, **classification_metrics(group, score, label)})
            except ValueError:
                continue
        grouped = pd.DataFrame(records)
    return metrics, grouped


def run_tcga(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    metrics = {"task": "TCGA", **classification_metrics(frame, "score", "label")}
    disease = "disease_id" if "disease_id" in frame else ("disase_id" if "disase_id" in frame else None)
    grouped = None
    if disease:
        records = []
        for key, group in frame.groupby(disease):
            try:
                records.append({"disease_id": key, **classification_metrics(group, "score", "label")})
            except ValueError:
                continue
        grouped = pd.DataFrame(records).sort_values("auroc", ascending=False) if records else pd.DataFrame()
    return metrics, grouped


def run_cdx(frame: pd.DataFrame, top_n: int) -> tuple[dict, pd.DataFrame]:
    clean = _finite(frame, ["ic50", "ic50_post_sh", "ic50_diff"])
    recomputed = clean["ic50_post_sh"] - clean["ic50"]
    max_error = float(np.max(np.abs(recomputed - clean["ic50_diff"]))) if len(clean) else float("nan")
    ranked = clean.sort_values("ic50_diff").head(top_n).copy()
    return {"task": "CDx", "n": len(clean), "max_difference_consistency_error": max_error}, ranked


def run_ac_dr(frame: pd.DataFrame, top_n: int) -> tuple[dict, pd.DataFrame]:
    required = ["drugbank_id", "moe_score", "te_score", "kg_score", "te_weight"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"AC-DR file is missing columns: {missing}")
    score_column = "moe_score"
    if "input_status" in frame and frame["input_status"].astype(str).str.fullmatch("KG_Only").all():
        score_column = "kg_score"
    ranked = frame.sort_values(score_column, ascending=False).head(top_n).copy()
    metrics = {
        "task": "AC-DR",
        "n_candidates": len(frame),
        "ranking_score": score_column,
        "n_unique_drugs": int(frame["drugbank_id"].nunique()),
        "top_score": float(pd.to_numeric(ranked[score_column]).iloc[0]),
    }
    return metrics, ranked


RUNNERS = {"ac-rp": run_ac_rp, "synergy": run_synergy, "tcga": run_tcga}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["ac-rp", "synergy", "cdx", "tcga", "ac-dr"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    if args.task == "cdx":
        metrics, table = run_cdx(frame, args.top_n)
    elif args.task == "ac-dr":
        metrics, table = run_ac_dr(frame, args.top_n)
    else:
        metrics, table = RUNNERS[args.task](frame)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    if table is not None:
        table.to_csv(args.output_dir / "details.csv", index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
