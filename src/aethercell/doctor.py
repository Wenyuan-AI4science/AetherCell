"""Preflight checker with exact recovery instructions for missing AetherCell assets."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


CORE_PROJECT_FILES = [
    "README.md",
    "pyproject.toml",
    "scripts/download_data.py",
    "scripts/download_models.py",
    "scripts/reviewer_smoke_test.py",
    "src/aethercell/train.py",
    "src/aethercell/batch_inference.py",
    "src/aethercell/losses.py",
]
MODEL_FILES = [
    "models/transcriptome_prediction/L1000_vae.pt",
    "models/transcriptome_prediction/RNA_vae.pt",
    "models/transcriptome_prediction/predictor_L_drug.pt",
    "models/transcriptome_prediction/predictor_L_sh.pt",
    "models/transcriptome_prediction/molformer",
    "models/moe_repurposing/standalone_expert_model.pt",
    "models/moe_repurposing/data_sub/static_data.h5",
]
DATA_FILES = [
    "compound_perturbed/df_train_drug.csv",
    "compound_perturbed/df_test_drug.csv",
    "compound_perturbed/L1000_exp.npy",
    "compound_perturbed/RNAseq.parquet",
    "shRNA_perturbed/sh_meta_s1_train.csv",
    "shRNA_perturbed/sh_meta_s1_test.csv",
    "gdsc2_data/cell_blind_train.csv",
]
DEPENDENCIES = ["numpy", "pandas", "scipy", "sklearn", "torch", "h5py", "pyarrow", "transformers", "peft"]


def _find_model_root(path: Path) -> Path:
    candidates = [
        path,
        path / "aethercell-drug-discovery-v1.0.0",
        path / "models" / "aethercell-drug-discovery-v1.0.0",
    ]
    for candidate in candidates:
        if (candidate / "models" / "transcriptome_prediction").is_dir():
            return candidate
    return path


def _find_data_root(path: Path) -> Path:
    candidates = [path, path / "data4zendo", path / "data4train" / "data4zendo"]
    for candidate in candidates:
        if (candidate / "compound_perturbed").is_dir():
            return candidate
    return path


def _check_paths(root: Path, relative_paths: list[str]) -> list[str]:
    return [relative for relative in relative_paths if not (root / relative).exists()]


def inspect(project_root: Path, model_dir: Path, data_dir: Path) -> dict:
    project_root = project_root.resolve()
    model_root = _find_model_root(model_dir.resolve())
    data_root = _find_data_root(data_dir.resolve())
    dependency_status = {}
    for name in DEPENDENCIES:
        available = importlib.util.find_spec(name) is not None
        version = None
        if available:
            package = "scikit-learn" if name == "sklearn" else name
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        dependency_status[name] = {"available": available, "version": version}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(project_root),
        "model_root": str(model_root),
        "data_root": str(data_root),
        "missing_project_files": _check_paths(project_root, CORE_PROJECT_FILES),
        "missing_model_files": _check_paths(model_root, MODEL_FILES),
        "missing_data_files": _check_paths(data_root, DATA_FILES),
        "dependencies": dependency_status,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/zenodo"))
    parser.add_argument("--full", action="store_true", help="fail unless both external data and models are present")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = inspect(args.project_root, args.model_dir, args.data_dir)
    required_dependencies = ["numpy", "pandas", "scipy", "sklearn", "torch"]
    missing_dependencies = [name for name in required_dependencies if not report["dependencies"][name]["available"]]
    report["missing_required_dependencies"] = missing_dependencies
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"AetherCell doctor | Python {report['python']}")
        print(f"Project: {report['project_root']}")
        if report["missing_project_files"]:
            print("[FAIL] Incomplete project checkout:")
            for path in report["missing_project_files"]:
                print(f"  - {path}")
        else:
            print("[OK] Project code and reproduction examples are present.")
        if missing_dependencies:
            print(f"[FAIL] Missing required Python packages: {', '.join(missing_dependencies)}")
            print('  Fix: pip install -e ".[model,test]"')
        else:
            print("[OK] Required metric/training dependencies are importable.")
        if report["missing_model_files"]:
            print(f"[INFO] Model assets are incomplete under {report['model_root']}.")
            print("  Download: python scripts/download_models.py --metadata-only")
            print("  Install:  python scripts/download_models.py --extract --output-dir models")
            print("  Source:   https://huggingface.co/liwenyuan99/AetherCell")
        else:
            print("[OK] Core drug, shRNA, AC-RP and AC-DR model assets are present.")
        if report["missing_data_files"]:
            print(f"[INFO] Full training data are incomplete under {report['data_root']}.")
            print("  Download: python scripts/download_data.py --metadata-only")
            print("  Install:  python scripts/download_data.py --extract --output-dir data/zenodo")
            print("  Source:   https://doi.org/10.5281/zenodo.18295255")
        else:
            print("[OK] Compound, shRNA and GDSC2 processed data are present.")
        print("Reviewer smoke test: python scripts/reviewer_smoke_test.py")
    project_ready = not report["missing_project_files"] and not missing_dependencies
    external_ready = not report["missing_model_files"] and not report["missing_data_files"]
    if not project_ready:
        return 2
    if args.full and not external_ready:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
