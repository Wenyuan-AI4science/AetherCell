"""AetherCell model construction and safe checkpoint loading."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path

import torch


def _state_dict(checkpoint: object, preferred_key: str) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a mapping")
    state = checkpoint.get(preferred_key, checkpoint)
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise TypeError(f"checkpoint does not contain a valid {preferred_key!r} state dict")
    return {k.removeprefix("module."): v for k, v in state.items()}


def safe_torch_load(path: str | Path, device: torch.device | str = "cpu") -> object:
    """Load tensor-only checkpoints; arbitrary pickle objects are rejected."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"required model file is missing: {path}\n"
            "Download the verified model package with: "
            "python scripts/download_models.py --extract --output-dir models"
        )
    return torch.load(path, map_location=device, weights_only=True)


def build_drug_model(
    *,
    legacy_src: str | Path,
    lincs_vae_checkpoint: str | Path,
    rna_vae_checkpoint: str | Path,
    molformer_dir: str | Path,
    device: torch.device,
    predictor_checkpoint: str | Path | None = None,
) -> torch.nn.Module:
    legacy_src = Path(legacy_src).resolve()
    if str(legacy_src) not in sys.path:
        sys.path.insert(0, str(legacy_src))
    LINCSVAE = importlib.import_module("LINCSvae").LINCSVAE
    RNAVAE = importlib.import_module("RNAvae").RNAVAE
    predictor_module = "aethercell_drug" if (legacy_src / "aethercell_drug.py").is_file() else "uniperturb_drug"
    JointPerturbationPredictor = importlib.import_module(predictor_module).JointPerturbationPredictor

    lincs = LINCSVAE("cpu")
    lincs.load_state_dict(_state_dict(safe_torch_load(lincs_vae_checkpoint), "vae_model_state_dict"))
    rna = RNAVAE("cpu")
    rna.load_state_dict(_state_dict(safe_torch_load(rna_vae_checkpoint), "vae_model_state_dict"))
    model = JointPerturbationPredictor(
        lincs.encoder,
        lincs.decoder,
        rna.encoder,
        str(Path(molformer_dir).resolve()),
        device,
    ).to(device)
    if predictor_checkpoint is not None:
        state = _state_dict(safe_torch_load(predictor_checkpoint, device), "model_state_dict")
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = {"global_background_mu"}
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "predictor checkpoint is incompatible: "
                f"missing={sorted(unexpected_missing)}, unexpected={sorted(incompatible.unexpected_keys)}"
            )
    return model


def build_sh_model(
    *,
    legacy_src: str | Path,
    lincs_vae_checkpoint: str | Path,
    rna_vae_checkpoint: str | Path,
    device: torch.device,
    predictor_checkpoint: str | Path | None = None,
) -> torch.nn.Module:
    """Build the released shRNA perturbation model with safe checkpoint loading."""
    legacy_src = Path(legacy_src).resolve()
    if str(legacy_src) not in sys.path:
        sys.path.insert(0, str(legacy_src))
    LINCSVAE = importlib.import_module("LINCSvae").LINCSVAE
    RNAVAE = importlib.import_module("RNAvae").RNAVAE
    module_name = "aethercell_sh" if (legacy_src / "aethercell_sh.py").is_file() else "uniperturb_sh"
    Predictor = importlib.import_module(module_name).JointPerturbationPredictor_sh
    lincs = LINCSVAE("cpu")
    lincs.load_state_dict(_state_dict(safe_torch_load(lincs_vae_checkpoint), "vae_model_state_dict"))
    rna = RNAVAE("cpu")
    rna.load_state_dict(_state_dict(safe_torch_load(rna_vae_checkpoint), "vae_model_state_dict"))
    model = Predictor(lincs.encoder, lincs.decoder, rna.encoder, device).to(device)
    if predictor_checkpoint is not None:
        state = _state_dict(safe_torch_load(predictor_checkpoint, device), "model_state_dict")
        incompatible = model.load_state_dict(state, strict=False)
        unexpected_missing = set(incompatible.missing_keys) - {"global_background_mu"}
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "shRNA checkpoint is incompatible: "
                f"missing={sorted(unexpected_missing)}, unexpected={sorted(incompatible.unexpected_keys)}"
            )
    return model


def trainable_parameters(model: torch.nn.Module):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    return params
