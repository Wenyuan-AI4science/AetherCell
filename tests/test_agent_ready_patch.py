import ast
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apply_agent_ready_patch.py"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("apply_agent_ready_patch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agent_patch_assets_match_manifest_and_contain_no_random_profile_calls():
    patcher = _load_patcher()
    patch_root = ROOT / "patches" / "agent_ready"
    for relative, hashes in patcher.FILES.items():
        path = patch_root / relative
        assert path.is_file()
        assert _sha256(path) == hashes["patched"]
        if path.suffix == ".py":
            source = path.read_text(encoding="utf-8")
            ast.parse(source)
            assert "torch.randn" not in source
            assert "np.random" not in source
            assert "never substitutes a random cell profile" in source


def test_public_context_is_pickle_free_and_checksum_pinned():
    patcher = _load_patcher()
    path = ROOT / "examples" / "data" / "api_context_examples.npz"
    assert _sha256(path) == patcher.EXAMPLE_SHA256

    import numpy as np

    with np.load(path, allow_pickle=False) as context:
        assert context["rna"].shape == (3, 10085)
        assert context["control"].shape == (3, 978)
        assert set(context["cell_id"].tolist()) == {"A549"}
        assert np.isfinite(context["rna"]).all()
        assert np.isfinite(context["control"]).all()
