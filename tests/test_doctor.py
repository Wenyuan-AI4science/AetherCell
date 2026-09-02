from pathlib import Path

from aethercell.doctor import main


def test_doctor_explains_external_asset_recovery(tmp_path: Path, capsys):
    project = tmp_path / "project"
    project.mkdir()
    for relative in (
        "README.md",
        "pyproject.toml",
        "scripts/download_data.py",
        "scripts/download_models.py",
        "scripts/reviewer_smoke_test.py",
        "src/aethercell/train.py",
        "src/aethercell/batch_inference.py",
        "src/aethercell/losses.py",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert main(["--project-root", str(project), "--model-dir", str(tmp_path / "models"), "--data-dir", str(tmp_path / "data")]) == 0
    output = capsys.readouterr().out
    assert "download_models.py --extract" in output
    assert "download_data.py --extract" in output
    assert "zenodo.18295255" in output
    assert main(["--project-root", str(project), "--model-dir", str(tmp_path / "models"), "--data-dir", str(tmp_path / "data"), "--full"]) == 3
