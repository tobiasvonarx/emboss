"""Checkpoint discovery must be portable and reject corrupted public artifacts."""

from pathlib import Path

import pytest
from emboss.segmentation import checkpoint


def test_explicit_environment_overrides_working_directory_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f'{checkpoint.CHECKPOINT_ENV_VAR}="dotenv.pt"\n')
    monkeypatch.delenv(checkpoint.CHECKPOINT_ENV_VAR, raising=False)
    assert checkpoint.default_checkpoint_path() == tmp_path / "dotenv.pt"
    monkeypatch.setenv(checkpoint.CHECKPOINT_ENV_VAR, "environment.pt")
    assert checkpoint.default_checkpoint_path() == tmp_path / "environment.pt"


def test_dotenv_does_not_search_parent_or_source_directory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"{checkpoint.CHECKPOINT_ENV_VAR}=parent.pt\n")
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    monkeypatch.chdir(invocation)
    monkeypatch.delenv(checkpoint.CHECKPOINT_ENV_VAR, raising=False)
    monkeypatch.setattr(checkpoint, "download_checkpoint", lambda: Path("public.pt"))
    assert checkpoint.default_checkpoint_path() == Path("public.pt")


def test_public_download_pins_revision_and_checks_contents(tmp_path, monkeypatch):
    import hashlib

    import huggingface_hub

    artifact = tmp_path / "best.pt"
    artifact.write_bytes(b"known checkpoint")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(artifact)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(checkpoint, "MODEL_SIZE_BYTES", artifact.stat().st_size)
    monkeypatch.setattr(
        checkpoint, "MODEL_SHA256", hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    checkpoint._verify_checkpoint.cache_clear()
    assert checkpoint.download_checkpoint() == artifact
    assert calls == [
        {
            "repo_id": "tvonarx/emboss-segmentation",
            "filename": "best.pt",
            "revision": checkpoint.MODEL_REVISION,
        }
    ]
    artifact.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        checkpoint.download_checkpoint()
    artifact.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="size mismatch"):
        checkpoint.download_checkpoint()
    checkpoint._verify_checkpoint.cache_clear()
