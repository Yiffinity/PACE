import os

import pytest

from pace.api.keys import KeyFileSecurityError, load_api_keys


def test_key_loader_requires_private_permissions(tmp_path) -> None:
    path = tmp_path / "keys"
    path.write_text("secret-a\nsecret-b\n", encoding="utf-8")
    os.chmod(path, 0o640)
    with pytest.raises(KeyFileSecurityError):
        load_api_keys(path)


def test_key_loader_reads_unique_nonempty_lines(tmp_path) -> None:
    path = tmp_path / "keys"
    path.write_text("secret-a\n\nsecret-b\n", encoding="utf-8")
    os.chmod(path, 0o600)
    assert load_api_keys(path) == ("secret-a", "secret-b")


def test_duplicate_keys_are_rejected(tmp_path) -> None:
    path = tmp_path / "keys"
    path.write_text("secret-a\nsecret-a\n", encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(ValueError, match="duplicate"):
        load_api_keys(path)
