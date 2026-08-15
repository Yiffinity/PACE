from __future__ import annotations

import os
import stat
from pathlib import Path


class KeyFileSecurityError(PermissionError):
    pass


def key_file_mode(path: str | Path) -> int:
    return stat.S_IMODE(Path(path).stat().st_mode)


def assert_key_file_secure(path: str | Path) -> None:
    key_path = Path(path)
    mode = key_file_mode(key_path)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise KeyFileSecurityError(
            f"API key file must not be accessible by group/others: {key_path} mode={mode:o}"
        )


def load_api_keys(path: str | Path) -> tuple[str, ...]:
    key_path = Path(path)
    assert_key_file_secure(key_path)
    keys = tuple(
        line.strip() for line in key_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not keys:
        raise ValueError(f"API key file has no non-empty records: {key_path}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"API key file contains duplicate records: {key_path}")
    return keys


def secure_key_file(path: str | Path) -> None:
    os.chmod(Path(path), 0o600)
