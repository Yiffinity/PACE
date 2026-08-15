from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pace.api.keys import assert_key_file_secure
from pace.config import AppConfig, load_config
from pace.data.manifest import build_training_manifests
from pace.utils.hashing import stable_hash


def _load_from_args(args: argparse.Namespace) -> AppConfig:
    return load_config(args.config, args.overlay)


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--overlay", action="append", default=[])


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = _load_from_args(args)
    dumped = config.model_dump(mode="json")
    print(json.dumps({"valid": True, "config_hash": stable_hash(dumped)}, indent=2))
    return 0


def _path_status(path: Path) -> dict[str, object]:
    return {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}


def _cmd_doctor(args: argparse.Namespace) -> int:
    config = _load_from_args(args)
    checks: dict[str, object] = {
        "datasets": _path_status(config.paths.datasets_root),
        "model_2b": _path_status(config.paths.model_2b),
        "model_4b": _path_status(config.paths.model_4b),
        "model_8b": _path_status(config.paths.model_8b),
        "retrieval_model": _path_status(config.paths.retrieval_model),
    }
    provider_checks: dict[str, object] = {}
    for name, provider in config.api.providers.items():
        try:
            assert_key_file_secure(provider.key_file)
            secure = True
            error = None
        except (OSError, PermissionError) as exc:
            secure = False
            error = str(exc)
        provider_checks[name] = {
            "key_file": str(provider.key_file),
            "exists": provider.key_file.exists(),
            "secure": secure,
            "error": error,
        }
    checks["api_providers"] = provider_checks
    ok = all(
        item.get("exists", False)
        for key, item in checks.items()
        if key != "api_providers" and isinstance(item, dict)
    ) and all(item["secure"] for item in provider_checks.values())
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 1


def _cmd_build_manifests(args: argparse.Namespace) -> int:
    config = _load_from_args(args)
    _, output, stats = build_training_manifests(
        config,
        output_directory=args.output,
    )
    result = {
        "output_directory": str(output),
        "manifest_hash": stats["manifest_hash"],
        "audit": stats["audit"],
        "statistics": stats["statistics"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pace")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    _add_config_arguments(validate)
    validate.set_defaults(handler=_cmd_validate_config)

    doctor = subparsers.add_parser("doctor")
    _add_config_arguments(doctor)
    doctor.set_defaults(handler=_cmd_doctor)

    manifests = subparsers.add_parser("build-manifests")
    _add_config_arguments(manifests)
    manifests.add_argument("--output", type=Path)
    manifests.set_defaults(handler=_cmd_build_manifests)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
