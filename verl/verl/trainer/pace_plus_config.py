"""Configuration loading with project-relative path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to read PACE_PLUS configuration") from exc
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"configuration must be a YAML object: {config_path}")
    config = deepcopy(value)
    config["_config_path"] = str(config_path)
    configured_root = config.get("output", {}).get("root")
    if configured_root:
        project_root = Path(configured_root).expanduser()
        if not project_root.is_absolute():
            project_root = config_path.parent.parent / project_root
    else:
        project_root = config_path.parent.parent
    config.setdefault("output", {})["root"] = str(project_root.resolve())
    for role in ("teacher", "student"):
        model = config.get("models", {}).get(role, {})
        if model.get("path"):
            model["path"] = str(project_path(config, model["path"]))
    return config


def get_required(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigError(f"missing required config value: {dotted_path}")
        value = value[part]
    if value is None or value == "":
        raise ConfigError(f"empty required config value: {dotted_path}")
    return value


def project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(get_required(config, "output.root")) / path
    return path.resolve()
