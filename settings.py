from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(os.getenv("MODELOAR_CONFIG", "config4h.json"))
DEFAULT_POLYGON_KEY_ENV = "POLYGON_API_KEY"


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    require_polygon_key: bool = False,
) -> dict[str, Any]:
    """Load local configuration and inject secrets from environment variables.

    The repository stores only non-sensitive configuration. The Polygon API key
    is read at runtime from the environment variable named by ``api.key_env``
    (``POLYGON_API_KEY`` by default).

    Parameters
    ----------
    config_path:
        Path to the local JSON configuration file. ``config4h.json`` is used by
        default and is intentionally ignored by Git.
    require_polygon_key:
        If True, raise an actionable error when the environment variable is not
        defined. Scripts that do not call Polygon can leave this as False.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file '{path}' was not found. "
            "Copy config4h.example.json to config4h.json and configure it locally."
        )

    with path.open("r", encoding="utf-8") as config_file:
        config: dict[str, Any] = json.load(config_file)

    api_config = config.setdefault("api", {})
    env_name = api_config.get("key_env", DEFAULT_POLYGON_KEY_ENV)
    api_key = os.getenv(env_name)

    if api_key:
        # The secret exists only in memory; it is never written back to disk.
        api_config["key"] = api_key
    else:
        # Ensure a stale/static key is not silently used from public config.
        api_config.pop("key", None)
        if require_polygon_key:
            raise RuntimeError(
                f"Polygon API key is not configured. Set environment variable '{env_name}' "
                "before running code that calls the Polygon API."
            )

    return config


def get_polygon_api_key(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    """Return the Polygon API key from the environment or raise a clear error."""
    config = load_config(config_path, require_polygon_key=True)
    return config["api"]["key"]
