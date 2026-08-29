"""Resolve registered Hugging Face models with optional node-local runtime overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Tuple


def resolve_model_load_spec(
    model_config: Mapping[str, Any],
    *,
    override_env: str,
) -> Tuple[str, str | None, bool]:
    """Return model source, revision, and local-only mode without changing identity."""

    model_name = str(model_config["model_name"])
    revision = str(model_config["revision"])
    override = os.environ.get(override_env)
    if override:
        local_path = Path(override).resolve()
        if not local_path.is_dir():
            raise FileNotFoundError(
                f"Node-local model override from {override_env} is missing: {local_path}"
            )
        return str(local_path), None, True
    return model_name, revision, bool(int(os.environ.get("LBD_LOCAL_FILES_ONLY", "0")))
