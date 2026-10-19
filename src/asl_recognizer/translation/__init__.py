"""Direct landmark-to-English translation experiment components."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .contracts import DirectTranslationResult, TranslationRequest
    from .features import TRANSLATION_FEATURE_SCHEMA, mmpose_translation_request
    from .manifest import (
        ClipManifest,
        ManifestRecord,
        build_manifest,
        read_manifest,
        write_manifest,
    )


def __getattr__(name: str) -> Any:
    """Keep scheduler imports usable before scientific packages are installed."""
    modules = {
        "contracts": ("DirectTranslationResult", "TranslationRequest"),
        "features": ("TRANSLATION_FEATURE_SCHEMA", "mmpose_translation_request"),
        "manifest": ("ClipManifest", "ManifestRecord", "build_manifest", "read_manifest",
                     "write_manifest"),
    }
    for module, names in modules.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(name)

__all__ = [
    "TRANSLATION_FEATURE_SCHEMA",
    "ClipManifest",
    "DirectTranslationResult",
    "ManifestRecord",
    "TranslationRequest",
    "build_manifest",
    "mmpose_translation_request",
    "read_manifest",
    "write_manifest",
]
