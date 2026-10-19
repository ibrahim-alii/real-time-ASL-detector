"""Feature identities prevent mixing coordinate systems or trained projections."""

from __future__ import annotations

from dataclasses import dataclass

MMPOSE_SCHEMA = "mmpose-coco-wholebody-133-xy-v1"


@dataclass(frozen=True, slots=True)
class TranslationSchema:
    landmarks: int
    coordinates: int

    @property
    def dimension(self) -> int:
        return self.landmarks * self.coordinates


SCHEMAS = {
    MMPOSE_SCHEMA: TranslationSchema(133, 2),
}


def translation_schema(name: str) -> TranslationSchema:
    try:
        return SCHEMAS[name]
    except KeyError as error:
        raise ValueError(f"unsupported translation feature schema: {name}") from error
