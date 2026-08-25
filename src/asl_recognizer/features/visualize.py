from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .schema import LandmarkFrame


def draw_landmarks(
    image: npt.NDArray[np.uint8], frame: LandmarkFrame
) -> npt.NDArray[np.uint8]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for landmark visualization") from error
    output = image.copy()
    height, width = output.shape[:2]
    for coordinate, present in zip(frame.coordinates, frame.presence, strict=True):
        if not present:
            continue
        x = int(round(float(coordinate[0]) * (width - 1)))
        y = int(round(float(coordinate[1]) * (height - 1)))
        cv2.circle(output, (x, y), 2, (40, 220, 90), -1, lineType=cv2.LINE_AA)
    return output
