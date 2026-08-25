from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .schema import LANDMARK_COUNT, LandmarkFrame

POSE_INDICES = (0, 11, 12, 13, 14, 15, 16)


class MediaPipeHolisticBackend:
    """Thin synchronous adapter around the MediaPipe Tasks Holistic landmarker."""

    def __init__(self, model_path: Path) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError("MediaPipe is required for landmark extraction") from error
        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self._mp = mp
        self._options = options
        self._landmarker = self._create_landmarker()

    def _create_landmarker(self) -> Any:
        return self._mp.tasks.vision.HolisticLandmarker.create_from_options(self._options)

    def reset(self) -> None:
        """Start an independent MediaPipe VIDEO timeline for a new clip."""
        self._landmarker.close()
        self._landmarker = self._create_landmarker()

    def detect(self, frame: npt.NDArray[np.generic], timestamp_ms: int) -> LandmarkFrame:
        rgb = np.ascontiguousarray(frame[..., ::-1], dtype=np.uint8)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        coordinates = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
        presence = np.zeros(LANDMARK_COUNT, dtype=np.bool_)
        self._copy_landmarks(result.left_hand_landmarks, coordinates, presence, 0)
        self._copy_landmarks(result.right_hand_landmarks, coordinates, presence, 21)
        pose = result.pose_landmarks
        if pose:
            if len(pose) <= max(POSE_INDICES):
                raise ValueError("MediaPipe pose result is incomplete")
            for output_index, source_index in enumerate(POSE_INDICES, start=42):
                landmark = pose[source_index]
                coordinates[output_index] = [landmark.x, landmark.y, landmark.z]
                presence[output_index] = getattr(landmark, "visibility", 1.0) >= 0.5
        return LandmarkFrame(coordinates, presence, timestamp_ms)

    @staticmethod
    def _copy_landmarks(
        landmarks: Any,
        coordinates: npt.NDArray[np.float32],
        presence: npt.NDArray[np.bool_],
        offset: int,
    ) -> None:
        if not landmarks:
            return
        if len(landmarks) != 21:
            raise ValueError("MediaPipe hand result must contain exactly 21 landmarks")
        for index, landmark in enumerate(landmarks):
            coordinates[offset + index] = [landmark.x, landmark.y, landmark.z]
            presence[offset + index] = True

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> MediaPipeHolisticBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
