"""COCO-WholeBody 2D extraction using explicitly supplied local MMPose assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

WHOLEBODY_POINTS = 133


@dataclass(frozen=True, slots=True)
class WholeBodyFrame:
    # COCO order: 17 body, 6 feet, 68 face, 21 left hand, 21 right hand.
    coordinates: npt.NDArray[np.float32]
    presence: npt.NDArray[np.bool_]
    timestamp_ms: int

    def __post_init__(self) -> None:
        if self.coordinates.shape != (WHOLEBODY_POINTS, 2):
            raise ValueError("whole-body coordinates must have shape [133, 2]")
        if self.presence.shape != (WHOLEBODY_POINTS,) or self.presence.dtype != np.bool_:
            raise ValueError("whole-body presence must be a 133-point boolean mask")
        if self.coordinates.dtype != np.float32 or not np.isfinite(self.coordinates).all():
            raise ValueError("whole-body coordinates must be finite float32")
        if self.timestamp_ms < 0 or np.any(self.coordinates[~self.presence] != 0):
            raise ValueError("invalid timestamp or nonzero missing landmarks")


class MMPoseWholeBodyBackend:
    """Single-signer adapter; reject multiple people instead of changing identities."""

    def __init__(
        self, *, pose_config: Path, pose_weights: Path, detector_config: Path,
        detector_weights: Path, device: str = "cpu", confidence_threshold: float = 0.5,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("keypoint confidence threshold must be in [0, 1]")
        for path in (pose_config, pose_weights, detector_config, detector_weights):
            if not path.is_file():
                raise FileNotFoundError(path)
        try:
            from mmdet.apis import inference_detector, init_detector
            from mmpose.apis import inference_topdown, init_model
            from mmpose.evaluation.functional import nms
            from mmpose.utils import adapt_mmdet_pipeline
        except ImportError as error:
            raise RuntimeError(
                "MMPose extraction needs a compatible MMPose/MMCV/MMDetection environment; "
                "see docs/MMPOSE_DATA_GUIDE.md"
            ) from error
        detector: Any = init_detector(str(detector_config), str(detector_weights), device=device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        pose: Any = init_model(str(pose_config), str(pose_weights), device=device)

        def boxes(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
            instances = inference_detector(detector, frame).pred_instances.cpu().numpy()
            candidates = np.concatenate((instances.bboxes, instances.scores[:, None]), axis=1)
            candidates = candidates[(instances.labels == 0) & (instances.scores > 0.3)]
            return np.asarray(candidates[nms(candidates, 0.3), :4], dtype=np.float32)

        self._boxes = boxes

        def inferencer(frame: npt.NDArray[np.uint8], **_kwargs: object) -> Any:
            detected = boxes(frame)
            if len(detected) > 1:
                raise ValueError("ambiguous multiple-person frame")
            people = []
            if len(detected) == 1:
                samples = inference_topdown(pose, frame, detected)
                for sample in samples:
                    instances = sample.pred_instances.cpu().numpy()
                    people.append({"keypoints": instances.keypoints[0],
                                   "keypoint_scores": instances.keypoint_scores[0]})
            # Crucially, never call top-down inference with an empty box array:
            # its whole-image fallback can invent a person on an empty frame.
            yield {"predictions": [people]}

        self._inferencer: Any = inferencer
        self.confidence_threshold = confidence_threshold

    def detect_translation(
        self, frame: npt.NDArray[np.uint8], timestamp_ms: int
    ) -> WholeBodyFrame:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("MMPose input must be a uint8 BGR image")
        result = next(self._inferencer(frame, return_vis=False, show=False))
        return self.frame_from_result(
            result, timestamp_ms, confidence_threshold=self.confidence_threshold
        )

    def count_people(self, frame: npt.NDArray[np.uint8]) -> int:
        """Sampled-frame validation using the same detector as extraction."""
        return min(2, len(self._boxes(frame)))

    @staticmethod
    def frame_from_result(
        result: dict[str, Any], timestamp_ms: int, *, confidence_threshold: float = 0.5
    ) -> WholeBodyFrame:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("keypoint confidence threshold must be in [0, 1]")
        predictions = result.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise ValueError("expected predictions for exactly one image")
        people = predictions[0]
        if not isinstance(people, list) or len(people) > 1:
            raise ValueError("ambiguous multiple-person frame")
        coordinates = np.zeros((WHOLEBODY_POINTS, 2), dtype=np.float32)
        presence = np.zeros(WHOLEBODY_POINTS, dtype=np.bool_)
        if people:
            points = np.asarray(people[0]["keypoints"], dtype=np.float32)
            scores = np.asarray(people[0]["keypoint_scores"], dtype=np.float32)
            if points.shape != coordinates.shape or scores.shape != presence.shape:
                raise ValueError("MMPose model must use COCO-WholeBody 133-point 2D output")
            presence = (
                np.isfinite(points).all(axis=1) & np.isfinite(scores)
                & (scores >= confidence_threshold)
            )
            coordinates[presence] = points[presence]
        return WholeBodyFrame(coordinates, presence, timestamp_ms)

    def reset(self) -> None:
        # Each numpy image is processed independently, without a video tracker.
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> MMPoseWholeBodyBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
