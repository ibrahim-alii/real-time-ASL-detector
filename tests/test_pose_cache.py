import numpy as np
import pytest

from asl_recognizer.features.mmpose_backend import MMPoseWholeBodyBackend, WholeBodyFrame


def test_raw_scores_and_low_confidence_coordinates_survive():
    points = np.ones((133, 2), np.float32) * 17
    scores = np.ones(133, np.float32)
    scores[91] = 0.2
    points[92] = np.nan
    frame = MMPoseWholeBodyBackend.frame_from_result(
        {"predictions": [[{"keypoints": points, "keypoint_scores": scores}]]}, 0)
    assert not frame.presence[91] and not frame.coordinates[91].any()
    assert frame.scores[91] == pytest.approx(0.2)
    np.testing.assert_array_equal(frame.raw_coordinates[91], [17, 17])
    assert not frame.raw_coordinates[92].any()


def test_raw_contract_rejects_malformed_scores():
    with pytest.raises(ValueError, match="scores"):
        WholeBodyFrame(np.zeros((133, 2), np.float32), np.zeros(133, np.bool_), 0,
                       scores=np.ones(132, np.float32))


def test_pose_cache_round_trip_and_regional_quality(tmp_path):
    from asl_recognizer.translation.pose_cache import PoseCache, pose_quality
    frames = []
    for timestamp in (0, 67, 133):
        presence = np.ones(133, np.bool_)
        presence[91:112] = False
        coordinates = np.ones((133, 2), np.float32)
        coordinates[~presence] = 0
        frames.append(WholeBodyFrame(coordinates, presence, timestamp,
            scores=presence.astype(np.float32), raw_coordinates=coordinates.copy()))
    cache = PoseCache(tmp_path)
    cache.store("clip", frames, metadata={"video_sha256": "fixture"})
    restored, metadata = cache.load("clip", expected_metadata={"video_sha256": "fixture"})
    assert metadata["video_sha256"] == "fixture"
    assert [frame.timestamp_ms for frame in restored] == [0, 67, 133]
    report = pose_quality(restored)
    assert report["regions"]["left_hand"]["visible_fraction"] == 0
    assert report["regions"]["right_hand"]["visible_fraction"] == 1
    assert report["regions"]["left_hand"]["longest_missing_frames"] == 3
    assert report["semantic_review"] == "pending"
    with pytest.raises(ValueError, match="identity"):
        cache.load("clip", expected_metadata={"video_sha256": "changed"})


def test_pose_cache_does_not_invent_raw_scores(tmp_path):
    from asl_recognizer.translation.pose_cache import PoseCache
    frame = WholeBodyFrame(np.zeros((133, 2), np.float32), np.zeros(133, np.bool_), 0)
    with pytest.raises(ValueError, match="raw"):
        PoseCache(tmp_path).store("clip", [frame], metadata={})
