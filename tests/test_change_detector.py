import numpy as np

from dualstream_agent.perception.change_detector import ChangeDetector


def test_change_detector_skips_identical_frame_then_forces_after_silence():
    detector = ChangeDetector(threshold=0.05, silence_ceiling_s=5.0)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert detector.score(frame, 0.0).is_novel
    assert not detector.score(frame, 1.0).is_novel
    forced = detector.score(frame, 6.0)
    assert forced.is_novel
    assert forced.forced
