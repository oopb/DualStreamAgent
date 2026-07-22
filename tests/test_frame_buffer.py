import pytest

from dualstream_agent.perception.frame_buffer import FrameBuffer
from dualstream_agent.schemas import FramePacket


def test_frame_buffer_is_bounded_and_handles_zero_count():
    buffer = FrameBuffer(maxlen=2)
    for frame_id in range(3):
        buffer.append(FramePacket(frame_id=frame_id, timestamp=float(frame_id), image=None))

    assert [frame.frame_id for frame in buffer.recent()] == [1, 2]
    assert buffer.recent(0) == []
    with pytest.raises(ValueError, match="non-negative"):
        buffer.recent(-1)
