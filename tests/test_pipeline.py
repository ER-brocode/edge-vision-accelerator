import pytest
import time
import numpy as np
from edge_vision_accelerator.pipeline.frame import Frame, FrameMeta, PixelFormat
from edge_vision_accelerator.pipeline.ring_buffer import ZeroCopyFrameRingBuffer
from edge_vision_accelerator.pipeline.ingestion import VideoStreamIngestor


def test_frame_metadata_and_views():
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame.from_numpy(raw, frame_id=42, format=PixelFormat.BGR, camera_id="front_cam")

    assert frame.frame_id == 42
    assert frame.width == 640
    assert frame.height == 480
    assert frame.data.shape == (480, 640, 3)

    # Zero copy view check
    frame.data[0, 0, 0] = 255
    assert raw[0, 0, 0] == 255


def test_ring_buffer_put_and_get():
    buf = ZeroCopyFrameRingBuffer(capacity=4, height=100, width=100, channels=3)
    assert buf.size() == 0
    assert buf.is_empty()

    test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 123
    slot_idx = buf.put_frame(test_frame, frame_id=1)
    assert slot_idx == 0
    assert buf.size() == 1
    assert not buf.is_empty()

    f = buf.get_frame(timeout=0.1)
    assert f is not None
    assert f.frame_id == 1
    assert f.data[0, 0, 0] == 123
    assert buf.size() == 0


def test_ring_buffer_drop_oldest_on_overflow():
    buf = ZeroCopyFrameRingBuffer(capacity=2, height=50, width=50, channels=3)

    f1 = np.ones((50, 50, 3), dtype=np.uint8) * 1
    f2 = np.ones((50, 50, 3), dtype=np.uint8) * 2
    f3 = np.ones((50, 50, 3), dtype=np.uint8) * 3

    buf.put_frame(f1, frame_id=1, drop_oldest=True)
    buf.put_frame(f2, frame_id=2, drop_oldest=True)
    # Should drop f1
    buf.put_frame(f3, frame_id=3, drop_oldest=True)

    assert buf.dropped_frames == 1
    assert buf.size() == 2

    consumed_1 = buf.get_frame()
    assert consumed_1 is not None
    assert consumed_1.frame_id == 2
    assert consumed_1.data[0, 0, 0] == 2

    consumed_2 = buf.get_frame()
    assert consumed_2 is not None
    assert consumed_2.frame_id == 3
    assert consumed_2.data[0, 0, 0] == 3


def test_stream_ingestor_mock_pipeline():
    buf = ZeroCopyFrameRingBuffer(capacity=8, height=480, width=640, channels=3)
    ingestor = VideoStreamIngestor(source="mock", ring_buffer=buf, target_fps=60.0)

    ingestor.start()
    time.sleep(0.15)
    ingestor.stop()

    assert buf.size() > 0
    f = buf.get_frame()
    assert f is not None
    assert f.shape == (480, 640, 3)
