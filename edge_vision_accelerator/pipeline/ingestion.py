"""Multi-source RTSP, V4L2 and mock camera ingestion pipelines."""

import threading
import time
import logging
from typing import Optional, Union, Callable
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from edge_vision_accelerator.pipeline.frame import Frame, PixelFormat
from edge_vision_accelerator.pipeline.ring_buffer import ZeroCopyFrameRingBuffer

logger = logging.getLogger(__name__)


class VideoStreamIngestor:
    """Base Ingestor that reads frames in a high-priority background worker thread

    and pushes directly into the zero-copy ring buffer without stalling upstream decoders.
    """

    def __init__(
        self,
        source: Union[str, int],
        ring_buffer: ZeroCopyFrameRingBuffer,
        target_fps: float = 30.0,
        camera_id: str = "cam_0",
        on_frame_callback: Optional[Callable[[Frame], None]] = None,
    ):
        self.source = source
        self.ring_buffer = ring_buffer
        self.target_fps = target_fps
        self.camera_id = camera_id
        self.on_frame_callback = on_frame_callback

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._capture = None
        self._fps_actual = 0.0
        self._last_fps_calc = time.time()
        self._frames_since_calc = 0

    def start(self):
        """Starts ingestion thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name=f"Ingestor-{self.camera_id}")
        self._thread.start()
        logger.info("Started stream ingestor for source: %s", self.source)

    def stop(self):
        """Stops ingestion thread and releases resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        logger.info("Stopped stream ingestor for source: %s", self.source)

    def _open_capture(self):
        if cv2 is None or self.source in ("mock", "synthetic"):
            return None
        try:
            cap = cv2.VideoCapture(self.source)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception:
            return None

    def _worker_loop(self):
        self._capture = self._open_capture()
        frame_interval = 1.0 / max(1.0, self.target_fps)

        while self._running:
            start_t = time.perf_counter()
            ret = False
            raw_frame = None

            if self._capture is not None and self._capture.isOpened():
                ret, raw_frame = self._capture.read()
                if not ret or raw_frame is None:
                    time.sleep(0.005)
                    continue
            else:
                raw_frame = self._generate_synthetic_frame()
                ret = True

            if ret and raw_frame is not None:
                self._frame_count += 1
                self._frames_since_calc += 1

                self.ring_buffer.put_frame(
                    raw_frame,
                    frame_id=self._frame_count,
                    format=PixelFormat.BGR,
                    camera_id=self.camera_id
                )

                now = time.time()
                if now - self._last_fps_calc >= 1.0:
                    self._fps_actual = self._frames_since_calc / (now - self._last_fps_calc)
                    self._frames_since_calc = 0
                    self._last_fps_calc = now

            elapsed = time.perf_counter() - start_t
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _generate_synthetic_frame(self) -> np.ndarray:
        h, w = self.ring_buffer.height, self.ring_buffer.width
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(20, 60, h, dtype=np.uint8)[:, None]
        arr[:, :, 1] = np.linspace(30, 80, w, dtype=np.uint8)[None, :]
        arr[:, :, 2] = 40

        t = self._frame_count * 0.05
        box_x = int((w - 120) * (0.5 + 0.4 * np.sin(t)))
        box_y = int((h - 120) * (0.5 + 0.4 * np.cos(t * 0.7)))
        arr[box_y:box_y + 100, box_x:box_x + 80] = [200, 180, 50]
        return arr

    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def frame_count(self) -> int:
        return self._frame_count


class RTSPStreamIngestor(VideoStreamIngestor):
    def __init__(self, rtsp_url: str, ring_buffer: ZeroCopyFrameRingBuffer, target_fps: float = 30.0, camera_id: str = "rtsp_cam"):
        super().__init__(source=rtsp_url, ring_buffer=ring_buffer, target_fps=target_fps, camera_id=camera_id)


class V4L2CameraIngestor(VideoStreamIngestor):
    def __init__(self, device_index: int, ring_buffer: ZeroCopyFrameRingBuffer, target_fps: float = 30.0, camera_id: str = "v4l2_cam"):
        super().__init__(source=device_index, ring_buffer=ring_buffer, target_fps=target_fps, camera_id=camera_id)
