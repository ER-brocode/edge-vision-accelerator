"""Pre-allocated circular ring buffer for zero-copy high-throughput video ingestion."""

import threading
import time
from typing import Optional, Tuple
import numpy as np
from edge_vision_accelerator.pipeline.frame import Frame, FrameMeta, PixelFormat


class ZeroCopyFrameRingBuffer:
    """Pre-allocated continuous memory ring buffer.

    Producers write into pre-allocated memory slices directly to avoid GC pressure
    and buffer re-allocation in 60fps / 120fps edge camera ingestion.
    """

    def __init__(self, capacity: int, height: int, width: int, channels: int = 3, dtype=np.uint8):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero.")
        self.capacity = capacity
        self.height = height
        self.width = width
        self.channels = channels
        self.dtype = dtype

        # Single contiguous block of memory allocated upfront
        self._buffer = np.zeros((capacity, height, width, channels), dtype=dtype)
        self._metas: list[Optional[FrameMeta]] = [None] * capacity

        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._count = 0
        self._dropped_frames = 0

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._shutdown = False

    def acquire_write_slot(self, drop_oldest: bool = True) -> Tuple[int, np.ndarray]:
        """Acquires a view into the buffer slot where the producer can write directly.

        Returns (slot_index, numpy_array_view).
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Ring buffer is shut down.")

            if self._count == self.capacity:
                if not drop_oldest:
                    raise OverflowError("Ring buffer is full and drop_oldest is False.")
                # Drop oldest
                self._tail = (self._tail + 1) % self.capacity
                self._count -= 1
                self._dropped_frames += 1

            slot_idx = self._head
            # Return memory view slice
            slot_view = self._buffer[slot_idx]
            return slot_idx, slot_view

    def commit_write_slot(self, slot_idx: int, frame_id: int, format: PixelFormat = PixelFormat.BGR, camera_id: str = "default"):
        """Commits the written frame slot and increments the write head."""
        with self._lock:
            if self._shutdown:
                return

            self._metas[slot_idx] = FrameMeta(
                frame_id=frame_id,
                timestamp_ns=time.time_ns(),
                width=self.width,
                height=self.height,
                channels=self.channels,
                format=format,
                stride=self._buffer[slot_idx].strides[0],
                camera_id=camera_id
            )
            self._head = (self._head + 1) % self.capacity
            self._count += 1
            self._not_empty.notify()

    def put_frame(self, frame_arr: np.ndarray, frame_id: int = 0, format: PixelFormat = PixelFormat.BGR, camera_id: str = "default", drop_oldest: bool = True) -> int:
        """Copies array data directly into preallocated slot without reallocation."""
        slot_idx, slot_view = self.acquire_write_slot(drop_oldest=drop_oldest)
        # Ensure dimensions match, or slice copy
        h, w = frame_arr.shape[:2]
        if (h, w) == (self.height, self.width):
            slot_view[:] = frame_arr
        else:
            # Clip or fit
            min_h, min_w = min(h, self.height), min(w, self.width)
            slot_view[:min_h, :min_w] = frame_arr[:min_h, :min_w]

        self.commit_write_slot(slot_idx, frame_id, format=format, camera_id=camera_id)
        return slot_idx

    def get_frame(self, timeout: Optional[float] = 1.0) -> Optional[Frame]:
        """Consumes a frame from the ring buffer with zero-copy wrapping."""
        with self._not_empty:
            start_time = time.time()
            while self._count == 0:
                if self._shutdown:
                    return None
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining)
                else:
                    self._not_empty.wait()

            if self._count == 0:
                return None

            slot_idx = self._tail
            meta = self._metas[slot_idx]
            frame_view = self._buffer[slot_idx]

            self._tail = (self._tail + 1) % self.capacity
            self._count -= 1

            if meta is None:
                meta = FrameMeta(
                    frame_id=0, timestamp_ns=time.time_ns(),
                    width=self.width, height=self.height,
                    channels=self.channels, format=PixelFormat.BGR,
                    stride=frame_view.strides[0]
                )

            return Frame(frame_view, meta)

    def size(self) -> int:
        with self._lock:
            return self._count

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames

    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    def is_full(self) -> bool:
        with self._lock:
            return self._count == self.capacity

    def shutdown(self):
        with self._lock:
            self._shutdown = True
            self._not_empty.notify_all()

    def reset(self):
        with self._lock:
            self._head = 0
            self._tail = 0
            self._count = 0
            self._dropped_frames = 0
            self._shutdown = False
