"""High-throughput Frame Pipeline Benchmark for Edge Vision Accelerator."""

import time
import numpy as np
from edge_vision_accelerator.pipeline.ring_buffer import ZeroCopyFrameRingBuffer
from edge_vision_accelerator.tracker.byte_track import ByteTracker
from edge_vision_accelerator.inference.detector import Detection


def benchmark_ring_buffer_throughput(num_frames: int = 1000, resolution=(1080, 1920, 3)):
    """Measures zero-copy memory push/pull bandwidth in GB/s and FPS."""
    h, w, c = resolution
    frame_bytes = h * w * c
    total_data_gb = (frame_bytes * num_frames) / (1024 ** 3)

    buf = ZeroCopyFrameRingBuffer(capacity=32, height=h, width=w, channels=c)
    dummy_frame = np.ones((h, w, c), dtype=np.uint8) * 42

    print(f"\n================ FRAME BUFFER BENCHMARK ================")
    print(f"Target Resolution: {w}x{h} ({c} channels) | Frames: {num_frames}")

    start = time.perf_counter()
    for i in range(num_frames):
        buf.put_frame(dummy_frame, frame_id=i, drop_oldest=True)
        out = buf.get_frame(timeout=0.01)
        assert out is not None
    duration = time.perf_counter() - start

    fps = num_frames / duration
    throughput_gbps = total_data_gb / duration

    print(f"Elapsed Time:       {duration:.4f} s")
    print(f"Throughput FPS:     {fps:.2f} frames/sec")
    print(f"Data Throughput:    {throughput_gbps:.3f} GB/sec")
    print(f"Dropped Frames:     {buf.dropped_frames}")
    print(f"========================================================\n")
    return fps, throughput_gbps


def benchmark_tracker_throughput(num_frames: int = 1000, num_objects: int = 50):
    """Benchmarks ByteTrack association loop speed with 50 concurrent objects."""
    tracker = ByteTracker()
    print(f"================ TRACKER BENCHMARK ================")
    print(f"Tracking {num_objects} objects across {num_frames} frames...")

    start = time.perf_counter()
    for frame_idx in range(num_frames):
        dets = []
        for obj_id in range(num_objects):
            # Synthetic moving bounding boxes
            x = 100.0 + (obj_id * 15.0) + (frame_idx % 20) * 1.5
            y = 100.0 + (obj_id * 10.0) + (frame_idx % 20) * 1.0
            dets.append(Detection(
                bbox=[x, y, x + 40.0, y + 80.0],
                class_id=0,
                confidence=0.85 if (obj_id % 3 != 0) else 0.40,
                label="person"
            ))
        tracks = tracker.update(dets)

    duration = time.perf_counter() - start
    fps = num_frames / duration
    print(f"Elapsed Time:   {duration:.4f} s")
    print(f"Tracking FPS:   {fps:.2f} frames/sec")
    print(f"Latency/Frame:  {(duration / num_frames) * 1000.0:.3f} ms")
    print(f"====================================================\n")
    return fps


if __name__ == "__main__":
    benchmark_ring_buffer_throughput(num_frames=500, resolution=(720, 1280, 3))
    benchmark_tracker_throughput(num_frames=500, num_objects=30)
