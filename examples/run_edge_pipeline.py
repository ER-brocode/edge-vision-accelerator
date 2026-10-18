"""End-to-end edge camera processing pipeline example."""

import time
import numpy as np
from edge_vision_accelerator.pipeline.ring_buffer import ZeroCopyFrameRingBuffer
from edge_vision_accelerator.pipeline.ingestion import VideoStreamIngestor
from edge_vision_accelerator.inference.detector import QuantizedYOLODetector
from edge_vision_accelerator.tracker.byte_track import ByteTracker


def run_pipeline():
    print(">>> Initializing Edge Vision AI Stream Pipeline...")

    # 1. Ring Buffer for 1080p stream
    ring_buf = ZeroCopyFrameRingBuffer(capacity=16, height=720, width=1280, channels=3)

    # 2. Camera Ingestor
    ingestor = VideoStreamIngestor(source="synthetic", ring_buffer=ring_buf, target_fps=30.0)
    ingestor.start()

    # 3. Neural Detector & Multi-Object Tracker
    detector = QuantizedYOLODetector()
    tracker = ByteTracker()

    print(">>> Ingestor running. Processing incoming frames...")
    processed = 0
    start_time = time.time()

    try:
        while processed < 60:
            frame = ring_buf.get_frame(timeout=0.5)
            if frame is None:
                continue

            processed += 1
            # Run inference
            dets = detector.detect(frame.data)

            # In our synthetic frame, inject detected object
            if not dets:
                h, w = frame.shape[:2]
                dets = [detector.postprocess([], 1.0, (0, 0), (h, w))] if hasattr(detector, '_test') else []

            # Multi-object tracking
            tracks = tracker.update(dets)

            if processed % 15 == 0:
                print(f"  [Frame {processed:03d}] Buffer Size: {ring_buf.size()} | Detections: {len(dets)} | Active Tracks: {len(tracks)}")

    finally:
        ingestor.stop()

    total_time = time.time() - start_time
    print(f">>> Completed {processed} frames in {total_time:.2f}s ({processed / total_time:.1f} FPS)")


if __name__ == "__main__":
    run_pipeline()
