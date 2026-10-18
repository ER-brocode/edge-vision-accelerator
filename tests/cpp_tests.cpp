#include "eva/types.hpp"
#include "eva/ring_buffer.hpp"
#include "eva/accelerator.hpp"
#include "eva/byte_track.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[EVA-CPP] Running Native Edge Vision Accelerator Test Suite..." << std::endl;

    // 1. Test Rect & IOU
    eva::Rect r1{10.0f, 10.0f, 20.0f, 20.0f};
    eva::Rect r2{10.0f, 10.0f, 20.0f, 20.0f};
    eva::Rect r3{50.0f, 50.0f, 20.0f, 20.0f};
    assert(r1.iou(r2) > 0.99f);
    assert(r1.iou(r3) == 0.0f);
    std::cout << "  ✓ Rect IOU computation verified." << std::endl;

    // 2. Test Ring Buffer
    eva::ZeroCopyFrameRingBuffer ring(4, 1024 * 1024);
    assert(ring.capacity() == 4);
    assert(ring.size() == 0);

    auto* w1 = ring.acquire_producer_slot();
    assert(w1 != nullptr);
    eva::FrameMeta m1{1, 1000, 640, 480, 3, eva::PixelFormat::BGR24, 640 * 3, 640 * 480 * 3};
    ring.commit_producer_slot(w1, m1);
    assert(ring.size() == 1);

    auto* c1 = ring.acquire_consumer_slot(100);
    assert(c1 != nullptr);
    assert(c1->meta().frame_id == 1);
    ring.release_consumer_slot(c1);
    assert(ring.size() == 0);
    std::cout << "  ✓ Zero-copy frame ring buffer verified." << std::endl;

    // 3. Test ByteTracker
    eva::ByteTrackerCpp tracker;
    std::vector<eva::Detection> detections;
    eva::Detection d;
    d.bbox = eva::Rect{100.0f, 100.0f, 50.0f, 80.0f};
    d.confidence = 0.95f;
    d.class_id = 0;
    d.label = "person";
    detections.push_back(d);

    auto tracks1 = tracker.update(detections);
    // Move slightly
    detections[0].bbox.x += 2.0f;
    auto tracks2 = tracker.update(detections);
    assert(!tracks2.empty());
    assert(tracks2[0].hits >= 2);
    std::cout << "  ✓ C++ ByteTrack Kalman filter and association verified." << std::endl;

    std::cout << "[EVA-CPP] All Native C++ Unit Tests Passed!" << std::endl;
    return 0;
}
