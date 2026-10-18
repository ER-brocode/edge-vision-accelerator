#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace eva {

/**
 * Pixel format for raw image frames
 */
enum class PixelFormat {
    BGR24,
    RGB24,
    RGBA32,
    BGRA32,
    NV12,
    YUV420P,
    GRAY8
};

/**
 * Hardware accelerator execution provider backend
 */
enum class BackendDevice {
    CPU,
    CUDA,
    TensorRT,
    OpenVINO,
    NPU_DirectML,
    RockchipNPU,
    CoralTPU
};

/**
 * Precision mode for neural acceleration
 */
enum class PrecisionMode {
    FP32,
    FP16,
    INT8
};

/**
 * Bounding Box representation in normalized [0, 1] or absolute pixel coordinates
 */
struct Rect {
    float x;      // left
    float y;      // top
    float width;  // width
    float height; // height

    float right() const { return x + width; }
    float bottom() const { return y + height; }
    float area() const { return width * height; }

    float iou(const Rect& other) const {
        float x1 = std::max(x, other.x);
        float y1 = std::max(y, other.y);
        float x2 = std::min(right(), other.right());
        float y2 = std::min(bottom(), other.bottom());

        float inter_w = std::max(0.0f, x2 - x1);
        float inter_h = std::max(0.0f, y2 - y1);
        float inter_area = inter_w * inter_h;

        float union_area = area() + other.area() - inter_area;
        if (union_area <= 0.0f) return 0.0f;
        return inter_area / union_area;
    }
};

/**
 * Single detected object hypothesis
 */
struct Detection {
    Rect bbox;
    int class_id;
    float confidence;
    std::string label;
    std::vector<float> embedding; // Optional feature embedding (for DeepSORT / ReID)
};

/**
 * Tracked object state with continuous lifecycle management
 */
enum class TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

struct TrackedObject {
    uint64_t track_id;
    Rect bbox;
    int class_id;
    float confidence;
    std::string label;
    TrackState state;
    uint32_t age_frames;
    uint32_t hits;
    uint32_t time_since_update;
    std::vector<float> velocity; // [vx, vy, va, vh]
    std::vector<float> feature_history;
};

/**
 * High-performance frame metadata structure
 */
struct FrameMeta {
    uint64_t frame_id;
    uint64_t timestamp_ns;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    PixelFormat format;
    size_t stride;
    size_t payload_bytes;
};

} // namespace eva
