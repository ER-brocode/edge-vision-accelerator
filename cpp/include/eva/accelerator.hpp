#pragma once

#include "eva/types.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace eva {

/**
 * Common configuration for neural accelerator execution
 */
struct AcceleratorConfig {
    std::string model_path;
    BackendDevice backend = BackendDevice::CPU;
    PrecisionMode precision = PrecisionMode::FP16;
    int device_id = 0;
    int input_width = 640;
    int input_height = 640;
    int input_channels = 3;
    float conf_threshold = 0.45f;
    float nms_threshold = 0.50f;
    int max_batch_size = 1;
    bool enable_profiling = false;
    std::string tensorrt_cache_path = "";
    int num_threads = 4;
};

/**
 * Abstract Edge Accelerator Base Class
 */
class IEvaAccelerator {
public:
    virtual ~IEvaAccelerator() = default;

    virtual bool initialize(const AcceleratorConfig& config) = 0;
    virtual std::vector<Detection> infer(const uint8_t* image_data, int width, int height, PixelFormat format) = 0;
    virtual void infer_batch(const std::vector<const uint8_t*>& batch_data,
                             int width, int height, PixelFormat format,
                             std::vector<std::vector<Detection>>& batch_results) = 0;
    virtual double get_last_latency_ms() const = 0;
    virtual const AcceleratorConfig& config() const = 0;
};

/**
 * NMS helper utility for postprocessing
 */
inline std::vector<Detection> non_maximum_suppression(
    std::vector<Detection>& detections, 
    float iou_threshold) {
    if (detections.empty()) return {};

    // Sort detections by descending confidence
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<Detection> selected;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        selected.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            // Only suppress if same class or agnostic
            if (detections[i].class_id == detections[j].class_id) {
                float iou = detections[i].bbox.iou(detections[j].bbox);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return selected;
}

} // namespace eva
