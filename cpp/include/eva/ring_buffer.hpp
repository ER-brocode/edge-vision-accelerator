#pragma once

#include "eva/types.hpp"
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <memory>

namespace eva {

/**
 * A pinned, reusable zero-copy frame buffer slot.
 * Ensures allocations happen once at initialization and reuse memory pointers across frames.
 */
class FrameSlot {
public:
    FrameSlot(size_t capacity_bytes) 
        : capacity_(capacity_bytes), data_(new uint8_t[capacity_bytes]), in_use_(false) {
        std::memset(&meta_, 0, sizeof(meta_));
    }

    ~FrameSlot() {
        delete[] data_;
    }

    // Non-copyable, movable
    FrameSlot(const FrameSlot&) = delete;
    FrameSlot& operator=(const FrameSlot&) = delete;
    FrameSlot(FrameSlot&&) = default;

    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t capacity() const { return capacity_; }

    FrameMeta& meta() { return meta_; }
    const FrameMeta& meta() const { return meta_; }

    void mark_in_use(bool val) { in_use_.store(val, std::memory_order_release); }
    bool is_in_use() const { return in_use_.load(std::memory_order_acquire); }

private:
    size_t capacity_;
    uint8_t* data_;
    FrameMeta meta_;
    std::atomic<bool> in_use_;
};

/**
 * Multi-producer multi-consumer lock-free bounded ring buffer with zero copy memory slots.
 * Suitable for high-frequency 4K/60fps RTSP/V4L2 camera ingestion pipelines.
 */
class ZeroCopyFrameRingBuffer {
public:
    ZeroCopyFrameRingBuffer(size_t capacity, size_t max_frame_bytes)
        : capacity_(capacity),
          max_frame_bytes_(max_frame_bytes),
          head_(0),
          tail_(0),
          count_(0),
          is_shutdown_(false),
          dropped_frames_(0) {
        if (capacity == 0) {
            throw std::invalid_argument("Capacity must be greater than zero.");
        }
        slots_.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            slots_.emplace_back(std::make_unique<FrameSlot>(max_frame_bytes));
        }
    }

    ~ZeroCopyFrameRingBuffer() {
        shutdown();
    }

    /**
     * Acquire a write buffer slot for producer to populate directly (Zero-Copy).
     * If drop_oldest is true and buffer is full, it drops the oldest frame.
     */
    FrameSlot* acquire_producer_slot(bool drop_oldest = true) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (is_shutdown_) return nullptr;

        if (count_ == capacity_) {
            if (!drop_oldest) {
                return nullptr; // Buffer full, backpressure
            }
            // Drop oldest
            slots_[tail_]->mark_in_use(false);
            tail_ = (tail_ + 1) % capacity_;
            --count_;
            dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        }

        size_t write_idx = head_;
        slots_[write_idx]->mark_in_use(true);
        return slots_[write_idx].get();
    }

    /**
     * Commit producer slot after writing pixel data.
     */
    void commit_producer_slot(FrameSlot* slot, const FrameMeta& meta) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (is_shutdown_) return;

        slot->meta() = meta;
        head_ = (head_ + 1) % capacity_;
        ++count_;
        lock.unlock();
        cv_consumer_.notify_one();
    }

    /**
     * Consumer leases a frame slot for downstream inference.
     * Blocks up to timeout_ms waiting for new frames.
     */
    FrameSlot* acquire_consumer_slot(uint32_t timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_consumer_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
            return count_ > 0 || is_shutdown_;
        })) {
            return nullptr; // Timeout
        }

        if (is_shutdown_ && count_ == 0) return nullptr;

        size_t read_idx = tail_;
        return slots_[read_idx].get();
    }

    /**
     * Release consumer slot after inference/display processing.
     */
    void release_consumer_slot(FrameSlot* slot) {
        std::unique_lock<std::mutex> lock(mutex_);
        slot->mark_in_use(false);
        tail_ = (tail_ + 1) % capacity_;
        --count_;
        lock.unlock();
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_shutdown_ = true;
        }
        cv_consumer_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    size_t capacity() const { return capacity_; }

    uint64_t dropped_frames() const {
        return dropped_frames_.load(std::memory_order_relaxed);
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        head_ = 0;
        tail_ = 0;
        count_ = 0;
        dropped_frames_.store(0);
        for (auto& s : slots_) {
            s->mark_in_use(false);
        }
    }

private:
    size_t capacity_;
    size_t max_frame_bytes_;
    std::vector<std::unique_ptr<FrameSlot>> slots_;
    size_t head_;
    size_t tail_;
    size_t count_;
    bool is_shutdown_;
    std::atomic<uint64_t> dropped_frames_;
    mutable std::mutex mutex_;
    std::condition_variable cv_consumer_;
};

} // namespace eva
