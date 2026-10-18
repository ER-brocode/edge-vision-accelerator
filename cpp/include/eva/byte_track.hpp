#pragma once

#include "eva/types.hpp"
#include <vector>
#include <deque>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace eva {

/**
 * 8-state Kalman Filter for 2D bounding boxes in format:
 * [x, y, a, h, vx, vy, va, vh]
 * where (x, y) is center, a is aspect ratio (w/h), h is height.
 */
class KalmanFilter2D {
public:
    KalmanFilter2D() {
        std::memset(mean_, 0, sizeof(mean_));
        std::memset(covariance_, 0, sizeof(covariance_));
    }

    void initiate(const Rect& bbox) {
        float cx = bbox.x + bbox.width / 2.0f;
        float cy = bbox.y + bbox.height / 2.0f;
        float a = bbox.width / std::max(1.0f, bbox.height);
        float h = bbox.height;

        mean_[0] = cx; mean_[1] = cy; mean_[2] = a; mean_[3] = h;
        mean_[4] = 0;  mean_[5] = 0;  mean_[6] = 0; mean_[7] = 0;

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                covariance_[i][j] = (i == j) ? ((i < 4) ? 1.0f : 10.0f) : 0.0f;
            }
        }
    }

    void predict() {
        // State transition: x' = x + vx, y' = y + vy, etc.
        for (int i = 0; i < 4; ++i) {
            mean_[i] += mean_[i + 4];
        }
        // Increase uncertainty
        for (int i = 0; i < 8; ++i) {
            covariance_[i][i] += (i < 4) ? 0.05f : 0.1f;
        }
    }

    void update(const Rect& measurement, float confidence = 1.0f) {
        float z[4];
        z[0] = measurement.x + measurement.width / 2.0f;
        z[1] = measurement.y + measurement.height / 2.0f;
        z[2] = measurement.width / std::max(1.0f, measurement.height);
        z[3] = measurement.height;

        // Kalman gain approximation tailored for real-time edge processing
        float k = 0.6f * std::clamp(confidence, 0.3f, 1.0f);
        for (int i = 0; i < 4; ++i) {
            float residual = z[i] - mean_[i];
            mean_[i] += k * residual;
            mean_[i + 4] += (k * 0.5f) * residual;
            covariance_[i][i] *= (1.0f - k * 0.5f);
        }
    }

    Rect to_rect() const {
        float cx = mean_[0];
        float cy = mean_[1];
        float a = std::max(0.01f, mean_[2]);
        float h = std::max(1.0f, mean_[3]);
        float w = a * h;

        return Rect{
            cx - w / 2.0f,
            cy - h / 2.0f,
            w,
            h
        };
    }

    const float* mean() const { return mean_; }

private:
    float mean_[8];
    float covariance_[8][8];
};

/**
 * ByteTrack Multi-Object Tracker Implementation in C++
 */
class ByteTrackerCpp {
public:
    ByteTrackerCpp(int max_lost_frames = 30, 
                   float high_thresh = 0.5f, 
                   float low_thresh = 0.1f, 
                   float match_thresh = 0.8f)
        : max_lost_frames_(max_lost_frames),
          high_thresh_(high_thresh),
          low_thresh_(low_thresh),
          match_thresh_(match_thresh),
          next_track_id_(1),
          frame_count_(0) {}

    struct TrackEntry {
        uint64_t track_id;
        KalmanFilter2D kf;
        Rect bbox;
        int class_id;
        float score;
        std::string label;
        TrackState state;
        int age;
        int hits;
        int time_since_update;
    };

    std::vector<TrackedObject> update(const std::vector<Detection>& detections) {
        ++frame_count_;

        // 1. Separate detections into high-score and low-score subsets
        std::vector<Detection> det_high;
        std::vector<Detection> det_low;
        for (const auto& det : detections) {
            if (det.confidence >= high_thresh_) {
                det_high.push_back(det);
            } else if (det.confidence >= low_thresh_) {
                det_low.push_back(det);
            }
        }

        // 2. Predict existing tracks
        for (auto& trk : tracks_) {
            trk.kf.predict();
            trk.bbox = trk.kf.to_rect();
            trk.age++;
            trk.time_since_update++;
        }

        // 3. First Association: High-score detections with active tracks
        std::vector<int> matched_track_idx;
        std::vector<int> matched_det_idx;
        associate(tracks_, det_high, match_thresh_, matched_track_idx, matched_det_idx);

        std::vector<bool> track_used(tracks_.size(), false);
        std::vector<bool> det_high_used(det_high.size(), false);

        for (size_t i = 0; i < matched_track_idx.size(); ++i) {
            int t_idx = matched_track_idx[i];
            int d_idx = matched_det_idx[i];
            track_used[t_idx] = true;
            det_high_used[d_idx] = true;

            tracks_[t_idx].kf.update(det_high[d_idx].bbox, det_high[d_idx].confidence);
            tracks_[t_idx].bbox = tracks_[t_idx].kf.to_rect();
            tracks_[t_idx].score = det_high[d_idx].confidence;
            tracks_[t_idx].hits++;
            tracks_[t_idx].time_since_update = 0;
            tracks_[t_idx].state = TrackState::Tracked;
        }

        // 4. Second Association: Remaining unmatched tracks with low-score detections
        std::vector<TrackEntry*> unmatched_tracks;
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (!track_used[i] && tracks_[i].state == TrackState::Tracked) {
                unmatched_tracks.push_back(&tracks_[i]);
            }
        }

        if (!unmatched_tracks.empty() && !det_low.empty()) {
            for (auto* trk : unmatched_tracks) {
                float best_iou = 0.0f;
                int best_det = -1;
                for (size_t j = 0; j < det_low.size(); ++j) {
                    float iou = trk->bbox.iou(det_low[j].bbox);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_det = static_cast<int>(j);
                    }
                }
                if (best_iou > 0.5f && best_det >= 0) {
                    trk->kf.update(det_low[best_det].bbox, det_low[best_det].confidence);
                    trk->bbox = trk->kf.to_rect();
                    trk->score = det_low[best_det].confidence;
                    trk->hits++;
                    trk->time_since_update = 0;
                    trk->state = TrackState::Tracked;
                }
            }
        }

        // 5. Initialize new tracks for unmatched high-confidence detections
        for (size_t i = 0; i < det_high.size(); ++i) {
            if (!det_high_used[i]) {
                TrackEntry new_trk;
                new_trk.track_id = next_track_id_++;
                new_trk.kf.initiate(det_high[i].bbox);
                new_trk.bbox = det_high[i].bbox;
                new_trk.class_id = det_high[i].class_id;
                new_trk.score = det_high[i].confidence;
                new_trk.label = det_high[i].label;
                new_trk.state = TrackState::New;
                new_trk.age = 1;
                new_trk.hits = 1;
                new_trk.time_since_update = 0;
                tracks_.push_back(new_trk);
            }
        }

        // 6. Prune lost / expired tracks
        std::vector<TrackEntry> active_tracks;
        std::vector<TrackedObject> output_tracks;

        for (auto& trk : tracks_) {
            if (trk.time_since_update > max_lost_frames_) {
                continue; // Prune
            }
            if (trk.time_since_update > 0) {
                trk.state = TrackState::Lost;
            }
            active_tracks.push_back(trk);

            if (trk.state == TrackState::Tracked || (trk.state == TrackState::New && trk.hits >= 2)) {
                TrackedObject obj;
                obj.track_id = trk.track_id;
                obj.bbox = trk.bbox;
                obj.class_id = trk.class_id;
                obj.confidence = trk.score;
                obj.label = trk.label;
                obj.state = trk.state;
                obj.age_frames = trk.age;
                obj.hits = trk.hits;
                obj.time_since_update = trk.time_since_update;
                const float* m = trk.kf.mean();
                obj.velocity = {m[4], m[5], m[6], m[7]};
                output_tracks.push_back(obj);
            }
        }
        tracks_ = std::move(active_tracks);

        return output_tracks;
    }

    void reset() {
        tracks_.clear();
        next_track_id_ = 1;
        frame_count_ = 0;
    }

private:
    void associate(const std::vector<TrackEntry>& trks,
                   const std::vector<Detection>& dets,
                   float threshold,
                   std::vector<int>& out_trk_idx,
                   std::vector<int>& out_det_idx) {
        if (trks.empty() || dets.empty()) return;

        std::vector<std::tuple<float, int, int>> matches;
        for (size_t t = 0; t < trks.size(); ++t) {
            for (size_t d = 0; d < dets.size(); ++d) {
                float iou = trks[t].bbox.iou(dets[d].bbox);
                if (iou >= threshold) {
                    matches.push_back({iou, static_cast<int>(t), static_cast<int>(d)});
                }
            }
        }

        std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) > std::get<0>(b);
        });

        std::vector<bool> trk_matched(trks.size(), false);
        std::vector<bool> det_matched(dets.size(), false);

        for (const auto& match : matches) {
            int t = std::get<1>(match);
            int d = std::get<2>(match);
            if (!trk_matched[t] && !det_matched[d]) {
                trk_matched[t] = true;
                det_matched[d] = true;
                out_trk_idx.push_back(t);
                out_det_idx.push_back(d);
            }
        }
    }

    int max_lost_frames_;
    float high_thresh_;
    float low_thresh_;
    float match_thresh_;
    uint64_t next_track_id_;
    uint64_t frame_count_;
    std::vector<TrackEntry> tracks_;
};

} // namespace eva
