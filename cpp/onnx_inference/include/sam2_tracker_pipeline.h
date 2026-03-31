#ifndef SAM2_TRACKER_PIPELINE_H
#define SAM2_TRACKER_PIPELINE_H

#include "sam2_tracker.h"
#include "bounded_thread_safe_queue.hpp"
#include "thread_pool.hpp"

struct FrameData
{
    int frameIdx;
    cv::Mat frame;
};

class SAM2TrackerPipeline : public SAM2Tracker {
public:
    void startPipeline();
    void stopPipeline();

private:
    // session pool
    std::vector<std::unique_ptr<Ort::Session>> imageEncoderPool;
    std::vector<std::unique_ptr<Ort::Session>> memoryAttentionPool;
    std::vector<std::unique_ptr<Ort::Session>> maskDecoderPool;
    std::vector<std::unique_ptr<Ort::Session>> memoryEncoderPool;

    // pipeline queues
    BoundedThreadSafeQueue<std::pair<FrameData, std::vector<float>>> frameQueue{maxQueueSize};
    BoundedThreadSafeQueue<std::pair<FrameData, std::vector<Ort::Value>>> encodedQueue{maxQueueSize};
    BoundedThreadSafeQueue<std::pair<FrameData, std::vector<Ort::Value>>> memoryAttentionQueue{maxQueueSize};
    BoundedThreadSafeQueue<std::pair<FrameData, cv::Mat>> processedQueue{maxQueueSize};

    // thread pool
    ThreadPool threadPool{4};
    std::atomic<bool> pipelineRunning{false};

    // mutex
    std::mutex pipelineMutex; // shared_mutex for read/write lock

    size_t maxQueueSize = 10;
};

#endif // SAM2_TRACKER_PIPELINE_H
