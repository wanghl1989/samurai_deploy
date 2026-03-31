#include "sam2_tracker_pipeline.h"

void SAM2TrackerPipeline::startPipeline() {
    pipelineRunning = true;

    // image encoder thread
    threadPool.enqueue([this] {
        while (pipelineRunning) {
            std::pair<FrameData, std::vector<float>> item;
            // if (frameQueue.dequeue(item)) {
            //     std::vector<Ort::Value> imageEncoderOutputTensors;
            //     imageEncoderInference(item.second, imageEncoderOutputTensors);
            //     encodedQueue.enqueue(std::make_pair(item.first, std::move(imageEncoderOutputTensors)));
            // }
        }
    });

    // memory attention thread
    threadPool.enqueue([this] {
        while (pipelineRunning) {
            std::pair<FrameData, std::vector<Ort::Value>> item;
            if (encodedQueue.dequeue(item)) {
                // std::vector<Ort::Value> memoryAttentionOutputTensors;
                // memoryAttentionInference(item.first.frameIdx, item.second, memoryAttentionOutputTensors);
                // memoryAttentionQueue.enqueue(std::make_pair(item.first, std::move(memoryAttentionOutputTensors)));
            }
        }
    });

    // mask decoder and postprocess thread
    threadPool.enqueue([this] {
        while (pipelineRunning) {
            std::pair<FrameData, std::vector<Ort::Value>> item;
            if (memoryAttentionQueue.dequeue(item)) {
                // cv::Mat predMask;
                // maskDecoderInference(item.first.frameIdx, item.second, predMask);
                // processedQueue.enqueue(std::make_pair(item.first, predMask));
            }
        }
    });
}

void SAM2TrackerPipeline::stopPipeline() {
    pipelineRunning = false;
    frameQueue.setFinished();
    encodedQueue.setFinished();
    memoryAttentionQueue.setFinished();
    processedQueue.setFinished();
}