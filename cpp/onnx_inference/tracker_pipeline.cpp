#include <iostream>
#include <opencv2/opencv.hpp>
#include "sam2_tracker_pipeline.h"
#include "bounded_thread_safe_queue.hpp"

std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 0, 255),     // red 0
    cv::Scalar(0, 255, 0),     // green 1
    cv::Scalar(255, 0, 0),     // blue 2
    cv::Scalar(255, 255, 0),   // cyan 3
    cv::Scalar(255, 0, 255),   // magenta 4
    cv::Scalar(0, 255, 255),   // yellow 5
    cv::Scalar(255, 255, 255), // white 6
    cv::Scalar(128, 128, 128), // gray 7
    cv::Scalar(128, 0, 0),
    cv::Scalar(128, 128, 0),
    cv::Scalar(0, 128, 0),
    cv::Scalar(128, 0, 128),
    cv::Scalar(0, 128, 128),
    cv::Scalar(0, 0, 128),
    cv::Scalar(0, 0, 0)
};

struct FrameData
{
    int frameIdx;
    cv::Mat frame;
};

int main(int argc, char **argv)
{
    std::string modelPath = "../../onnx_model";
    SAM2TrackerPipeline tracker(modelPath, 0, 0);

    std::string videoPath = "../../data/1917-1.mp4";
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: cannot open video file : " << videoPath << std::endl;
        return -1;
    }

    // 获取videoPath的文件名
    std::string videoName = videoPath.substr(videoPath.find_last_of("/\\") + 1); // 1917-1.mp4
    // videoName = videoName.substr(0, videoName.find_last_of(".")); // 1917-1
    // std::cout << "videoName: " << videoName << std::endl;
    std::cout << "start tracking video: " << videoName << std::endl;
    cv::namedWindow(videoName, cv::WINDOW_NORMAL);

    int numframes = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // cv VideoWriter_fourcc
    cv::VideoWriter writer("../../output/c++_ort/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(frameWidth, frameHeight));

    // Initialize queues with bounded capacity
    const size_t maxQueueSize = 10; // Double buffering
    // BoundedThreadSafeQueue<cv::Mat> frameQueue(maxQueueSize);
    // BoundedThreadSafeQueue<cv::Mat> resultQueue(maxQueueSize);
    // std::atomic<bool> stopFlag(false);

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat frame;
    cap >> frame;
    // cv::Rect firstBbox = cv::selectROI(videoName, frame);
    cv::Rect firstBbox(384, 304, 342, 316);
    std::cout << "first_bbox (x, y, w, h): " << firstBbox.x << ", " << firstBbox.y << ", " << firstBbox.width << ", " << firstBbox.height << std::endl;
    cv::Mat predMask = tracker.addFirstFrameBbox(0, frame, firstBbox);

    FrameData firstFrameData{0, frame.clone()};
    tracker.resultQueue.enqueue(std::make_pair(firstFrameData, predMask));

    tracker.startPipeline();

    // 异步视频读取
    std::thread reader([&]{
        cv::VideoCapture cap(videoPath);
        cv::Mat frame;
        while(cap.read(frame)) {
            tracker.preprocessQueue.enqueue(frame.clone());
        }
        tracker.preprocessQueue.set_finished();
    });

    // 结果处理线程
    BoundedThreadSafeQueue<cv::Mat> resultQueue(10);
    std::thread processor([&]{
        cv::Mat result;
        while(tracker.resultQueue.dequeue(result)) {
            cv::imshow("Result", result);
            cv::waitKey(1);
        }
    });

    reader.join();
    processor.join();
    tracker.stopPipeline();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "total spent: " << duration.count() << " ms" << std::endl;

    std::cout << "FPS: " << numframes / (duration.count() / 1000.0) << std::endl;
    std::cout << "average frame spent: " << duration.count() / numframes << " ms" << std::endl;

    cap.release();
    cv::destroyAllWindows();

    return 0;
}