#include <iostream>
#include <opencv2/opencv.hpp>
#include "sam2_tracker.h"
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
    SAM2Tracker tracker(modelPath, 0, 0);

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
    BoundedThreadSafeQueue<std::pair<FrameData, std::vector<float>>> frameQueue(maxQueueSize);
    BoundedThreadSafeQueue<std::pair<FrameData, std::vector<Ort::Value>>> encodedQueue(maxQueueSize);
    BoundedThreadSafeQueue<std::pair<FrameData, std::vector<Ort::Value>>> memoryAttentionQueue(maxQueueSize);
    BoundedThreadSafeQueue<std::pair<FrameData, cv::Mat>> processedQueue(maxQueueSize);
    // std::atomic<bool> stopFlag(false);

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat frame;
    cap >> frame;
    // cv::Rect firstBbox = cv::selectROI(videoName, frame);
    cv::Rect firstBbox(384, 304, 342, 316);
    std::cout << "first_bbox (x, y, w, h): " << firstBbox.x << ", " << firstBbox.y << ", " << firstBbox.width << ", " << firstBbox.height << std::endl;
    cv::Mat predMask = tracker.addFirstFrameBbox(0, frame, firstBbox);

    FrameData firstFrameData{0, frame.clone()};
    processedQueue.enqueue(std::make_pair(firstFrameData, predMask));

    // Producer thread: read frames from the video capture, preprocess them and put them into the frame queue
    std::thread producerThread([&]() {
        int frameIdx = 1;
        while (cap.read(frame))
        {
            auto readFrameBegin = std::chrono::high_resolution_clock::now();
            std::cout << "\033[32mframeIdx: " << frameIdx << "\033[0m" << std::endl; // green

            FrameData frameData{frameIdx, frame.clone()};
            std::vector<float> inputImage;
            tracker.preprocessImage(frame, inputImage);
            if (!frameQueue.enqueue(std::make_pair(frameData, inputImage)))
            {
                std::cout << "frameQueue is finished" << std::endl;
                break;
            }

            frameIdx++;

            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            auto producerSpend  = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readFrameBegin);
            std::cout << "producer spent: " << producerSpend.count() << " ms" << std::endl;
        }
        frameQueue.setFinished();
    });

    // Image encoder thread: read frames from the frame queue, run image encoder inference and put the results into the encoded queue
    std::thread imageEncoderThread([&]() {
        std::pair<FrameData, std::vector<float>> item;
        while (frameQueue.dequeue(item))
        {
            auto imageEncoderBegin = std::chrono::high_resolution_clock::now();
            std::cout << "imageEncoderThread frameIdx: " << item.first.frameIdx << std::endl;

            std::vector<Ort::Value> imageEncoderOutputTensors;
            tracker.imageEncoderInference(item.second, imageEncoderOutputTensors);
            if (!encodedQueue.enqueue(std::make_pair(item.first, std::move(imageEncoderOutputTensors))))
            {
                std::cout << "encodedQueue is finished" << std::endl;
                break;
            }

            auto imageEncoderSpend = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - imageEncoderBegin);
            std::cout << "imageEncoderThread spent: " << imageEncoderSpend.count() << " ms" << std::endl;
        }
        encodedQueue.setFinished();
    });

    // Memory attention thread: read frames from the encoded queue, run memory attention inference and put the results into the attention queue
    std::thread memoryAttentionThread([&]() {
        std::pair<FrameData, std::vector<Ort::Value>> item;
        while (encodedQueue.dequeue(item))
        {
            auto memoryAttentionBegin = std::chrono::high_resolution_clock::now();

            int frameIdx = item.first.frameIdx;
            // std::vector<Ort::Value> imageEncoderOutputTensors = item.second;
            //imageEncoderOutputTensors[2]在memoryAttentionInference和memoryEncoderInference中都会用到，所以保留副本
            // imageEncoderOutputTensors[2].GetTensorMutableData<float>();


            std::cout << "memoryAttentionThread frameIdx: " << frameIdx << std::endl;

            std::vector<Ort::Value> memoryAttentionOutputTensors;
            tracker.memoryAttentionInference(frameIdx, item.second, memoryAttentionOutputTensors);

            std::vector<Ort::Value> tensors;
            tensors.push_back(std::move(item.second[0]));
            tensors.push_back(std::move(item.second[1]));
            tensors.push_back(std::move(memoryAttentionOutputTensors[0]));
            if (!memoryAttentionQueue.enqueue(std::make_pair(item.first, std::move(tensors))))
            {
                std::cout << "memoryAttentionQueue is finished" << std::endl;
                break;
            }

            auto memoryAttentionSpend = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - memoryAttentionBegin);
            std::cout << "memoryAttentionThread spent: " << memoryAttentionSpend.count() << " ms" << std::endl;
        }
        memoryAttentionQueue.setFinished();
    });

    // Mask decoder and postprocess thread: read frames from the memory attention queue, run mask decoder inference, postprocess the results and put them into the processed queue
    std::thread decoderThread([&]() {
        std::pair<FrameData, std::vector<Ort::Value>> item;
        while (memoryAttentionQueue.dequeue(item))
        {
            auto decoderBegin = std::chrono::high_resolution_clock::now();

            int frameIdx = item.first.frameIdx;
            // std::vector<Ort::Value> tensors = item.second;
            std::cout << "decoderThread frameIdx: " << frameIdx << std::endl;

            std::vector<float> inputPoints = {0, 0, 0, 0};
            std::vector<int32_t> inputLabels = {-1, -1};

            std::vector<Ort::Value> maskDecoderOutputTensors;
            tracker.maskDecoderInference(inputPoints, inputLabels, item.second, item.second[2], maskDecoderOutputTensors);

            PostprocessResult result = tracker.postprocessOutput(maskDecoderOutputTensors);
            int bestIoUIndex = result.bestIoUIndex;
            float bestIouScore = result.bestIouScore;
            float kfScore = result.kfScore;

            auto lowResMultiMasks  = maskDecoderOutputTensors[0].GetTensorMutableData<float>();
            auto objPtrs           = maskDecoderOutputTensors[2].GetTensorMutableData<float>();
            auto objScoreLogits    = maskDecoderOutputTensors[3].GetTensorMutableData<float>();

            int lowResMaskHeight = 128;
            int lowResMaskWidth  = 128;

            auto lowResMask = lowResMultiMasks + bestIoUIndex * lowResMaskHeight * lowResMaskWidth;

            cv::Mat predMask(lowResMaskHeight, lowResMaskWidth, CV_32FC1, lowResMask);
            processedQueue.enqueue(std::make_pair(item.first, predMask));

            auto decoderSpend = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - decoderBegin);
            std::cout << "decoderThread spent: " << decoderSpend.count() << " ms" << std::endl;
        }
        processedQueue.setFinished();
    });

    // Display thread: read frames from the processed queue and display them
    std::thread displayThread([&]() {
        // int frameIdx = 0;
        std::pair<FrameData, cv::Mat> item;
        while (processedQueue.dequeue(item))
        {

            auto displayBegin = std::chrono::high_resolution_clock::now();

            int frameIdx = item.first.frameIdx;
            cv::Mat frame = item.first.frame;

            cv::Mat predMask = item.second;

            std::cout << "\033[33mdisplayThread frameIdx: " << frameIdx << "\033[0m" << std::endl; // yellow

            cv::resize(predMask, predMask, cv::Size(frameWidth, frameHeight));

            // 结果可视化与保存
            cv::Mat binaryMask;
            cv::threshold(predMask, binaryMask, 0.01, 1.0, cv::THRESH_BINARY);
            binaryMask.convertTo(binaryMask, CV_8UC1, 255);

            cv::Mat maskImg = cv::Mat::zeros(frame.size(), CV_8UC3);
            maskImg.setTo(colors[5], binaryMask);
            cv::addWeighted(frame, 1, maskImg, 0.3, 0, frame);

            // std::vector<int> bbox = {0, 0, 0, 0};
            cv::Rect bbox(0, 0, 0, 0);
            std::vector<cv::Point> nonZeroPoints;
            cv::findNonZero(binaryMask, nonZeroPoints);
            if (!nonZeroPoints.empty()) {
                bbox = cv::boundingRect(nonZeroPoints);
            }
            cv::rectangle(frame, bbox, colors[5], 2);

            // cv::imshow(videoName, frame);
            // cv::imwrite("../../output/c++_ort/" + std::to_string(frameIdx) + ".jpg", frame);
            // writer.write(frame);
            // cv::waitKey(1);
            // frameIdx++;

            auto displaySpend = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - displayBegin);
            std::cout << "displayThread spent: " << displaySpend.count() << " ms" << std::endl;
        }
    });

    // Join all threads
    producerThread.join();
    imageEncoderThread.join();
    memoryAttentionThread.join();
    decoderThread.join();
    displayThread.join();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "total spent: " << duration.count() << " ms" << std::endl;

    std::cout << "FPS: " << numframes / (duration.count() / 1000.0) << std::endl;
    std::cout << "average frame spent: " << duration.count() / numframes << " ms" << std::endl;

    cap.release();
    cv::destroyAllWindows();

    return 0;
}