#include <iostream>
#include <opencv2/opencv.hpp>
#include "sam2_tracker.h"

std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 0, 255),     // red 0
    cv::Scalar(0, 255, 0),     // green 1
    cv::Scalar(255, 0, 0),     // blue 2
    cv::Scalar(255, 255, 0),   // cyan 3
    cv::Scalar(255, 0, 255),   // magenta 4
    cv::Scalar(0, 255, 255),   // yellow 5
    cv::Scalar(255, 255, 255), // white 6
    cv::Scalar(128, 128, 128), // gray 7
    cv::Scalar(140, 140, 0),   // mars green 8
    cv::Scalar(167, 47, 0),    // klein blue 9
    cv::Scalar(39, 88, 232),   // hermes orange 10
    cv::Scalar(32, 0, 128),    // burgundy 11
    cv::Scalar(208, 216, 129), // tiffany blue 12
    cv::Scalar(9, 0, 76),      // bordeaux 13
    cv::Scalar(36, 220, 249),  // sennelier yellow 14
};

int main(int argc, char** argv) {
    std::string modelPath = "../../onnx_model";
    std::string videoPath = "../../assets/1917.mp4";

    if (argc > 1) {
        modelPath = argv[1];
    }
    std::cout << "Using model path: " << modelPath << std::endl;

    if (argc > 2) {
        videoPath = argv[2];
    }
    std::cout << "Using video path: " << videoPath << std::endl;

    SAM2Tracker tracker(modelPath, true, true);

    cv::VideoCapture cap(videoPath);
    if(!cap.isOpened()){
        std::cerr << "Error: cannot open video file : " << videoPath << std::endl;
        return -1;
    }

    // 获取videoPath的文件名
    std::string videoName = videoPath.substr(videoPath.find_last_of("/\\") + 1);  // 1917-1.mp4
    // videoName = videoName.substr(0, videoName.find_last_of(".")); // 1917-1
    // std::cout << "videoName: " << videoName << std::endl;
    std::cout << "start tracking video: " << videoName << std::endl;
    cv::namedWindow(videoName, cv::WINDOW_NORMAL); // 创建以videoName为名的窗口

    int numframes = cap.get(cv::CAP_PROP_FRAME_COUNT);    // 获取视频帧数
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);   // 获取视频帧宽度
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT); // 获取视频帧高度

    // cv VideoWriter_fourcc
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(frameWidth, frameHeight));

    int frameIdx = 0;
    cv::Mat frame;
    cv::Mat predMask;
    auto start = std::chrono::high_resolution_clock::now();
    while(cap.read(frame)){
        auto startStep = std::chrono::high_resolution_clock::now();
        std::cout << "\033[32mframeIdx: " << frameIdx << "\033[0m" << std::endl;
        if(frameIdx == 0){
            // cv select roi
            // cv::Rect firstBbox = cv::selectROI(videoName, frame);
            cv::Rect firstBbox(384, 304, 342, 316);
            std::cout << "first_bbox (x, y, w, h): " << firstBbox.x << ", " << firstBbox.y << ", " << firstBbox.width << ", " << firstBbox.height << std::endl;
            predMask = tracker.addFirstFrameBbox(frameIdx, frame, firstBbox);
        } else {
            predMask = tracker.trackStep(frameIdx, frame);
        }

        cv::resize(predMask, predMask, cv::Size(frameWidth, frameHeight));

        // 结果可视化与保存

        cv::Mat binaryMask;
        cv::threshold(predMask, binaryMask, 0.1, 1.0, cv::THRESH_BINARY); // 将输入图像灰度值大于阈值的点的灰度值设置为1，小于阈值的值设置为0
        binaryMask.convertTo(binaryMask, CV_8UC1, 255); // 将二值图像的灰度值由[0,1]变换到[0,255]

        cv::Mat maskImg = cv::Mat::zeros(frame.size(), CV_8UC3); // 生成一个与frame相同大小的全黑图像
        maskImg.setTo(colors[5], binaryMask); // 将maskImg中的binaryMask区域设置为colors[5]的颜色
        cv::addWeighted(frame, 1, maskImg, 0.2, 0, frame); // 将frame和maskImg按照一定的权重相加

        cv::Rect bbox(0, 0, 0, 0); 
        std::vector<cv::Point> nonZeroPoints; // 非零点坐标
        cv::findNonZero(binaryMask, nonZeroPoints); // 根据二值图像获取非零点坐标
        if (!nonZeroPoints.empty()) { // 如果非零点坐标不为空，则获取包围非零点的最小矩形，否则bbox为(0, 0, 0, 0)
            bbox = cv::boundingRect(nonZeroPoints); 
        }
        cv::rectangle(frame, bbox, colors[5], 2); // 在frame上绘制bbox

        auto durationStep = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startStep);
        std::cout << "step spent: " << durationStep.count() << " ms" << std::endl;

        std::string text = "fps " + cv::format("%.1f", 1000. / float(durationStep.count())); // 计算帧率
        cv::putText(frame, text, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 1, colors[5], 2, cv::LINE_AA); // 在frame上绘制帧率

        cv::imshow(videoName, frame);
        cv::imwrite(std::to_string(frameIdx) + ".jpg", frame);
        writer.write(frame);
        cv::waitKey(1);
        frameIdx++;
        
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "total spent: " << duration.count() << " ms" << std::endl;
    std::cout << "average frame spent: " << duration.count() / frameIdx << " ms" << std::endl;
    // FPS
    std::cout << "FPS: " << frameIdx / (duration.count() / 1000.0) << std::endl;
    cap.release();
    return 0;
}