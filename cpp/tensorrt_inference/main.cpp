#include "common.h"
#include "sam2_tracker.h"
#include "spdlog.h"
#include <chrono>
#include <cstddef>
#include <exception>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

vector<cv::Scalar> colors = {
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

vector<int> readNumbersFromFile(const string &filename) {
  vector<int> numbers;          // 存储读取到的数字
  std::ifstream file(filename); // 打开文件

  // 检查文件是否成功打开
  if (!file.is_open()) {
    SPDLOG_ERROR("错误：无法打开文件 {}", filename);
    return numbers;
  }

  string line;
  // 读取文件的第一行（假设数字都在第一行）
  if (getline(file, line)) {
    // 将字符串转换为字符串流，方便按分隔符处理
    std::stringstream ss(line);
    string token;

    // 按逗号分割字符串
    while (getline(ss, token, ',')) {
      // 去除token中的空格
      size_t start = token.find_first_not_of(" \t");
      size_t end = token.find_last_not_of(" \t");
      if (start != string::npos && end != string::npos) {
        token = token.substr(start, end - start + 1);
      } else {
        continue; // 如果是空字符串，跳过
      }

      // 将处理后的字符串转换为整数
      try {
        int num = stoi(token);
        numbers.push_back(num);
      } catch (const std::exception &e) {
        SPDLOG_ERROR("无法将 {} 转换为数字", token);
      }
    }
  }

  file.close(); // 关闭文件
  return numbers;
}

int main(int argc, char **argv) {
  // 设置pattern，包含：时间、日志级别、文件名、函数名、行号、消息
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%# %!] %v");
  spdlog::set_level(spdlog::level::debug);

  string onnxModelPath = "../../onnx_model";
  string videoPath = "../../assets/1917.mp4";
  string labelPath = "../../first_frame_bbox.txt";

  if (argc > 1) {
    onnxModelPath = argv[1];
  }
  SPDLOG_INFO("Using onnx model path: {}", onnxModelPath);

  if (argc > 2) {
    videoPath = argv[2];
  }

  SPDLOG_INFO("Using video path: {}", videoPath);

  if (argc > 3) {
    labelPath = argv[3];
  }

  SPDLOG_INFO("Using label path: {}", labelPath);

  const vector<int> label = readNumbersFromFile(labelPath);
  if (label.size() != 4) {
    SPDLOG_ERROR("读取第一帧bbox位置失败");
    return -1;
  }
  const SAM2Config config = SAM2Config{};
  SAM2Tracker tracker(onnxModelPath, string{}, config);

  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened()) {
    SPDLOG_ERROR("Error: cannot open video file : {}", videoPath);
    return -1;
  }

  // 获取videoPath的文件名
  string videoName =
      videoPath.substr(videoPath.find_last_of("/\\") + 1); // 1917-1.mp4
  // videoName = videoName.substr(0, videoName.find_last_of(".")); // 1917-1

  SPDLOG_INFO("start tracking video: {}", videoName);
  cv::namedWindow(videoName, cv::WINDOW_NORMAL); // 创建以videoName为名的窗口

  // int numframes = cap.get(cv::CAP_PROP_FRAME_COUNT);    // 获取视频帧数
  int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);   // 获取视频帧宽度
  int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT); // 获取视频帧高度

  // cv VideoWriter_fourcc
  // cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4',
  // 'v'), 30, cv::Size(frameWidth, frameHeight));

  int frameIdx = 0;
  cv::Mat frame;
  cv::Mat predMask;
  auto start = std::chrono::high_resolution_clock::now();
  while (cap.read(frame)) {
    auto startStep = std::chrono::high_resolution_clock::now();
    SPDLOG_INFO("\033[32mframeIdx: {}\033[0m", frameIdx);
    if (frameIdx == 0) {
      // cv select roi
      // cv::Rect firstBbox = cv::selectROI(videoName, frame);
      cv::Rect firstBbox(label[0], label[1], label[2], label[3]);
      SPDLOG_INFO("first_bbox (x, y, w, h): {}, {}, {}, {}", firstBbox.x,
                  firstBbox.y, firstBbox.width, firstBbox.height);
      predMask = tracker.addFirstFrameBbox(frameIdx, frame, firstBbox);
    } else {
      predMask = tracker.trackStep(frameIdx, frame);
    }

    cv::resize(predMask, predMask, cv::Size(frameWidth, frameHeight));

    // 结果可视化与保存

    cv::Mat binaryMask;
    cv::threshold(
        predMask, binaryMask, 0.1, 1.0,
        cv::THRESH_BINARY); // 将输入图像灰度值大于阈值的点的灰度值设置为1，小于阈值的值设置为0
    binaryMask.convertTo(binaryMask, CV_8UC1,
                         255); // 将二值图像的灰度值由[0,1]变换到[0,255]

    cv::Mat maskImg = cv::Mat::zeros(
        frame.size(), CV_8UC3); // 生成一个与frame相同大小的全黑图像
    maskImg.setTo(
        colors[5],
        binaryMask); // 将maskImg中的binaryMask区域设置为colors[5]的颜色
    cv::addWeighted(frame, 1, maskImg, 0.2, 0,
                    frame); // 将frame和maskImg按照一定的权重相加

    cv::Rect bbox(0, 0, 0, 0);
    std::vector<cv::Point> nonZeroPoints;       // 非零点坐标
    cv::findNonZero(binaryMask, nonZeroPoints); // 根据二值图像获取非零点坐标
    if (!nonZeroPoints
             .empty()) { // 如果非零点坐标不为空，则获取包围非零点的最小矩形，否则bbox为(0,
                         // 0, 0, 0)
      bbox = cv::boundingRect(nonZeroPoints);
    }
    cv::rectangle(frame, bbox, colors[5], 2); // 在frame上绘制bbox

    auto durationStep = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - startStep);
    SPDLOG_DEBUG("step spent: {:.3f} ms", durationStep.count() * 1000);

    string text =
        "fps " +
        cv::format("%.1f", 1. / float(durationStep.count())); // 计算帧率
    cv::putText(frame, text, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 1,
                colors[5], 2, cv::LINE_AA); // 在frame上绘制帧率

    cv::imshow(videoName, frame);
    // cv::imwrite(std::to_string(frameIdx) + ".jpg", frame);
    // writer.write(frame);
    cv::waitKey(1);
    frameIdx++;
  }
  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  SPDLOG_DEBUG("total spent: {:.3f} ms", duration.count() * 1000);
  SPDLOG_DEBUG("average frame spent: {:.3f} ms",
               duration.count() * 1000 / frameIdx);
  // FPS
  SPDLOG_DEBUG("FPS: {}", frameIdx / duration.count());
  cap.release();
  return 0;
}
