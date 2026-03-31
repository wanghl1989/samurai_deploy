#ifndef SAM2_TRACKER_H
#define SAM2_TRACKER_H

#include "engine.h"

// #include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <omp.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "kalman_filter.h"

// // Utility method for checking if a file exists on disk
// inline bool doesFileExist(const std::string &name) {
//     std::ifstream f(name.c_str());
//     return f.good();
// }

struct MemoryBankEntry {
  std::vector<float> maskmem_features;
  std::vector<float> maskmem_pos_enc;
  std::vector<float> obj_ptr;
  float best_iou_score;
  float obj_score_logits;
  float kf_score;
};

struct PostprocessResult {
  int bestIoUIndex;
  float bestIouScore;
  float kfScore;
};

struct SAM2Config {
  std::string modelPath;
  bool useGPU;
  bool enableFp16;
};

class SAM2Tracker {
public:
  SAM2Tracker() {}
  SAM2Tracker(const std::string &onnxModelPath, const std::string &trtModelPath,
              const SAM2Config &config);
  ~SAM2Tracker() {}

  void loadNetwork(const std::string &modelPath, bool useGPU, bool enableFp16);

  cv::Mat addFirstFrameBbox(int frameIdx, const cv::Mat &firstFrame,
                            const cv::Rect &bbox);

  cv::Mat trackStep(int frameIdx, const cv::Mat &frame);

  void imageEncoderInference(
      const std::vector<float> &frame,
      std::vector<std::vector<float>> &imageEncoderOutputTensors);
  // void imageEncoderInference(const cv::cuda::GpuMat &frame,
  // std::vector<std::vector<float>> &imageEncoderOutputTensors); // CUDA
  // version - disabled

  void memoryAttentionInference(
      int frameIdx,
      const std::vector<std::vector<float>> &imageEncoderOutputTensors,
      std::vector<std::vector<float>> &memoryAttentionOutputTensors);

  void maskDecoderInference(
      const std::vector<float> &inputPoints,
      const std::vector<int32_t> &inputLabels,
      const std::vector<float> &pixFeatWithMem,
      const std::vector<float> &highResFeatures0,
      const std::vector<float> &highResFeatures1,
      std::vector<std::vector<float>> &maskDecoderOutputTensors);

  void memoryEncoderInference(
      const std::vector<float> &visionFeaturesTensor,
      const std::vector<float> &highResMasksForMemTensor,
      const std::vector<float> &objectScoreLogitsTensor, bool isMaskFromPts,
      std::vector<std::vector<float>> &memoryEncoderOutputTensors);

  void preprocessImage(const cv::Mat &src, std::vector<float> &dest);
  // void preprocessImage(const cv::Mat &inputImageBGR, cv::cuda::GpuMat &dest);
  // // CUDA version - disabled

  PostprocessResult postprocessOutput(
      const std::vector<std::vector<float>> &maskDecoderOutputTensors);

private:
  // TensorRT Engine
  std::vector<std::unique_ptr<Engine>> m_trtEngines;

  cv::Scalar _mean = cv::Scalar(0.485, 0.456, 0.406);
  cv::Scalar _std = cv::Scalar(0.229, 0.224, 0.225);
  int _imageSize = 512;
  int _videoWidth = 0;
  int _videoHeight = 0;

  // mkskmem_tpos_enc
  std::vector<float> _maskMemTposEnc;

  // Memory bank
  // std::map<int, MemoryBankEntry> _memoryBank;
  std::unordered_map<int, MemoryBankEntry> _memoryBank;

  // samurai parameters
  KalmanFilter _kf;
  Eigen::VectorXf _kfMean;
  Eigen::MatrixXf _kfCovariance;
  int _stableFrames = 0;

  int _stableFrameCount = 0;
  float _stableFramesThreshold = 15;
  float _stableIousThreshold = 0.3;
  float _kfScoreWeight = 0.25;
  float _memoryBankIouThreshold = 0.5;
  float _memoryBankObjScoreThreshold = 0.0;
  float _memoryBankKfScoreThreshold = 0.0;
  int _maxObjPtrsInEncoder = 16;
  int _numMaskmem = 7;
}; // class SAM2Tracker

#endif // SAM2_TRACKER_H
