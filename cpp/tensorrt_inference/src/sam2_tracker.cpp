#include "sam2_tracker.h"
#include "Eigen/src/Core/Matrix.h"
#include "engine.h"
#include "spdlog.h"
#include <NvInferRuntimeBase.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// void printVector(const std::vector<std::vector<float>> &featureVectors)
// {
//     for (size_t i = 0; i < featureVectors.size(); i++) {
//         std::cout << "featureVectors[" << i << "] size: " <<
//         featureVectors[i].size() << std::endl; float sum =
//         std::accumulate(featureVectors[i].begin(), featureVectors[i].end(),
//         0.0f); std::cout << "  sum: " << sum << ", first 10 elements: "; for
//         (size_t j = 0; j < std::min(featureVectors[i].size(),
//         static_cast<size_t>(10)); j++) {
//             // std::cout << "  sum: " << sum << ", last 10 elements: ";
//             // for (size_t j = featureVectors[i].size() - 10; j <
//             featureVectors[i].size(); j++) {
//                 std::cout << featureVectors[i][j] << ", ";
//         }
//         std::cout << std::endl;
//     }

// }

SAM2Tracker::SAM2Tracker(const std::string &onnxModelPath,
                         const std::string &trtModelPath,
                         const SAM2Config &config) {

  // Create our TensorRT inference engine
  Options options{
      config.enableFp16 ? Precision::FP16 : Precision::FP32, "", 128, 1, 1, 0};

  std::vector<std::string> onnx_models{
      "image_encoder.onnx", "memory_attention.onnx", "mask_decoder.onnx",
      "memory_encoder.onnx"};
  std::vector<std::string> trt_models{
      "image_encoder.engine", "memory_attention.engine", "mask_decoder.engine",
      "memory_encoder.engine"};

  // Build the onnx model into a TensorRT engine file, cache the file to disk,
  // and then load the TensorRT engine file into memory. If the engine file
  // already exists on disk, this function will not rebuild but only load into
  // memory. The engine file is rebuilt any time the above Options are changed.
  if (!onnxModelPath.empty()) {
    // Build the ONNX model into a TensorRT engine file
    for (const auto &model_name : onnx_models) {
      auto trtEngine = std::make_unique<Engine>(options);
      auto succ = trtEngine->buildLoadNetwork(onnxModelPath + "/" + model_name);
      if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the "
                                   "TensorRT engine from ONNX model : " +
                                   model_name;
        SPDLOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
      }
      m_trtEngines.push_back(std::move(trtEngine));
    }
  } else if (!trtModelPath.empty()) { // If no ONNX model, check for TRT model
    // Load the TensorRT engine file directly
    for (const auto &model_name : trt_models) {
      auto trtEngine = std::make_unique<Engine>(options);
      auto succ = trtEngine->loadNetwork(trtModelPath + "/" + model_name);
      if (!succ) {
        const std::string errMsg =
            "Error: Unable to load the TensorRT engine from file : " +
            model_name;
        SPDLOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
      }
      m_trtEngines.push_back(std::move(trtEngine));
    }
  } else {
    std::string errMsg =
        "Error: Neither ONNX model nor TensorRT engine path provided.";
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }
}

void SAM2Tracker::imageEncoderInference(
    const std::vector<float> &frame,
    std::vector<std::vector<float>> &imageEncoderOutputTensors) {
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::vector<float>> inputs{frame};

  bool succ = m_trtEngines[0]->runInference(inputs, imageEncoderOutputTensors);
  if (!succ) {
    std::string errMsg = "Unable to run imageEncoder inference.";
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  SPDLOG_DEBUG("image_encoder spent: {:.3f} ms", duration.count() * 1000);
}

// void SAM2Tracker::imageEncoderInference(const cv::cuda::GpuMat &frame,
// std::vector<std::vector<float>> &imageEncoderOutputTensors) {
//     auto start = std::chrono::high_resolution_clock::now();
//
//     // Convert to format expected by our inference engine
//     // The reason for the strange format is because it supports models with
//     multiple inputs as well as batching
//     // In our case though, the model only has a single input and we are using
//     a batch size of 1. std::vector<cv::Mat> input{std::move(frame)};
//     std::vector<std::vector<cv::Mat>> inputs{std::move(input)};
//
//     bool succ = m_trtEngines[0]->runInference(inputs,
//     imageEncoderOutputTensors); if (!succ) {
//         std::string errMsg = "Unable to run imageEncoder inference.";
//         SPDLOG_ERROR(errMsg);
//         throw std::runtime_error(errMsg);
//     }
//
//     auto duration =
//     std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
//     start); SPDLOG_DEBUG("image_encoder spent: {:.3f} ms", duration.count() *
//     1000);
// }

void SAM2Tracker::memoryAttentionInference(
    int frameIdx,
    const std::vector<std::vector<float>> &imageEncoderOutputTensors,
    std::vector<std::vector<float>> &memoryAttentionOutputTensors) {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<float> memmaskFeatures = _memoryBank[0].maskmem_features;
  std::vector<float> memmaskPosEncs = _memoryBank[0].maskmem_pos_enc;
  std::vector<float> objectPtrs = _memoryBank[0].obj_ptr;
  // std::cout << "memmaskFeatures.size(): " << memmaskFeatures.size() <<
  // std::endl; std::cout << "memmaskPosEncs.size(): " << memmaskPosEncs.size()
  // << std::endl; std::cout << "objectPtrs.size(): " << objectPtrs.size() <<
  // std::endl;

  std::vector<int> validIndices;
  if (frameIdx > 1) {
    for (int i = frameIdx - 1; i > 0; i--) {
      float iouScore = _memoryBank[i].best_iou_score;
      float objScore = _memoryBank[i].obj_score_logits;
      float kfScore = _memoryBank[i].kf_score;
      if (validIndices.size() >= _maxObjPtrsInEncoder - 1) {
        break;
      }
      if (iouScore > _memoryBankIouThreshold &&
          objScore > _memoryBankObjScoreThreshold &&
          (kfScore > _memoryBankKfScoreThreshold)) {
        validIndices.insert(validIndices.begin(), i);
      }
    }
  }
  // std::cout << "validIndices.size(): " << validIndices.size() << std::endl;
  // std::cout << "validIndices : ";
  // for (int i = 0; i < validIndices.size(); i++) {
  //     std::cout << validIndices[i] << ", ";
  // }
  // std::cout << std::endl;

  size_t maskmemFeaturesSize = m_trtEngines[3]->getOutputElementCount().at(0);
  size_t maskmemPosEncSize = m_trtEngines[3]->getOutputElementCount().at(1);

  auto maskDecoderOutputNodeDims = m_trtEngines[2]->getOutputDims();
  size_t objPtrSize = maskDecoderOutputNodeDims.at(2).d[2]; // 1*3*256
  // std::cout << "maskmemFeaturesSize: " << maskmemFeaturesSize << std::endl;
  // std::cout << "maskmemPosEncSize: " << maskmemPosEncSize << std::endl;
  // std::cout << "objPtrSize: " << objPtrSize << std::endl;
  size_t memmaskFeaturesNum =
      std::min(static_cast<size_t>(_numMaskmem), validIndices.size() + 1);
  size_t memmaskPosEncNum =
      std::min(static_cast<size_t>(_numMaskmem), validIndices.size() + 1);

  memmaskFeatures.reserve(
      maskmemFeaturesSize *
      memmaskFeaturesNum); // 最近 num_maskmem-1 帧 + 0帧 的maskmem_features
  memmaskPosEncs.reserve(
      maskmemPosEncSize *
      memmaskPosEncNum); // 最近 num_maskmem-1 帧 + 0帧 的maskmem_pos_enc

  // std::cout << "memmaskFeatures idx: ";
  int validIndicesSize = validIndices.size();
  for (int i = validIndicesSize - _numMaskmem + 1; i < validIndicesSize;
       i++) { // 最近 num_maskmem-1 帧
    if (i < 0) {
      continue;
    }
    // std::cout << i << ": " << validIndices[i] << ", ";

    int prevFrameIdx = validIndices[i];
    MemoryBankEntry mem = _memoryBank[prevFrameIdx];
    memmaskFeatures.insert(memmaskFeatures.end(), mem.maskmem_features.begin(),
                           mem.maskmem_features.end());
    memmaskPosEncs.insert(memmaskPosEncs.end(), mem.maskmem_pos_enc.begin(),
                          mem.maskmem_pos_enc.end());
  }
  // std::cout << std::endl;

  auto start2 = std::chrono::high_resolution_clock::now();
  // std::cout << "memmaskFeaturesNum: " << memmaskFeaturesNum << std::endl;
  nvinfer1::Dims tposEncDims = maskDecoderOutputNodeDims.at(4);

  // #pragma omp parallel for
  for (int i = 1; i < memmaskFeaturesNum; i++) {
    int start = (memmaskFeaturesNum - i) * maskmemPosEncSize;
    int end = start + maskmemPosEncSize;
    // #pragma omp parallel for
    for (int j = start; j < end; j++) {
      memmaskPosEncs[j] += _maskMemTposEnc.at((i - 1) * tposEncDims.d[3] +
                                              (j % tposEncDims.d[3]));
    }
  }
  // #pragma omp parallel for
  for (int i = 0; i < maskmemFeaturesSize; i++) {
    memmaskPosEncs[i] += _maskMemTposEnc.at(
        (tposEncDims.d[0] - 1) * tposEncDims.d[3] + (i % tposEncDims.d[3]));
  }
  auto duration2 = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start2);
  SPDLOG_DEBUG("memmaskPosEncs spent: {:.3f} ms", duration2.count() * 1000);

  std::vector<int> objPosEnc = {frameIdx};
  // std::cout << "objPosEnc : ";
  for (int i = 1; i < frameIdx; i++) {
    // std::cout << i << ", ";
    if (objPosEnc.size() >= _maxObjPtrsInEncoder) {
      break;
    }
    objPosEnc.push_back(i);
  }
  // std::cout << std::endl;

  objectPtrs.reserve(
      objPtrSize *
      objPosEnc.size()); // 最近 maxObjPtrsInEncoder-1 帧 + 0帧 的obj_ptr
  // std::cout << "objectPtrs : ";
  for (int i = frameIdx - 1; i > 0; i--) {
    // std::cout << i << ", ";
    if (objectPtrs.size() >= _maxObjPtrsInEncoder * objPtrSize) {
      break;
    }
    MemoryBankEntry mem = _memoryBank[i];
    objectPtrs.insert(objectPtrs.end(), mem.obj_ptr.begin(), mem.obj_ptr.end());
  }
  // std::cout << std::endl;

  // std::cout << "memmaskFeaturesNum: " << memmaskFeaturesNum << std::endl;
  // std::cout << "memmaskPosEncNum: " << memmaskPosEncNum << std::endl;
  // std::cout << "validIndices.size(): " << validIndices.size() << std::endl;
  // std::cout << "objectPtrs.size(): " << objectPtrs.size() / objPtrSize <<
  // std::endl; std::cout << "objPosEnc.size(): " << objPosEnc.size() <<
  // std::endl; std::cout << "memmaskFeatures.size(): " <<
  // memmaskFeatures.size() / maskmemFeaturesSize << ", " <<
  // memmaskFeatures.capacity() / maskmemFeaturesSize << std::endl; std::cout <<
  // "memmaskPosEncs.size(): " << memmaskPosEncs.size() / maskmemPosEncSize<< ",
  // " << memmaskPosEncs.capacity() / maskmemPosEncSize << std::endl;

  std::vector<std::vector<float>> inputValues;
  // inputValues.reserve(6);
  inputValues.push_back(
      std::move(imageEncoderOutputTensors[2])); // lowResFeatures
  inputValues.push_back(
      std::move(imageEncoderOutputTensors[3])); // visionPosEmbedding
  inputValues.push_back(std::move(memmaskFeatures));
  inputValues.push_back(std::move(memmaskPosEncs));
  inputValues.push_back(std::move(objectPtrs));

  std::vector<float> float_objPosEnc(objPosEnc.begin(), objPosEnc.end());
  inputValues.push_back(std::move(float_objPosEnc));

  auto memoryAttentionInputDims = m_trtEngines[1]->getInputDims();
  auto memmaskFeaturesDims = memoryAttentionInputDims.at(2);
  auto memmaskPosEncsDims = memoryAttentionInputDims.at(3);
  auto objectPtrsDims = memoryAttentionInputDims.at(4);
  auto objPosEncDims = memoryAttentionInputDims.at(5);

  memmaskFeaturesDims.d[1] = memmaskFeaturesNum;
  memmaskPosEncsDims.d[1] = memmaskPosEncNum;
  objectPtrsDims.d[0] = objPosEnc.size();
  objPosEncDims.d[0] = objPosEnc.size();
  m_trtEngines[1]->setInputDims(2, memmaskFeaturesDims);
  m_trtEngines[1]->setInputDims(3, memmaskPosEncsDims);
  m_trtEngines[1]->setInputDims(4, objectPtrsDims);
  m_trtEngines[1]->setInputDims(5, objPosEncDims);

  bool succ =
      m_trtEngines[1]->runInference(inputValues, memoryAttentionOutputTensors);
  if (!succ) {
    std::string errMsg = "Unable to run memoryAttention inference.";
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  SPDLOG_DEBUG("memory_attention spent: {:.3f} ms", duration.count() * 1000);
}

void SAM2Tracker::maskDecoderInference(
    const std::vector<float> &inputPoints,
    const std::vector<int32_t> &inputLabels,
    const std::vector<float> &pixFeatWithMem,
    const std::vector<float> &highResFeatures0,
    const std::vector<float> &highResFeatures1,
    std::vector<std::vector<float>> &maskDecoderOutputTensors) {
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<float> float_vec(inputLabels.begin(), inputLabels.end());

  std::vector<std::vector<float>> inputValues;
  inputValues.push_back(std::move(inputPoints));
  inputValues.push_back(std::move(float_vec));
  inputValues.push_back(std::move(pixFeatWithMem));
  inputValues.push_back(std::move(highResFeatures0));
  inputValues.push_back(std::move(highResFeatures1));

  bool succ =
      m_trtEngines[2]->runInference(inputValues, maskDecoderOutputTensors);
  if (!succ) {
    std::string errMsg = "Unable to run maskDecoder inference.";
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  SPDLOG_DEBUG("mask_decoder spent: {:.3f} ms", duration.count() * 1000);
}

void SAM2Tracker::memoryEncoderInference(
    const std::vector<float> &visionFeaturesTensor,
    const std::vector<float> &highResMasksForMemTensor,
    const std::vector<float> &objectScoreLogitsTensor, bool isMaskFromPts,
    std::vector<std::vector<float>> &memoryEncoderOutputTensors) {
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<float> isMaskFromPtsTensor{static_cast<float>(isMaskFromPts)};

  std::vector<std::vector<float>> inputValues;
  inputValues.push_back(std::move(visionFeaturesTensor)); // lowResFeatures
  inputValues.push_back(
      std::move(highResMasksForMemTensor)); // highResMasksForMem
  inputValues.push_back(
      std::move(objectScoreLogitsTensor)); // objectScoreLogits
  inputValues.push_back(std::move(isMaskFromPtsTensor));

  bool succ =
      m_trtEngines[3]->runInference(inputValues, memoryEncoderOutputTensors);
  if (!succ) {
    std::string errMsg = "Unable to run memoryEncoder inference.";
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  SPDLOG_DEBUG("memory_encoder spent: {:.3f} ms", duration.count() * 1000);
}

cv::Mat SAM2Tracker::addFirstFrameBbox(int frameIdx, const cv::Mat &firstFrame,
                                       const cv::Rect &bbox) {

  _videoWidth = static_cast<int>(firstFrame.cols);
  _videoHeight = static_cast<int>(firstFrame.rows);

  // cv::cuda::GpuMat inputImage;
  // preprocessImage(firstFrame, inputImage);

  // // 1) image_encoder 推理
  // std::vector<std::vector<float>> imageEncoderOutputFeatureVectors;
  // std::vector<Ort::Value> imageEncoderOutputTensors;
  // imageEncoderInference(inputImage, imageEncoderOutputFeatureVectors);

  std::vector<float> inputImage;
  preprocessImage(firstFrame, inputImage);

  // 1) image_encoder 推理
  std::vector<std::vector<float>> imageEncoderOutputs;
  imageEncoderInference(inputImage, imageEncoderOutputs);

  // 2) mask_decoder 推理
  std::vector<float> inputPoints = {static_cast<float>(bbox.x),
                                    static_cast<float>(bbox.y),
                                    static_cast<float>(bbox.x + bbox.width),
                                    static_cast<float>(bbox.y + bbox.height)};
  inputPoints[0] = (inputPoints[0] / firstFrame.cols) * _imageSize;
  inputPoints[1] = (inputPoints[1] / firstFrame.rows) * _imageSize;
  inputPoints[2] = (inputPoints[2] / firstFrame.cols) * _imageSize;
  inputPoints[3] = (inputPoints[3] / firstFrame.rows) * _imageSize;

  std::vector<int32_t> boxLabels = {2, 3};

  std::vector<std::vector<float>> maskDecoderOutputs;
  maskDecoderInference(inputPoints, boxLabels, imageEncoderOutputs[4],
                       imageEncoderOutputs[0], imageEncoderOutputs[1],
                       maskDecoderOutputs);

  PostprocessResult result = postprocessOutput(maskDecoderOutputs);
  int bestIoUIndex = result.bestIoUIndex;
  float bestIouScore = result.bestIouScore;
  float kfScore = result.kfScore;

  auto lowResMultiMasks = maskDecoderOutputs[0].data();
  // auto ious              = maskDecoderOutputs[1]data();
  auto objPtrs = maskDecoderOutputs[2].data();
  auto objScoreLogits = maskDecoderOutputs[3].data();
  auto maskMemTposEncTmp = maskDecoderOutputs[4].data(); // 7*1*1*64
  _maskMemTposEnc = std::vector<float>(
      maskMemTposEncTmp,
      maskMemTposEncTmp + m_trtEngines[2]->getOutputElementCount().at(4));
  // // check _maskMemTposEnc
  // for (int i = 0; i < _maskDecoderOutputNodeDims[4][0]; i++) { // 7
  //     std::cout << "maskMemTposEnc[" << i << "]: ";
  //     for (int j = 0; j < _maskDecoderOutputNodeDims[4][3]; j++) { // 64
  //         std::cout << _maskMemTposEnc[i * _maskDecoderOutputNodeDims[4][3] +
  //         j] << ", ";
  //     }
  //     std::cout << std::endl;
  // }
  auto maskDecoderOutputNodeDims = m_trtEngines[2]->getOutputDims();
  int lowResMaskHeight = maskDecoderOutputNodeDims.at(0).d[2];
  int lowResMaskWidth = maskDecoderOutputNodeDims.at(0).d[3];
  auto lowResMask =
      lowResMultiMasks + bestIoUIndex * lowResMaskHeight * lowResMaskWidth;
  // auto highResMask = highResMultiMasks + bestIoUIndex * _imageSize *
  // _imageSize;

  cv::Mat predMask(lowResMaskHeight, lowResMaskWidth, CV_32FC1, lowResMask);
  // cv::Mat predMask(_videoHeight, _videoWidth, CV_32FC1, lowResMask);

  // 3) memory_encoder 推理
  bool isMaskFromPts = frameIdx == 0;

  cv::Mat highResMaskMat;
  cv::resize(predMask, highResMaskMat, cv::Size(_imageSize, _imageSize));

  // std::vector<float> highResMask((float*)highResMaskMat.data,
  // (float*)highResMaskMat.data + highResMaskMat.total());
  std::vector<float> highResMask(highResMaskMat.begin<float>(),
                                 highResMaskMat.end<float>());
  // std::vector<int64_t> highResMaskDims = {1, 1, _imageSize, _imageSize};

  std::vector<std::vector<float>> memoryEncoderOutputs;
  memoryEncoderInference(imageEncoderOutputs[2], highResMask,
                         maskDecoderOutputs[3], isMaskFromPts,
                         memoryEncoderOutputs);

  auto maskmemFeatures = memoryEncoderOutputs.at(0).data();
  auto maskmemPosEnc = memoryEncoderOutputs.at(1).data();

  // 4) save memory bank
  size_t maskmemFeaturesSize = m_trtEngines[3]->getOutputElementCount().at(
      0); // memoryEncoderOutputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
          // // 262144=4096*1*64
  size_t maskmemPosEncSize = m_trtEngines[3]->getOutputElementCount().at(
      1); // memoryEncoderOutputTensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
          // // 262144=4096*1*64
  size_t objPtrSize = maskDecoderOutputNodeDims.at(2).d[2]; // 1,3,256
  // std::cout << "maskmemFeaturesSize: " << maskmemFeaturesSize << std::endl;
  // std::cout << "maskmemPosEncSize: " << maskmemPosEncSize << std::endl;
  // std::cout << "objPtrSize: " << objPtrSize << std::endl;

  MemoryBankEntry entry;
  entry.maskmem_features = std::vector<float>(
      maskmemFeatures, maskmemFeatures + maskmemFeaturesSize);
  entry.maskmem_pos_enc =
      std::vector<float>(maskmemPosEnc, maskmemPosEnc + maskmemPosEncSize);
  entry.obj_ptr = std::vector<float>(objPtrs + bestIoUIndex * objPtrSize,
                                     objPtrs + (bestIoUIndex + 1) * objPtrSize);
  entry.best_iou_score = bestIouScore;
  entry.obj_score_logits = objScoreLogits[0];
  entry.kf_score = kfScore;

  _memoryBank[frameIdx] = entry;

  return predMask;
}

cv::Mat SAM2Tracker::trackStep(int frameIdx, const cv::Mat &frame) {
  std::vector<float> inputImage;
  preprocessImage(frame, inputImage);

  // 1) image_encoder 推理
  std::vector<std::vector<float>> imageEncoderOutputs;
  imageEncoderInference(inputImage, imageEncoderOutputs);

  auto lowResFeatures = imageEncoderOutputs
      [2]; // 下面要用到两次，保留副本，因为使用std::move后，原来的数据所有权转移

  // 2) memory_attention 推理
  std::vector<std::vector<float>> memoryAttentionOutputs;
  memoryAttentionInference(frameIdx, imageEncoderOutputs,
                           memoryAttentionOutputs);

  // 3) mask_decoder 推理
  std::vector<float> inputPoints = {0, 0, 0, 0};
  std::vector<int32_t> inputLabels = {-1, -1};

  std::vector<std::vector<float>> maskDecoderOutputs;
  maskDecoderInference(inputPoints, inputLabels, memoryAttentionOutputs[0],
                       imageEncoderOutputs[0], imageEncoderOutputs[1],
                       maskDecoderOutputs);

  PostprocessResult result = postprocessOutput(maskDecoderOutputs);
  int bestIoUIndex = result.bestIoUIndex;
  float bestIouScore = result.bestIouScore;
  float kfScore = result.kfScore;

  auto lowResMultiMasks = maskDecoderOutputs.at(0).data();
  // auto ious              = maskDecoderOutputs.at(1).data();
  auto objPtrs = maskDecoderOutputs.at(2).data();
  auto objScoreLogits = maskDecoderOutputs.at(3).data();

  auto maskDecoderOutputNodeDims = m_trtEngines[2]->getOutputDims();
  int lowResMaskHeight = maskDecoderOutputNodeDims.at(0).d[2];
  int lowResMaskWidth = maskDecoderOutputNodeDims.at(0).d[3];

  auto lowResMask =
      lowResMultiMasks + bestIoUIndex * lowResMaskHeight * lowResMaskWidth;
  // auto highResMask = highResMultiMasks + bestIoUIndex * _imageSize *
  // _imageSize;

  cv::Mat predMask(lowResMaskHeight, lowResMaskWidth, CV_32FC1, lowResMask);
  // cv::Mat predMask(_videoHeight, _videoWidth, CV_32FC1, lowResMask);

  // 4) memory_encoder 推理
  bool isMaskFromPts = frameIdx == 0;

  cv::Mat highResMaskMat;
  cv::resize(predMask, highResMaskMat, cv::Size(_imageSize, _imageSize));

  // std::vector<float> highResMask((float*)highResMaskMat.data,
  // (float*)highResMaskMat.data + highResMaskMat.total());
  std::vector<float> highResMask(highResMaskMat.begin<float>(),
                                 highResMaskMat.end<float>());
  // std::vector<int64_t> highResMaskDims = {1, 1, _imageSize, _imageSize};

  std::vector<std::vector<float>> memoryEncoderOutputs;
  memoryEncoderInference(lowResFeatures, highResMask, maskDecoderOutputs[3],
                         isMaskFromPts, memoryEncoderOutputs);

  auto maskmemFeatures = memoryEncoderOutputs.at(0).data();
  auto maskmemPosEnc = memoryEncoderOutputs.at(1).data();

  // 5) save memory bank
  size_t maskmemFeaturesSize = m_trtEngines[3]->getOutputElementCount().at(
      0); // memoryEncoderOutputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
          // // 262144=4096*64
  size_t maskmemPosEncSize = m_trtEngines[3]->getOutputElementCount().at(
      1); // memoryEncoderOutputTensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
          // // 262144=4096*64
  size_t objPtrSize = maskDecoderOutputNodeDims.at(2).d[2]; // 1,3,256
  // std::cout << "maskmemFeaturesSize: " << maskmemFeaturesSize << std::endl;
  // std::cout << "maskmemPosEncSize: " << maskmemPosEncSize << std::endl;
  // std::cout << "objPtrSize: " << objPtrSize << std::endl;

  MemoryBankEntry entry;
  entry.maskmem_features = std::vector<float>(
      maskmemFeatures, maskmemFeatures + maskmemFeaturesSize);
  entry.maskmem_pos_enc =
      std::vector<float>(maskmemPosEnc, maskmemPosEnc + maskmemPosEncSize);
  entry.obj_ptr = std::vector<float>(objPtrs + bestIoUIndex * objPtrSize,
                                     objPtrs + (bestIoUIndex + 1) * objPtrSize);
  entry.best_iou_score = bestIouScore;
  entry.obj_score_logits = objScoreLogits[0];
  entry.kf_score = kfScore;

  // if (_memoryBank.size() >= _maxObjPtrsInEncoder) {
  //     // 保留第一帧的数据,清除第二帧的数据
  //     auto firstIt = _memoryBank.begin();
  //     auto secondIt = std::next(firstIt);
  //     _memoryBank.erase(secondIt);
  // }
  // 改为unorder_map, 插入和删除操作为O(1)
  if (_memoryBank.size() >= _maxObjPtrsInEncoder) {
    int eraseIdx = frameIdx - _maxObjPtrsInEncoder + 1;
    _memoryBank.erase(eraseIdx);
  }
  _memoryBank[frameIdx] = entry;

  return predMask;
}

// void SAM2Tracker::preprocessImage(const cv::Mat& inputImageBGR, cv::Mat&
// dest) {
//     auto start = std::chrono::high_resolution_clock::now();
//
//     // The model expects RGB input
//     cv::cvtColor(inputImageBGR, dest, cv::COLOR_BGR2RGB);
//     cv::resize(dest, dest, cv::Size(_imageSize, _imageSize));
//     dest.convertTo(dest, CV_32FC3);
//
//     auto duration =
//     std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
//     start); SPDLOG_DEBUG("preprocessImage spent: {:.3f} ms", duration.count()
//     * 1000);
// }

void SAM2Tracker::preprocessImage(const cv::Mat &src,
                                  std::vector<float> &dest) {
  auto start = std::chrono::high_resolution_clock::now();

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(_imageSize, _imageSize));
  cv::Mat rgbImage;
  cv::cvtColor(resized, rgbImage, cv::COLOR_BGR2RGB); // 转换为RGB
  rgbImage.convertTo(rgbImage, CV_32FC3);             // 转换为float
  dest.assign((float *)rgbImage.data,
              (float *)rgbImage.data + rgbImage.total() * rgbImage.channels());

  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  SPDLOG_DEBUG("preprocessImage spent: {:.3f} ms", duration.count() * 1000);
}

PostprocessResult SAM2Tracker::postprocessOutput(
    const std::vector<std::vector<float>> &maskDecoderOutputTensors) {
  // maskDecoderOutputTensors[0] : lowResMultiMasks，(3, videoW, videoH)
  // maskDecoderOutputTensors[1] : highResMultiMasks, (3, _imageSize,
  // _imageSize) maskDecoderOutputTensors[2] : ious, (3)
  // maskDecoderOutputTensors[3] : objPtr, (3, 256)
  // maskDecoderOutputTensors[4] : objScoreLogits, (1)
  // maskDecoderOutputTensors[5] : maskMemTposEnc, (7, 64)
  auto start = std::chrono::high_resolution_clock::now();

  auto lowResMultiMasks = maskDecoderOutputTensors.at(0).data();
  auto ious = maskDecoderOutputTensors.at(1).data();
  auto objPtr = maskDecoderOutputTensors.at(2).data();
  auto objScoreLogits = maskDecoderOutputTensors.at(3).data();
  auto maskMemTposEnc = maskDecoderOutputTensors.at(4).data();

  int numMasks = m_trtEngines[2]
                     ->getOutputDims()
                     .at(1)
                     .d[1]; //_maskDecoderOutputNodeDims[1][1];

  // // print ious
  // std::cout << "ious: ";
  // for (int i = 0; i < numMasks; i++) {
  //     std::cout << ious[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "objScoreLogits: " << *objScoreLogits << std::endl;

#if 0 // sam2 选择ious最高的index
    int bestIoUIndex = std::distance(ious, std::max_element(ious, ious + numMasks));
    float bestIouScore = ious[bestIoUIndex];
    float kfScore = 1.0;
    // std::cout << "bestIoUIndex: " << bestIoUIndex << std::endl;
    // std::cout << "bestIouScore: " << bestIouScore << std::endl;

#else // samurai, 加入卡尔曼滤波预测
  int bestIoUIndex;
  float bestIouScore;
  float kfScore = 1.0;

  if ((_kfMean.size() == 0 && _kfCovariance.size() == 0) ||
      _stableFrameCount == 0) {
    bestIoUIndex = std::distance(ious, std::max_element(ious, ious + numMasks));
    bestIouScore = ious[bestIoUIndex];
    // auto highResMask = highResMultiMasks + bestIoUIndex * _imageSize *
    // _imageSize; cv::Mat predMask(_imageSize, _imageSize, CV_32FC1,
    // highResMask);
    int lowResMaskSize = m_trtEngines[2]
                             ->getOutputDims()
                             .at(0)
                             .d[2]; //_maskDecoderOutputNodeDims[0][2];
    auto lowResMask =
        lowResMultiMasks + bestIoUIndex * lowResMaskSize * lowResMaskSize;
    cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1,
                     const_cast<float *>(lowResMask));

    cv::Mat binaryMask;
    cv::threshold(predMask, binaryMask, 0.1, 1.0, cv::THRESH_BINARY);

    cv::Rect bbox(0, 0, 0, 0);
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(binaryMask, nonZeroPoints);
    if (!nonZeroPoints.empty()) {
      bbox = cv::boundingRect(nonZeroPoints);
    }

    // std::cout << "bbox: [" << bbox.x << ", " << bbox.y << ", " << bbox.width
    // << ", " << bbox.height << "]" << std::endl;

    std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult =
        _kf.initiate(_kf.xyxy2xyah(Eigen::Vector4f(
            bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)));
    _kfMean = kfResult.first;
    _kfCovariance = kfResult.second;

    _stableFrameCount++;
  } else if (_stableFrameCount < _stableFramesThreshold) {
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult =
        _kf.predict(_kfMean, _kfCovariance);
    _kfMean = kfResult.first;
    _kfCovariance = kfResult.second;

    bestIoUIndex = std::distance(ious, std::max_element(ious, ious + numMasks));
    bestIouScore = ious[bestIoUIndex];
    // auto highResMask = highResMultiMasks + bestIoUIndex * _imageSize *
    // _imageSize; cv::Mat predMask(_imageSize, _imageSize, CV_32FC1,
    // highResMask);
    int lowResMaskSize = m_trtEngines[2]
                             ->getOutputDims()
                             .at(0)
                             .d[2]; //_maskDecoderOutputNodeDims[0][2];
    auto lowResMask =
        lowResMultiMasks + bestIoUIndex * lowResMaskSize * lowResMaskSize;
    cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1,
                     const_cast<float *>(lowResMask));

    cv::Mat binaryMask;
    cv::threshold(predMask, binaryMask, 0.1, 1.0, cv::THRESH_BINARY);

    cv::Rect bbox(0, 0, 0, 0);
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(binaryMask, nonZeroPoints);
    if (!nonZeroPoints.empty()) {
      bbox = cv::boundingRect(nonZeroPoints);
    }

    // std::cout << "bbox: [" << bbox.x << ", " << bbox.y << ", " << bbox.width
    // << ", " << bbox.height << "]" << std::endl;

    if (bestIouScore > _stableIousThreshold) {
      std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.update(
          _kfMean, _kfCovariance,
          _kf.xyxy2xyah(Eigen::Vector4f(bbox.x, bbox.y, bbox.x + bbox.width,
                                        bbox.y + bbox.height)));
      _kfMean = kfResult.first;
      _kfCovariance = kfResult.second;
      _stableFrameCount++;
    } else {
      _stableFrameCount = 0;
    }
  } else {
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult =
        _kf.predict(_kfMean, _kfCovariance);
    _kfMean = kfResult.first;
    _kfCovariance = kfResult.second;

    std::vector<Eigen::Vector4f> predBboxs;
    for (int i = 0; i < numMasks; i++) {
      // auto highResMask = highResMultiMasks + i * _imageSize * _imageSize;
      // cv::Mat predMask(_imageSize, _imageSize, CV_32FC1, highResMask);
      int lowResMaskSize = m_trtEngines[2]
                               ->getOutputDims()
                               .at(0)
                               .d[2]; //_maskDecoderOutputNodeDims[0][2];
      auto lowResMask = lowResMultiMasks + i * lowResMaskSize * lowResMaskSize;
      cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1,
                       const_cast<float *>(lowResMask));

      cv::Mat binaryMask;
      cv::threshold(predMask, binaryMask, 0.1, 1.0, cv::THRESH_BINARY);

      cv::Rect bbox(0, 0, 0, 0);
      std::vector<cv::Point> nonZeroPoints;
      cv::findNonZero(binaryMask, nonZeroPoints);
      if (!nonZeroPoints.empty()) {
        bbox = cv::boundingRect(nonZeroPoints);
      }

      predBboxs.push_back(Eigen::Vector4f(bbox.x, bbox.y, bbox.x + bbox.width,
                                          bbox.y + bbox.height));
    }

    std::vector<float> kfIousVec = _kf.computeIoUs(_kfMean.head(4), predBboxs);

    std::vector<float> weightedIous;
    for (int i = 0; i < numMasks; i++) {
      weightedIous.push_back(_kfScoreWeight * kfIousVec[i] +
                             (1 - _kfScoreWeight) * ious[i]);
    }

    bestIoUIndex = std::distance(
        weightedIous.begin(),
        std::max_element(weightedIous.begin(), weightedIous.end()));
    bestIouScore = ious[bestIoUIndex];
    kfScore = kfIousVec[bestIoUIndex];

    if (bestIouScore > _stableIousThreshold) {
      std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.update(
          _kfMean, _kfCovariance, _kf.xyxy2xyah(predBboxs[bestIoUIndex]));
      _kfMean = kfResult.first;
      _kfCovariance = kfResult.second;
    } else {
      _stableFrameCount = 0;
    }
  }

#endif

  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  SPDLOG_DEBUG("postprocess spent: {:.3f} ms", duration.count() * 1000);

  return {bestIoUIndex, bestIouScore, kfScore};
}
