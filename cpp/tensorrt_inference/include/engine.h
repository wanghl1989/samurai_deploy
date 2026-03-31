#pragma once

#include "NvInfer.h"
#include "driver_types.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudaimgproc.hpp>
// #include <opencv2/cudawarping.hpp>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
// #include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      SPDLOG_ERROR(                                                            \
          "CUDA operation failed with code: {} ({}), with message: {}",        \
          static_cast<int>(err__), cudaGetErrorName(err__),                    \
          cudaGetErrorString(err__));                                          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Utility methods
namespace Util {
inline bool doesFileExist(const std::string &filepath) {
  std::ifstream f(filepath.c_str());
  return f.good();
}

inline void checkCudaErrorCode(cudaError_t code) {
  if (code != 0) {
    std::string errMsg =
        "CUDA operation failed with code: " + std::to_string(code) + "(" +
        cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }
}

std::vector<std::string> getFilesInDirectory(const std::string &dirPath);
} // namespace Util
// Utility Timer
template <typename Clock = std::chrono::high_resolution_clock> class Stopwatch {
  typename Clock::time_point start_point;

public:
  Stopwatch() : start_point(Clock::now()) {}

  // Returns elapsed time
  template <typename Rep = typename Clock::duration::rep,
            typename Units = typename Clock::duration>
  Rep elapsedTime() const {
    std::atomic_thread_fence(std::memory_order_relaxed);
    auto counted_time =
        std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
    std::atomic_thread_fence(std::memory_order_relaxed);
    return static_cast<Rep>(counted_time);
  }
};

using preciseStopwatch = Stopwatch<>;

// Precision used for GPU inference
enum class Precision {
  // Full precision floating point value
  FP32,
  // Half prevision floating point value
  FP16,
  // Int8 quantization.
  // Has reduced dynamic range, may result in slight loss in accuracy.
  // If INT8 is selected, must provide path to calibration dataset directory.
  INT8,
};

// Options for the network
struct Options {
  // Precision to use for GPU inference.
  Precision precision = Precision::FP16;
  // If INT8 precision is selected, must provide path to calibration dataset
  // directory.
  std::string calibrationDataDirectoryPath;
  // The batch size to be used when computing calibration data for INT8
  // inference. Should be set to as large a batch number as your GPU will
  // support.
  int32_t calibrationBatchSize = 128;
  // The batch size which should be optimized for.
  int32_t optBatchSize = 1;
  // Maximum allowable batch size
  int32_t maxBatchSize = 16;
  // GPU device index
  int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override;
};

class Engine {
public:
  Engine(const Options &options);
  ~Engine();

  // Build the onnx model into a TensorRT engine file, cache the model to
  // disk(to avoid rebuilding in future)
  bool buildLoadNetwork(const std::string &onnxModelPath);

  // Load a TensorRT engine file from disk into memory
  bool loadNetwork(const std::string &trtModelPath);

  // Run inference.
  // Input format [input][batch][cv::Mat] (CPU version)
  // Output format [batch][output][feature_vector]
  bool runInference(const std::vector<std::vector<cv::Mat>> &inputs,
                    std::vector<std::vector<float>> &featureVectors);

  // Run inference.
  // Input format [input][float]
  // Output format [output][feature_vector]
  bool runInference(const std::vector<std::vector<float>> &inputs,
                    std::vector<std::vector<float>> &featureVectors);

  // Utility method for resizing an image while maintaining the aspect ratio by
  // adding padding to smaller dimension after scaling While letterbox padding
  // normally adds padding to top & bottom, or left & right sides, this
  // implementation only adds padding to the right or bottom side This is done
  // so that it's easier to convert detected coordinates (ex. YOLO model) back
  // to the original reference frame.
  static cv::Mat resizeKeepAspectRatioPadRightBottom(
      const cv::Mat &input, size_t height, size_t width,
      const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));

  [[nodiscard]] const std::vector<nvinfer1::Dims> &getInputDims() const {
    return m_inputDims;
  };
  [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const {
    return m_outputDims;
  };
  [[nodiscard]] const std::vector<size_t> &getOutputElementCount() const {
    return m_outputLengths;
  };

  void setInputDims(int index, nvinfer1::Dims shape);

  // Utility method for transforming triple nested output array into 2D array
  // Should be used when the output batch size is 1, but there are multiple
  // output feature vectors
  static void
  transformOutput(std::vector<std::vector<std::vector<float>>> &input,
                  std::vector<std::vector<float>> &output);

  // Utility method for transforming triple nested output array into single
  // array Should be used when the output batch size is 1, and there is only a
  // single output feature vector
  static void
  transformOutput(std::vector<std::vector<std::vector<float>>> &input,
                  std::vector<float> &output);
  // Convert NHWC to NCHW and apply scaling and mean subtraction (CPU version)
  static cv::Mat blobFromMats(const std::vector<cv::Mat> &batchInput,
                              const std::array<float, 3> &subVals,
                              const std::array<float, 3> &divVals,
                              bool normalize);

private:
  // Build the network
  bool build(const std::string &onnxModelPath);

  // Converts the engine options into a string
  std::string serializeEngineOptions(const Options &options,
                                     const std::string &onnxModelPath);

  void getDeviceNames(std::vector<std::string> &deviceNames);

  void clearGpuBuffers();

  // Get size of data type
  size_t getTypeSize(nvinfer1::DataType type);

  // caclulate total number of elements given Dims
  size_t getTotalElements(const nvinfer1::Dims &dims);

  // Normalization, scaling, and mean subtraction of inputs
  // std::array<float, 3> m_subVals{};
  // std::array<float, 3> m_divVals{};
  // bool m_normalize;

  // Holds pointers to the input and output GPU buffers
  std::vector<void *> m_buffers;
  std::vector<size_t> m_outputLengths{};
  std::vector<nvinfer1::Dims> m_inputDims;
  std::vector<nvinfer1::Dims> m_outputDims;
  std::vector<std::string> m_IOTensorNames;
  // int32_t m_inputBatchSize;

  // Must keep IRuntime around for inference, see:
  // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
  std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
  const Options m_options;
  Logger m_logger;
};
