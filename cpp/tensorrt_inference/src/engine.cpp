#include "engine.h"
#include "cuda_fp16.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "spdlog.h"
#include <NvInfer.h>
#include <NvInferLegacyDims.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using namespace Util;

std::vector<std::string> Util::getFilesInDirectory(const std::string &dirPath) {
  std::vector<std::string> filepaths;
  for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
    filepaths.emplace_back(entry.path().string());
  }
  return filepaths;
}

void Logger::log(Severity severity, const char *msg) noexcept {
  switch (severity) {
  case Severity::kVERBOSE:
    SPDLOG_DEBUG(msg);
    break;
  case Severity::kINFO:
    SPDLOG_INFO(msg);
    break;
  case Severity::kWARNING:
    SPDLOG_WARN(msg);
    break;
  case Severity::kERROR:
    SPDLOG_ERROR(msg);
    break;
  case Severity::kINTERNAL_ERROR:
    SPDLOG_CRITICAL(msg);
    break;
  default:
    SPDLOG_INFO("Unexpected severity level");
  }
}

Engine::Engine(const Options &options) : m_options(options) {}

Engine::~Engine() { clearGpuBuffers(); }

void Engine::clearGpuBuffers() {
  if (!m_buffers.empty()) {
    // Free GPU memory of outputs
    const auto numInputs = m_inputDims.size();
    for (int32_t outputBinding = numInputs;
         outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
      CUDA_CHECK(cudaFree(m_buffers[outputBinding]));
    }
    m_buffers.clear();
  }
}

size_t Engine::getTypeSize(nvinfer1::DataType type) {
  switch (type) {
  case nvinfer1::DataType::kFLOAT:
    return sizeof(float);
  case nvinfer1::DataType::kHALF:
    return sizeof(__half);
  case nvinfer1::DataType::kINT8:
    return sizeof(int8_t);
  case nvinfer1::DataType::kINT32:
    return sizeof(int32_t);
  case nvinfer1::DataType::kBOOL:
    return sizeof(bool);
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

size_t Engine::getTotalElements(const nvinfer1::Dims &dims) {
  // size_t total = 1;
  // for (int i = 0; i < dims.nbDims; ++i) {
  //     total *= dims.d[i] > 0 ? dims.d[i] : -dims.d[i];
  // }
  // return total;

  return std::accumulate(dims.d, dims.d + dims.nbDims, static_cast<size_t>(1),
                         std::multiplies<int>());
}

void Engine::setInputDims(int index, nvinfer1::Dims shape) {
  if (shape.nbDims != m_inputDims.at(index).nbDims) {
    std::string errMsg =
        "Error, provided shape has incorrect number of dimensions! Expected: " +
        std::to_string(m_inputDims.at(index).nbDims) +
        ", got: " + std::to_string(shape.nbDims);
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  for (int i = 0; i < shape.nbDims; ++i) {
    m_inputDims.at(index).d[i] = shape.d[i];
  }
}

bool Engine::buildLoadNetwork(const std::string &onnxModelPath) {
  // Only regenerate the engine file if it has not already been generated for
  // the specified options, otherwise load cached version from disk
  const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
  SPDLOG_INFO("Searching for engine file with name: {}", engineName);

  if (Util::doesFileExist(engineName)) {
    SPDLOG_INFO("Engine found, not regenerating...");
  } else {
    if (!Util::doesFileExist(onnxModelPath)) {
      std::string errMsg =
          "Could not find onnx model at path: " + onnxModelPath;
      SPDLOG_ERROR(errMsg);
      throw std::runtime_error(errMsg);
    }

    // Was not able to find the engine file, generate...
    SPDLOG_INFO("Engine not found, generating. This could take a while...");

    // Build the onnx model into a TensorRT engine
    auto ret = build(onnxModelPath);
    if (!ret) {
      return false;
    }
  }

  // Load the TensorRT engine file into memory
  return loadNetwork(engineName);
}

bool Engine::loadNetwork(const std::string &trtModelPath) {
  // Read the serialized model from disk
  if (!Util::doesFileExist(trtModelPath)) {
    SPDLOG_ERROR("Error, unable to read TensorRT model at path: {}",
                 trtModelPath);
    return false;
  } else {
    SPDLOG_INFO("Loading TensorRT engine file at path: {}", trtModelPath);
  }

  std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    std::string errMsg = "Unable to read engine file: " + trtModelPath;
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  // Create a runtime to deserialize the engine file.
  m_runtime = std::unique_ptr<nvinfer1::IRuntime>{
      nvinfer1::createInferRuntime(m_logger)};
  if (!m_runtime) {
    SPDLOG_ERROR("Error, unable to create TensorRT runtime.");
    return false;
  }

  // Set the device index
  auto ret = cudaSetDevice(m_options.deviceIndex);
  if (ret != 0) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    auto errMsg = "Unable to set GPU device index to: " +
                  std::to_string(m_options.deviceIndex) +
                  ". Note, your device has " + std::to_string(numGPUs) +
                  " CUDA-capable GPU(s).";
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  // Create an engine, a representation of the optimized model.
  m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!m_engine) {
    SPDLOG_ERROR("Error, unable to create TensorRT engine.");
    return false;
  }

  // The execution context contains all of the state associated with a
  // particular invocation
  m_context = std::unique_ptr<nvinfer1::IExecutionContext>(
      m_engine->createExecutionContext());
  if (!m_context) {
    return false;
  }

  // Storage for holding the input and output buffers
  // This will be passed to TensorRT for inference
  clearGpuBuffers();
  m_buffers.resize(m_engine->getNbIOTensors());
  SPDLOG_INFO("Number of IO Tensors: {}", m_engine->getNbIOTensors());

  m_outputLengths.clear();
  m_inputDims.clear();
  m_outputDims.clear();
  m_IOTensorNames.clear();

  // Create a cuda stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Allocate GPU memory for input and output buffers
  m_outputLengths.clear();
  for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
    const auto tensorName = m_engine->getIOTensorName(i);
    m_IOTensorNames.emplace_back(tensorName);
    const auto tensorType = m_engine->getTensorIOMode(tensorName);
    const auto tensorShape = m_engine->getTensorShape(tensorName);
    const auto tensorDataType = m_engine->getTensorDataType(tensorName);

    std::cout << "Binding " << i << " name: " << std::setw(20) << tensorName
              << ", type: "
              << (tensorType == nvinfer1::TensorIOMode::kINPUT ? "Input"
                                                               : "Output")
              << ", dataType: " << static_cast<int>(tensorDataType)
              << ", shape: [";
    for (int j = 0; j < tensorShape.nbDims; ++j) {
      std::cout << tensorShape.d[j];
      if (j < tensorShape.nbDims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;

    if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
      // Don't need to allocate memory for inputs as we will be using the OpenCV
      // GpuMat buffer directly.

      // Store the input dims for later use
      m_inputDims.emplace_back(tensorShape);
      // m_inputBatchSize = 1; //tensorShape.d[0];
    } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
      // The binding is an output
      m_outputDims.push_back(tensorShape);

      // Now size the output buffer appropriately, taking into account the max
      // possible batch size (although we could actually end up using less
      // memory)
      size_t outputLength = getTotalElements(tensorShape);
      m_outputLengths.push_back(outputLength);

      size_t typeSize = getTypeSize(tensorDataType);
      CUDA_CHECK(cudaMallocAsync(
          &m_buffers[i], outputLength * m_options.maxBatchSize * typeSize,
          stream));
    } else {
      std::string errMsg = "Error, IO Tensor is neither an input or output!";
      SPDLOG_ERROR(errMsg);
      throw std::runtime_error(errMsg);
    }
  }

  // Synchronize and destroy the cuda stream
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return true;
}

bool Engine::build(const std::string &onnxModelPath) {
  // 1. create builder and network to be ready for load the model
  // Create our engine builder.
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(m_logger));
  if (!builder) {
    SPDLOG_ERROR("Failed to create TensorRT builder.");
    return false;
  }

  // Create the network definition object
  auto networkFlags =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(networkFlags));
  if (!network) {
    SPDLOG_ERROR("Failed to create TensorRT network.");
    return false;
  }

  // 2. Create a parser for reading the onnx file.
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, m_logger));
  if (!parser) {
    SPDLOG_ERROR("Failed to create ONNX parser.");
    return false;
  }
  // We are going to first read the onnx file into memory, then pass that buffer
  // to the parser. Had our onnx model file been encrypted, this approach would
  // allow us to first decrypt the buffer.
  std::ifstream file(onnxModelPath,
                     std::ios::binary |
                         std::ios::ate); // ate = open at end of file
  std::streamsize size =
      file.tellg();             // Get size by reading position at end of file
  file.seekg(0, std::ios::beg); // Seek back to beginning of file

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    std::string errMsg = "Unable to read onnx file: " + onnxModelPath;
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  // Parse the buffer we read into memory.
  auto parsed = parser->parse(buffer.data(), buffer.size());
  if (!parsed) {
    SPDLOG_ERROR("Failed to parse ONNX model.");
    return false;
  }
  
  
  // 3. Build the engine of the model
  // create a build configuration specifying how TensorRT should optimize the model
  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    SPDLOG_ERROR("Failed to create TensorRT builder config.");
    return false;
  }

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             1ULL << 30); // 1GB

  // samplesCommon::enableDLA(builder.get(), config.get(), 0);

  // Set the precision level
  const auto engineName = serializeEngineOptions(m_options, onnxModelPath);

  if (m_options.precision == Precision::FP16) {
    // Ensure the GPU supports FP16 inference
    if (!builder->platformHasFastFp16()) {
      std::string errMsg = "Error: GPU does not support FP16 precision";
      SPDLOG_ERROR(errMsg);
      throw std::runtime_error(errMsg);
    }
    config->setFlag(nvinfer1::BuilderFlag::kBF16);
    // config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
  }

  const auto numInputs = network->getNbInputs();
  SPDLOG_INFO("Number of model inputs: {}", numInputs);
  for (int32_t i = 0; i < numInputs; ++i) {
    // Must specify dimensions for all the inputs the model expects.
    const auto input = network->getInput(i);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    SPDLOG_INFO("Input {} name: {}", i, inputName);
    SPDLOG_INFO("Input {} dimensions: ", i, inputDims.nbDims);
    std::cout << "Input " << i << " shape: [";
    for (int32_t d = 0; d < inputDims.nbDims; ++d)
      std::cout << inputDims.d[d] << ", ";
    std::cout << "]" << std::endl;
  }

  if (onnxModelPath.find("memory_attention") !=
      std::string::npos) { // memory_attention has dynamic shape for maskmem_feats, memory_pos_embed and obj_ptrs
    // Register a single optimization profile
    nvinfer1::IOptimizationProfile *optProfile =
        builder->createOptimizationProfile();

    // Specify the optimization profile`
    nvinfer1::Dims maskmemDims =
        network->getInput(2)
            ->getDimensions(); // maskmem_feats shepe [1024, -1, 1, 64], which axis 1 is dynamic.
    optProfile->setDimensions(
        "maskmem_feats", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4(maskmemDims.d[0], 1, maskmemDims.d[2],
                        maskmemDims.d[3]));
    optProfile->setDimensions(
        "maskmem_feats", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4(maskmemDims.d[0], 7, maskmemDims.d[2],
                        maskmemDims.d[3]));
    optProfile->setDimensions(
        "maskmem_feats", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4(maskmemDims.d[0], 7, maskmemDims.d[2],
                        maskmemDims.d[3]));

    nvinfer1::Dims memPosEmbedDims =
        network->getInput(3)
            ->getDimensions(); // memory_pos_embed shape [1024, -1, 1, 64]
    optProfile->setDimensions(
        "memory_pos_embed", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4(memPosEmbedDims.d[0], 1, memPosEmbedDims.d[2],
                        memPosEmbedDims.d[3]));
    optProfile->setDimensions(
        "memory_pos_embed", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4(memPosEmbedDims.d[0], 7, memPosEmbedDims.d[2],
                        memPosEmbedDims.d[3]));
    optProfile->setDimensions(
        "memory_pos_embed", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4(memPosEmbedDims.d[0], 7, memPosEmbedDims.d[2],
                        memPosEmbedDims.d[3]));

    nvinfer1::Dims objPtrDims =
        network->getInput(4)->getDimensions(); // obj_ptrs shape [-1, 1, 256]
    optProfile->setDimensions(
        "obj_ptrs", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims3(1, objPtrDims.d[1], objPtrDims.d[2]));
    optProfile->setDimensions(
        "obj_ptrs", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims3(16, objPtrDims.d[1], objPtrDims.d[2]));
    optProfile->setDimensions(
        "obj_ptrs", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims3(16, objPtrDims.d[1], objPtrDims.d[2]));

    nvinfer1::Dims objPosDims =
        network->getInput(5)->getDimensions(); // obj_pos shape [-1]
    optProfile->setDimensions("obj_pos", nvinfer1::OptProfileSelector::kMIN,
                              nvinfer1::Dims{1, {1}});
    optProfile->setDimensions("obj_pos", nvinfer1::OptProfileSelector::kOPT,
                              nvinfer1::Dims{1, {16}});
    optProfile->setDimensions("obj_pos", nvinfer1::OptProfileSelector::kMAX,
                              nvinfer1::Dims{1, {16}});

    config->addOptimizationProfile(optProfile);
  }

  // CUDA stream used for profiling by the builder.
  cudaStream_t profileStream;
  CUDA_CHECK(cudaStreamCreate(&profileStream));
  config->setProfileStream(profileStream);
  // auto profileStream = samplesCommon::makeCudaStream();
  // if (!profileStream)
  // {
  //     return false;
  // }
  // config->setProfileStream(*profileStream);

  // Build the engine
  // If this call fails, it is suggested to increase the logger verbosity to
  // kVERBOSE and try rebuilding the engine. Doing so will provide you with more
  // information on why exactly it is failing.
  std::unique_ptr<nvinfer1::IHostMemory> plan{
      builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return false;
  }

  // Write the engine to disk
  std::ofstream outfile(engineName, std::ofstream::binary);
  outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

  SPDLOG_INFO("Success, saved engine to {}", engineName);

  CUDA_CHECK(cudaStreamDestroy(profileStream));
  return true;
}

bool Engine::runInference(const std::vector<std::vector<cv::Mat>> &inputs,
                          std::vector<std::vector<float>> &featureVectors) {
  // First we do some error checking
  if (inputs.empty() || inputs[0].empty()) {
    SPDLOG_ERROR("Provided input vector is empty!");
    return false;
  }

  const auto numInputs = m_inputDims.size();

  const auto batchSize = static_cast<int32_t>(inputs[0].size());

  // Create the cuda stream that will be used for inference
  cudaStream_t inferenceCudaStream;
  CUDA_CHECK(cudaStreamCreate(&inferenceCudaStream));

  // Preprocess all the inputs (CPU version - upload to GPU)
  std::vector<void *> inputDevicePtrs(numInputs, nullptr);
  for (size_t i = 0; i < numInputs; ++i) {
    const auto &batchInput = inputs[i];
    const auto &dims = m_inputDims[i];

    auto input = batchInput[0];

    // Upload CPU Mat to GPU
    size_t inputSize = input.total() * input.elemSize();
    CUDA_CHECK(cudaMalloc(&inputDevicePtrs[i], inputSize));
    CUDA_CHECK(cudaMemcpyAsync(inputDevicePtrs[i], input.data, inputSize,
                               cudaMemcpyHostToDevice, inferenceCudaStream));

    m_context->setInputShape(m_IOTensorNames[i].c_str(), dims);
    m_buffers[i] = inputDevicePtrs[i];
  }

  // Ensure all dynamic bindings have been defined.
  if (!m_context->allInputDimensionsSpecified()) {
    std::string errMsg = "Error, not all required dimensions specified.";
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  // Set the address of the input and output buffers
  for (size_t i = 0; i < m_buffers.size(); ++i) {
    bool status =
        m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
    if (!status) {
      return false;
    }
  }

  // for (int32_t i = 0, e = m_engine->getNbIOTensors(); i < e; i++)
  // {
  //     auto const name = m_engine->getIOTensorName(i);
  //     m_context->setTensorAddress(name, buffers.getDeviceBuffer(name));
  // }

  // Run inference.
  bool status = m_context->enqueueV3(inferenceCudaStream);
  if (!status) {
    return false;
  }

  // Copy the outputs back to CPU
  for (int batch = 0; batch < batchSize; ++batch) {
    for (int32_t outputBinding = numInputs;
         outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
      // We start at index m_inputDims.size() to account for the inputs in our
      // m_buffers
      std::vector<float> output;
      auto outputLength = m_outputLengths[outputBinding - numInputs];
      output.resize(outputLength);
      // Copy the output
      CUDA_CHECK(cudaMemcpyAsync(output.data(),
                                 static_cast<char *>(m_buffers[outputBinding]) +
                                     (batch * sizeof(float) * outputLength),
                                 outputLength * sizeof(float),
                                 cudaMemcpyDeviceToHost, inferenceCudaStream));
      featureVectors.emplace_back(std::move(output));
    }
  }

  // Synchronize the cuda stream
  CUDA_CHECK(cudaStreamSynchronize(inferenceCudaStream));

  // Clean up input GPU memory
  for (auto &ptr : inputDevicePtrs) {
    if (ptr) {
      CUDA_CHECK(cudaFree(ptr));
    }
  }

  CUDA_CHECK(cudaStreamDestroy(inferenceCudaStream));
  return true;
}

bool Engine::runInference(const std::vector<std::vector<float>> &inputs,
                          std::vector<std::vector<float>> &featureVectors) {

  // First we do some error checking
  if (inputs.empty()) {
    SPDLOG_ERROR("Provided input vector is empty!");
    return false;
  }

  const auto numInputs = m_inputDims.size();

  // 检查输入数量是否匹配
  if (inputs.size() != numInputs) {
    SPDLOG_ERROR("Number of inputs mismatch! Expected: {}, Got: ", numInputs,
                 inputs.size());
    return false;
  }

  // Create the cuda stream that will be used for inference
  cudaStream_t inferenceCudaStream;
  CUDA_CHECK(cudaStreamCreate(&inferenceCudaStream));

  // 1. 准备GPU内存用于输入数据
  std::vector<void *> inputDevicePtrs(numInputs, nullptr);

  try {
    // 为每个输入分配GPU内存并复制数据
    for (size_t i = 0; i < numInputs; ++i) {
      const auto &input = inputs[i];
      const auto &dims = m_inputDims[i];

      const auto tensorName = m_engine->getIOTensorName(i);
      // const auto tensorShape = m_engine->getTensorShape(tensorName);
      const auto tensorDataType = m_engine->getTensorDataType(tensorName);

      SPDLOG_DEBUG(" tensorName : {}", tensorName);

      // 计算每个输入的体积（元素总数）
      size_t volume = getTotalElements(dims);

      // 检查输入数据大小是否匹配
      if (input.size() != volume) {
        std::stringstream ss;
        ss << "Input " << i << " size mismatch! ";
        ss << "Expected: " << volume << " elements (";
        for (int j = 0; j < dims.nbDims; ++j) {
          ss << dims.d[j];
          if (j < dims.nbDims - 1)
            ss << " × ";
        }
        ss << "), ";
        ss << "Got: " << input.size() << " elements";
        throw std::runtime_error(ss.str());
      }

      // 分配GPU内存（单batch，所以不需要乘以batch size）
      CUDA_CHECK(cudaMalloc(&inputDevicePtrs[i],
                            volume * getTypeSize(tensorDataType)));

      // 调试输出
      // std::cout << "Input " << i << ": allocating " << volume
      //           << " floats (" << volume * sizeof(float)
      //           << " bytes) at GPU ptr " << inputDevicePtrs[i] << std::endl;

      if (tensorDataType == nvinfer1::DataType::kINT32) {
        // std::cout << "point_labels " << std::endl;
        // std::cout << "tensorDataType : " <<
        // static_cast<int32_t>(tensorDataType) << std::endl;

        std::vector<int32_t> input_int(inputs[i].begin(), inputs[i].end());

        CUDA_CHECK(cudaMemcpyAsync(
            inputDevicePtrs[i],                   // 目标：GPU内存地址
            input_int.data(),                     // 源：CPU内存中的连续数据
            volume * getTypeSize(tensorDataType), // 数据大小：所有元素
            cudaMemcpyHostToDevice,               // 方向：CPU -> GPU
            inferenceCudaStream                   // 流
            ));
      } else if (tensorName == std::string("is_mask_from_pts") &&
                 tensorDataType == nvinfer1::DataType::kBOOL) {
        // std::cout << "is_mask_from_pts " << std::endl;
        // std::cout << "tensorDataType : " <<
        // static_cast<int32_t>(tensorDataType) << std::endl;

        bool input_bool = static_cast<bool>(inputs[i][0]);

        CUDA_CHECK(cudaMemcpyAsync(
            inputDevicePtrs[i],                   // 目标：GPU内存地址
            &input_bool,                          // 源：CPU内存中的连续数据
            volume * getTypeSize(tensorDataType), // 数据大小：所有元素
            cudaMemcpyHostToDevice,               // 方向：CPU -> GPU
            inferenceCudaStream                   // 流
            ));
      } else {
        // std::cout << "tensorDataType : " <<
        // static_cast<int32_t>(tensorDataType) << std::endl;
        // 复制数据到GPU（单batch，直接复制全部数据）
        CUDA_CHECK(cudaMemcpyAsync(
            inputDevicePtrs[i],                   // 目标：GPU内存地址
            input.data(),                         // 源：CPU内存中的连续数据
            volume * getTypeSize(tensorDataType), // 数据大小：所有元素
            cudaMemcpyHostToDevice,               // 方向：CPU -> GPU
            inferenceCudaStream                   // 流
            ));
      }

      // 设置TensorRT输入形状（单batch）
      // 注意：dims应该已经包含了batch维度（通常是1）
      m_context->setInputShape(m_IOTensorNames[i].c_str(), dims);

      // 设置输入缓冲区指针
      m_buffers[i] = inputDevicePtrs[i];
    }

    // 确保所有动态绑定都已定义
    if (!m_context->allInputDimensionsSpecified()) {
      std::string errMsg = "Error, not all required dimensions specified.";
      SPDLOG_ERROR(errMsg);
      throw std::runtime_error(errMsg);
    }

    // 设置输入和输出缓冲区的地址
    for (size_t i = 0; i < m_buffers.size(); ++i) {
      if (!m_buffers[i]) {
        std::string errMsg = "Buffer " + std::to_string(i) + " is null!";
        SPDLOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
      }

      bool status =
          m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
      if (!status) {
        std::string errMsg =
            "Failed to set tensor address for: " + m_IOTensorNames[i];
        SPDLOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
      }
    }

    // 运行推理（单batch）
    // std::cout << "Executing inference..." << std::endl;
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
      std::string errMsg = "Failed to execute inference";
      SPDLOG_ERROR(errMsg);
      throw std::runtime_error(errMsg);
    }

    // 将输出复制回CPU（单batch）
    for (int32_t outputBinding = numInputs;
         outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
      auto outputLength = m_outputLengths[outputBinding - numInputs];

      // 检查输出缓冲区是否有效
      if (!m_buffers[outputBinding]) {
        std::string errMsg =
            "Output buffer " + std::to_string(outputBinding) + " is null";
        SPDLOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
      }

      std::vector<float> output(outputLength);

      // 调试输出
      // std::cout << "Output " << outputBinding - numInputs << ": copying "
      //           << outputLength << " floats from GPU ptr "
      //           << m_buffers[outputBinding] << std::endl;

      // 从GPU复制输出数据（单batch，无偏移）
      CUDA_CHECK(cudaMemcpyAsync(output.data(),                // 目标：CPU内存
                                 m_buffers[outputBinding],     // 源：GPU内存
                                 outputLength * sizeof(float), // 数据大小
                                 cudaMemcpyDeviceToHost, // 方向：GPU -> CPU
                                 inferenceCudaStream     // 流
                                 ));

      featureVectors.emplace_back(std::move(output));
    }

    // 同步CUDA流（等待所有异步操作完成）
    // std::cout << "Synchronizing CUDA stream..." << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(inferenceCudaStream));
    // std::cout << "Inference completed successfully!" << std::endl;

  } catch (const std::exception &e) {
    SPDLOG_ERROR("===== Inference Error ===== {}", e.what());

    // 清理GPU内存
    for (auto &ptr : inputDevicePtrs) {
      if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
      }
    }

    CUDA_CHECK(cudaStreamDestroy(inferenceCudaStream));
    return false;
  }

  // 清理GPU内存
  for (auto &ptr : inputDevicePtrs) {
    if (ptr) {
      CUDA_CHECK(cudaFree(ptr));
    }
  }

  CUDA_CHECK(cudaStreamDestroy(inferenceCudaStream));
  return true;
}

cv::Mat Engine::blobFromMats(const std::vector<cv::Mat> &batchInput,
                             const std::array<float, 3> &subVals,
                             const std::array<float, 3> &divVals,
                             bool normalize) {
  cv::Mat dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(),
              CV_8UC3);

  size_t width = batchInput[0].cols * batchInput[0].rows;
  for (size_t img = 0; img < batchInput.size(); img++) {
    std::vector<cv::Mat> input_channels{
        cv::Mat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                &(dst.ptr()[0 + width * 3 * img])),
        cv::Mat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                &(dst.ptr()[width + width * 3 * img])),
        cv::Mat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                &(dst.ptr()[width * 2 + width * 3 * img]))};
    cv::split(batchInput[img], input_channels); // HWC -> CHW
  }

  cv::Mat mfloat;
  if (normalize) {
    // [0.f, 1.f]
    dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
  } else {
    // [0.f, 255.f]
    dst.convertTo(mfloat, CV_32FC3);
  }

  // Apply scaling and mean subtraction
  cv::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat);
  cv::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat);

  return mfloat;
}

std::string Engine::serializeEngineOptions(const Options &options,
                                           const std::string &onnxModelPath) {
  const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
  std::string engineName =
      onnxModelPath.substr(filenamePos,
                           onnxModelPath.find_last_of('.') - filenamePos) +
      ".engine";

  // Add the GPU device name to the file to ensure that the model is only used
  // on devices with the exact same GPU
  std::vector<std::string> deviceNames;
  getDeviceNames(deviceNames);

  if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
    std::string errMsg = "Error, provided device index is out of range!";
    SPDLOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
  }

  auto deviceName = deviceNames[options.deviceIndex];
  // Remove spaces from the device name
  deviceName.erase(
      std::remove_if(deviceName.begin(), deviceName.end(), ::isspace),
      deviceName.end());

  engineName += "." + deviceName;

  // Serialize the specified options into the filename
  if (options.precision == Precision::FP16) {
    engineName += ".fp16";
  } else if (options.precision == Precision::FP32) {
    engineName += ".fp32";
  } else {
    engineName += ".int8";
  }

  engineName += "." + std::to_string(options.maxBatchSize);
  engineName += "." + std::to_string(options.optBatchSize);

  return engineName;
}

void Engine::getDeviceNames(std::vector<std::string> &deviceNames) {
  int numGPUs;
  cudaGetDeviceCount(&numGPUs);

  for (int device = 0; device < numGPUs; device++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    deviceNames.push_back(std::string(prop.name));
  }
}

cv::Mat Engine::resizeKeepAspectRatioPadRightBottom(const cv::Mat &input,
                                                    size_t height, size_t width,
                                                    const cv::Scalar &bgcolor) {
  float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
  int unpad_w = r * input.cols;
  int unpad_h = r * input.rows;
  cv::Mat re(unpad_h, unpad_w, CV_8UC3);
  cv::resize(input, re, re.size());
  cv::Mat out(height, width, CV_8UC3, bgcolor);
  re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
  return out;
}

void Engine::transformOutput(
    std::vector<std::vector<std::vector<float>>> &input,
    std::vector<std::vector<float>> &output) {
  if (input.size() != 1) {
    std::string errMsg = "The feature vector has incorrect dimensions!";
    SPDLOG_ERROR(errMsg);
    throw std::logic_error(errMsg);
  }

  output = std::move(input[0]);
}

void Engine::transformOutput(
    std::vector<std::vector<std::vector<float>>> &input,
    std::vector<float> &output) {
  if (input.size() != 1 || input[0].size() != 1) {
    std::string errMsg = "The feature vector has incorrect dimensions!";
    SPDLOG_ERROR(errMsg);
    throw std::logic_error(errMsg);
  }

  output = std::move(input[0][0]);
}
