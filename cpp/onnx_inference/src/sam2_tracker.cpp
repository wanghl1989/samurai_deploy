#include "sam2_tracker.h"

SAM2Tracker::SAM2Tracker(const std::string& modelPath, bool useGPU, bool enableFp16) {
    loadNetwork(modelPath, useGPU, enableFp16);
}

void SAM2Tracker::loadNetwork(const std::string& modelPath, bool useGPU, bool enableFp16) {
    // 1) init ONNX Runtime SessionOptions
    _sessionOptions = Ort::SessionOptions();
    // std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << std::endl;
    _sessionOptions.SetIntraOpNumThreads(std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
    _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 2) create sessions : image_encoder / memory_attention / memory_encoder / mask_decoder
    // Retrieve available execution providers (CPU, GPU, etc.)
    std::vector<std::string> available_providers = Ort::GetAvailableProviders();
    std::cout << "Available execution providers: ";
    for (const auto& provider : available_providers) {
        std::cout << provider << " ";
    }
    std::cout << std::endl;

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    // Configure session options based on whether GPU is to be used and available
    if (useGPU && cudaAvailable != availableProviders.end())
    {
        std::cout << "Inference device: GPU" << std::endl;
        OrtCUDAProviderOptions cudaOption;
        // cudaOption.device_id = 0;
        // cudaOption.arena_extend_strategy = 0;
        // cudaOption.gpu_mem_limit = 1 * 1024 * 1024 * 1024;
        // // cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE; // 1.8.0 
        // cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        // cudaOption.do_copy_in_default_stream = 1;
        // cudaOption.default_memory_arena_cfg = nullptr;
        _sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
    }
    else
    {
        if (useGPU)
        {
            std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        }
        std::cout << "Inference device: CPU" << std::endl;
    }

    if (access(modelPath.c_str(), F_OK) == -1) {
        throw std::runtime_error("Model path does not exist: " + modelPath);
    }
    std::vector<std::string> modelFiles(4);
    if (enableFp16 && cudaAvailable != availableProviders.end()) {
        if (!useGPU) {
            throw std::runtime_error("FP16 is only supported on GPU, please set useGPU to true.");
        }
        modelFiles[0] = modelPath + "/image_encoder_FP16.onnx";
        modelFiles[1] = modelPath + "/memory_attention_FP16.onnx";
        modelFiles[2] = modelPath + "/mask_decoder_FP16.onnx";
        modelFiles[3] = modelPath + "/memory_encoder_FP16.onnx";
    } else {
        modelFiles[0] = modelPath + "/image_encoder.onnx";
        modelFiles[1] = modelPath + "/memory_attention.onnx";
        modelFiles[2] = modelPath + "/mask_decoder.onnx";
        modelFiles[3] = modelPath + "/memory_encoder.onnx";
        // modelFiles[0] = modelPath + "/image_encoder_simplified.onnx";
        // modelFiles[1] = modelPath + "/memory_attention_simplified.onnx";
        // modelFiles[2] = modelPath + "/mask_decoder_simplified.onnx";
        // modelFiles[3] = modelPath + "/memory_encoder.onnx";
    }

    _imageEncoderSession    = std::make_unique<Ort::Session>(_env, modelFiles[0].c_str(), _sessionOptions);
    _memoryAttentionSession = std::make_unique<Ort::Session>(_env, modelFiles[1].c_str(), _sessionOptions);
    _maskDecoderSession     = std::make_unique<Ort::Session>(_env, modelFiles[2].c_str(), _sessionOptions);
    _memoryEncoderSession   = std::make_unique<Ort::Session>(_env, modelFiles[3].c_str(), _sessionOptions);

    std::cout << "image_encoder model、memory_attention model、mask_decoder model、memory_encoder model loaded successfully." << std::endl;

    // 3) print model info and get input/output node names and dims
    getModelInfo(_imageEncoderSession.get(), "image_encoder", 
                    _imageEncoderInputNodeNames, _imageEncoderOutputNodeNames,
                    _imageEncoderInputNodeDims, _imageEncoderOutputNodeDims);
    getModelInfo(_memoryAttentionSession.get(), "memory_attention", 
                    _memoryAttentionInputNodeNames, _memoryAttentionOutputNodeNames,
                    _memoryAttentionInputNodeDims, _memoryAttentionOutputNodeDims);
    getModelInfo(_maskDecoderSession.get(), "mask_decoder", 
                    _maskDecoderInputNodeNames, _maskDecoderOutputNodeNames,
                    _maskDecoderInputNodeDims, _maskDecoderOutputNodeDims);
    getModelInfo(_memoryEncoderSession.get(), "memory_encoder", 
                    _memoryEncoderInputNodeNames, _memoryEncoderOutputNodeNames,
                    _memoryEncoderInputNodeDims, _memoryEncoderOutputNodeDims);

    if (_imageEncoderInputNodeDims[0][1] != _imageSize) {
        std::cerr << "_imageSize: " << _imageSize << ", _imageEncoderInputNodeDims[0][1]: " << _imageEncoderInputNodeDims[0][1] << std::endl;
        throw std::runtime_error("image_encoder input size should be equal to _imageSize");
    }
}

void SAM2Tracker::getModelInfo(const Ort::Session* session, const std::string& modelName,
                                std::vector<const char*>& inputNodeNames,
                                std::vector<const char*>& outputNodeNames,
                                std::vector<std::vector<int64_t>>& inputNodeDims,
                                std::vector<std::vector<int64_t>>& outputNodeDims) {
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session->GetInputCount();
    size_t numOutputNodes = session->GetOutputCount();
    std::cout << "\033[33mNumber of " << modelName << " input nodes: " << numInputNodes << "\033[0m" << std::endl;
    std::cout << "\033[33mNumber of " << modelName << " output nodes: " << numOutputNodes << "\033[0m" << std::endl;

    for (size_t i = 0; i < numInputNodes; i++) {
        Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << ": " << inputName.get();
        _inputNodeNameAllocatedStrings.push_back(std::move(inputName));
        inputNodeNames.push_back(_inputNodeNameAllocatedStrings.back().get());

        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputDims = tensorInfo.GetShape();
        inputNodeDims.push_back(inputDims);

        std::cout << ", DataType: ";
        printDataType(tensorInfo.GetElementType());

        std::cout << " Shape: [";
        for (size_t j = 0; j < inputDims.size(); j++) {
            std::cout << inputDims[j];
            if (j < inputDims.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    for (size_t i = 0; i < numOutputNodes; i++) {
        Ort::AllocatedStringPtr outputName = session->GetOutputNameAllocated(i, allocator);
        std::cout << "Output " << i << ": " << outputName.get();
        _outputNodeNameAllocatedStrings.push_back(std::move(outputName));
        outputNodeNames.push_back(_outputNodeNameAllocatedStrings.back().get());

        Ort::TypeInfo typeInfo = session->GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputDims = tensorInfo.GetShape();
        outputNodeDims.push_back(outputDims);

        std::cout << ", DataType: ";
        printDataType(tensorInfo.GetElementType());

        std::cout << " Shape: [";
        for (size_t j = 0; j < outputDims.size(); j++) {
            std::cout << outputDims[j];
            if (j < outputDims.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

void SAM2Tracker::printDataType(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            std::cout << "float" << std::endl;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            std::cout << "float16" << std::endl;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            std::cout << "double" << std::endl;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            std::cout << "uint8" << std::endl;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            std::cout << "int8" << std::endl;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            std::cout << "int32" << std::endl;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            std::cout << "int64" << std::endl;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            std::cout << "bool" << std::endl;
            break;
        default:
            std::cout << "ONNXTensorElementDataType : " << type << std::endl;
            break;
    }
}

void SAM2Tracker::imageEncoderInference(std::vector<float>& frame, std::vector<Ort::Value>& imageEncoderOutputTensors) {
    auto start = std::chrono::high_resolution_clock::now();

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputImageTensor = Ort::Value::CreateTensor<float>(memoryInfo, frame.data(), frame.size(),
                                                            _imageEncoderInputNodeDims[0].data(), _imageEncoderInputNodeDims[0].size());
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputImageTensor));
    imageEncoderOutputTensors = _imageEncoderSession->Run(Ort::RunOptions{nullptr},
                                                                    _imageEncoderInputNodeNames.data(),
                                                                    inputTensors.data(),
                                                                    inputTensors.size(),
                                                                    _imageEncoderOutputNodeNames.data(),
                                                                    _imageEncoderOutputNodeNames.size());
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "image_encoder spent: " << duration.count() << " ms" << std::endl;
}

void SAM2Tracker::memoryAttentionInference(int frameIdx,
                                           std::vector<Ort::Value>& imageEncoderOutputTensors,
                                           std::vector<Ort::Value>& memoryAttentionOutputTensors)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> memmaskFeatures = _memoryBank[0].maskmem_features;
    std::vector<float> memmaskPosEncs  = _memoryBank[0].maskmem_pos_enc;
    std::vector<float> objectPtrs      = _memoryBank[0].obj_ptr;
    // std::cout << "memmaskFeatures.size(): " << memmaskFeatures.size() << std::endl;
    // std::cout << "memmaskPosEncs.size(): " << memmaskPosEncs.size() << std::endl;
    // std::cout << "objectPtrs.size(): " << objectPtrs.size() << std::endl;

    std::vector<int> validIndices;
    if (frameIdx > 1) {
        for (int i = frameIdx - 1; i > 0; i--) {
            float iouScore = _memoryBank[i].best_iou_score;
            float objScore = _memoryBank[i].obj_score_logits;
            float kfScore = _memoryBank[i].kf_score;
            if (validIndices.size() >= _maxObjPtrsInEncoder - 1) {
                break;
            }
            if (iouScore > _memoryBankIouThreshold && objScore > _memoryBankObjScoreThreshold && (kfScore > _memoryBankKfScoreThreshold)) {
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

    size_t maskmemFeaturesSize = _memoryEncoderSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementCount(); // 262144=4096*64
    size_t maskmemPosEncSize   = _memoryEncoderSession->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetElementCount(); // 262144=4096*64
    size_t objPtrSize          = _maskDecoderOutputNodeDims[2][2]; // 1*3*256
    // std::cout << "maskmemFeaturesSize: " << maskmemFeaturesSize << std::endl;
    // std::cout << "maskmemPosEncSize: " << maskmemPosEncSize << std::endl;
    // std::cout << "objPtrSize: " << objPtrSize << std::endl;
    size_t memmaskFeaturesNum = std::min(static_cast<size_t>(_numMaskmem), validIndices.size() + 1);
    size_t memmaskPosEncNum = std::min(static_cast<size_t>(_numMaskmem), validIndices.size() + 1);

    memmaskFeatures.reserve(maskmemFeaturesSize * memmaskFeaturesNum); // 最近 num_maskmem-1 帧 + 0帧 的maskmem_features 
    memmaskPosEncs.reserve(maskmemPosEncSize * memmaskPosEncNum); // 最近 num_maskmem-1 帧 + 0帧 的maskmem_pos_enc

    // std::cout << "memmaskFeatures idx: ";
    int validIndicesSize = validIndices.size();
    for (int i = validIndicesSize - _numMaskmem + 1; i < validIndicesSize; i++) { // 最近 num_maskmem-1 帧
        if (i < 0) {
            continue;
        }
        // std::cout << i << ": " << validIndices[i] << ", ";

        int prevFrameIdx = validIndices[i];
        MemoryBankEntry mem = _memoryBank[prevFrameIdx];
        memmaskFeatures.insert(memmaskFeatures.end(), mem.maskmem_features.begin(), mem.maskmem_features.end());
        memmaskPosEncs.insert(memmaskPosEncs.end(), mem.maskmem_pos_enc.begin(), mem.maskmem_pos_enc.end());
    }
    // std::cout << std::endl;
    
    auto start2 = std::chrono::high_resolution_clock::now();
    // std::cout << "memmaskFeaturesNum: " << memmaskFeaturesNum << std::endl;
    std::vector<int64_t> tposEncSize = _maskDecoderOutputNodeDims[4]; // 7*1*1*64
    std::vector<int64_t> maskmemPosEncShape = _memoryEncoderOutputNodeDims[1]; // 4096*1*64
    // #pragma omp parallel for
    for (int i = 1; i < memmaskFeaturesNum; i++) {
        int start = (memmaskFeaturesNum - i) * maskmemPosEncSize;
        int end = start + maskmemPosEncSize;
        // #pragma omp parallel for
        for (int j = start; j < end; j++) {
            memmaskPosEncs[j] += _maskMemTposEnc[(i - 1) * tposEncSize[3] + (j % tposEncSize[3])];
        }
    }
    // #pragma omp parallel for
    for (int i = 0; i < maskmemFeaturesSize; i++) {
        memmaskPosEncs[i] += _maskMemTposEnc[(tposEncSize[0] - 1) * tposEncSize[3] + (i % tposEncSize[3])];
    }
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start2);
    std::cout << "memmaskPosEncs spent: " << duration2.count() << " ms" << std::endl;

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

    objectPtrs.reserve(objPtrSize * objPosEnc.size()); // 最近 maxObjPtrsInEncoder-1 帧 + 0帧 的obj_ptr
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

    std::cout << "memmaskFeaturesNum: " << memmaskFeaturesNum << std::endl;
    std::cout << "memmaskPosEncNum: " << memmaskPosEncNum << std::endl;
    std::cout << "validIndices.size(): " << validIndices.size() << std::endl;
    std::cout << "objectPtrs.size(): " << objectPtrs.size() / objPtrSize << std::endl;
    std::cout << "objPosEnc.size(): " << objPosEnc.size() << std::endl;
    std::cout << "memmaskFeatures.size(): " << memmaskFeatures.size() / maskmemFeaturesSize << ", " << memmaskFeatures.capacity() / maskmemFeaturesSize << std::endl;
    std::cout << "memmaskPosEncs.size(): " << memmaskPosEncs.size() / maskmemPosEncSize<< ", " << memmaskPosEncs.capacity() / maskmemPosEncSize << std::endl;

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputTensors;

    _memoryAttentionInputNodeDims[2][1] = memmaskFeaturesNum;
    _memoryAttentionInputNodeDims[3][1] = memmaskPosEncNum;
    _memoryAttentionInputNodeDims[4][0] = objPosEnc.size();
    _memoryAttentionInputNodeDims[5][0] = objPosEnc.size();
    Ort::Value memoryTensor = Ort::Value::CreateTensor<float>(memoryInfo, memmaskFeatures.data(), memmaskFeatures.size(),
                                                            _memoryAttentionInputNodeDims[2].data(), _memoryAttentionInputNodeDims[2].size());
    Ort::Value memoryPosEncTensor = Ort::Value::CreateTensor<float>(memoryInfo, memmaskPosEncs.data(), memmaskPosEncs.size(),
                                                            _memoryAttentionInputNodeDims[3].data(), _memoryAttentionInputNodeDims[3].size());
    Ort::Value objectPtrsTensor = Ort::Value::CreateTensor<float>(memoryInfo, objectPtrs.data(), objectPtrs.size(),
                                                            _memoryAttentionInputNodeDims[4].data(), _memoryAttentionInputNodeDims[4].size());
    Ort::Value objPosEncTensor = Ort::Value::CreateTensor<int>(memoryInfo, objPosEnc.data(), objPosEnc.size(),
                                                            _memoryAttentionInputNodeDims[5].data(), _memoryAttentionInputNodeDims[5].size());
    inputTensors.clear();
    inputTensors.push_back(std::move(imageEncoderOutputTensors[2])); // lowResFeatures
    inputTensors.push_back(std::move(imageEncoderOutputTensors[3])); // visionPosEmbedding
    inputTensors.push_back(std::move(memoryTensor));
    inputTensors.push_back(std::move(memoryPosEncTensor));
    inputTensors.push_back(std::move(objectPtrsTensor));
    inputTensors.push_back(std::move(objPosEncTensor));

    memoryAttentionOutputTensors = _memoryAttentionSession->Run(Ort::RunOptions{nullptr},
                                                                    _memoryAttentionInputNodeNames.data(),
                                                                    inputTensors.data(),
                                                                    inputTensors.size(),
                                                                    _memoryAttentionOutputNodeNames.data(),
                                                                    _memoryAttentionOutputNodeNames.size());
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "memory_attention spent: " << duration.count() << " ms" << std::endl;
}

void SAM2Tracker::maskDecoderInference(std::vector<float>& inputPoints,
                                       std::vector<int32_t>& inputLabels,
                                       std::vector<Ort::Value>& imageEncoderOutputTensors,
                                       Ort::Value& pixFeatWithMem,
                                       std::vector<Ort::Value>& maskDecoderOutputTensors)
{
    auto start = std::chrono::high_resolution_clock::now();

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    _maskDecoderInputNodeDims[0][1] = 2;
    _maskDecoderInputNodeDims[1][1] = 2;
    Ort::Value inputPointsTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputPoints.data(), inputPoints.size(),
                                                            _maskDecoderInputNodeDims[0].data(), _maskDecoderInputNodeDims[0].size());
    Ort::Value inputLabelsTensor = Ort::Value::CreateTensor<int32_t>(memoryInfo, inputLabels.data(), inputLabels.size(),
                                                            _maskDecoderInputNodeDims[1].data(), _maskDecoderInputNodeDims[1].size());

    // Ort::Value videoWidthTensor  = Ort::Value::CreateTensor<int>(memoryInfo, &_videoWidth, 1,  _maskDecoderInputNodeDims[5].data(), 1);
    // Ort::Value videoHeightTensor = Ort::Value::CreateTensor<int>(memoryInfo, &_videoHeight, 1, _maskDecoderInputNodeDims[6].data(), 1);

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputPointsTensor));
    inputTensors.push_back(std::move(inputLabelsTensor));
    inputTensors.push_back(std::move(pixFeatWithMem)); // pixFeatWithMem
    inputTensors.push_back(std::move(imageEncoderOutputTensors[0])); // highResFeatures0
    inputTensors.push_back(std::move(imageEncoderOutputTensors[1])); // highResFeatures1
    // inputTensors.push_back(std::move(videoWidthTensor));
    // inputTensors.push_back(std::move(videoHeightTensor));

    maskDecoderOutputTensors = _maskDecoderSession->Run(Ort::RunOptions{nullptr},
                                                                    _maskDecoderInputNodeNames.data(),
                                                                    inputTensors.data(),
                                                                    inputTensors.size(),
                                                                    _maskDecoderOutputNodeNames.data(),
                                                                    _maskDecoderOutputNodeNames.size());
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "mask_decoder spent: " << duration.count() << " ms" << std::endl;
}

void SAM2Tracker::memoryEncoderInference(Ort::Value& visionFeaturesTensor,
                                         Ort::Value& highResMasksForMemTensor,
                                         Ort::Value& objectScoreLogitsTensor,
                                         bool isMaskFromPts,
                                         std::vector<Ort::Value>& memoryEncoderOutputTensors)
{
    auto start = std::chrono::high_resolution_clock::now();

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value isMaskFromPtsTensor = Ort::Value::CreateTensor<bool>(memoryInfo, &isMaskFromPts, 1, nullptr, 0);

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(visionFeaturesTensor));     // lowResFeatures
    inputTensors.push_back(std::move(highResMasksForMemTensor)); // highResMasksForMem
    inputTensors.push_back(std::move(objectScoreLogitsTensor));  // objectScoreLogits
    inputTensors.push_back(std::move(isMaskFromPtsTensor));

    memoryEncoderOutputTensors = _memoryEncoderSession->Run(Ort::RunOptions{nullptr},
                                                                    _memoryEncoderInputNodeNames.data(),
                                                                    inputTensors.data(),
                                                                    inputTensors.size(),
                                                                    _memoryEncoderOutputNodeNames.data(),
                                                                    _memoryEncoderOutputNodeNames.size());
                                                            
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "memory_encoder spent: " << duration.count() << " ms" << std::endl;
}

cv::Mat SAM2Tracker::addFirstFrameBbox(int frameIdx, const cv::Mat& firstFrame, const cv::Rect& bbox) {

    _videoWidth = static_cast<int>(firstFrame.cols);
    _videoHeight = static_cast<int>(firstFrame.rows);

    std::vector<float> inputImage;
    preprocessImage(firstFrame, inputImage);

    // 1) image_encoder 推理
    std::vector<Ort::Value> imageEncoderOutputTensors;
    imageEncoderInference(inputImage, imageEncoderOutputTensors);

    // 2) mask_decoder 推理
    std::vector<float> inputPoints = {static_cast<float>(bbox.x), static_cast<float>(bbox.y), 
                                      static_cast<float>(bbox.x + bbox.width), static_cast<float>(bbox.y + bbox.height)};
    inputPoints[0] = (inputPoints[0] / firstFrame.cols) * _imageSize;
    inputPoints[1] = (inputPoints[1] / firstFrame.rows) * _imageSize;
    inputPoints[2] = (inputPoints[2] / firstFrame.cols) * _imageSize;
    inputPoints[3] = (inputPoints[3] / firstFrame.rows) * _imageSize;

    std::vector<int32_t> boxLabels = {2, 3};

    std::vector<Ort::Value> maskDecoderOutputTensors;
    maskDecoderInference(inputPoints, boxLabels,
                         imageEncoderOutputTensors,
                         imageEncoderOutputTensors[4],
                         maskDecoderOutputTensors);

    PostprocessResult result = postprocessOutput(maskDecoderOutputTensors);
    int bestIoUIndex = result.bestIoUIndex;
    float bestIouScore = result.bestIouScore;
    float kfScore = result.kfScore;

    auto lowResMultiMasks  = maskDecoderOutputTensors[0].GetTensorMutableData<float>();
    // auto highResMultiMasks = maskDecoderOutputTensors[1].GetTensorMutableData<float>();
    // auto ious              = maskDecoderOutputTensors[1].GetTensorMutableData<float>();
    auto objPtrs           = maskDecoderOutputTensors[2].GetTensorMutableData<float>();
    auto objScoreLogits    = maskDecoderOutputTensors[3].GetTensorMutableData<float>();
    auto maskMemTposEncTmp = maskDecoderOutputTensors[4].GetTensorMutableData<float>(); // 7*1*1*64
    _maskMemTposEnc = std::vector<float>(maskMemTposEncTmp, maskMemTposEncTmp + maskDecoderOutputTensors[4].GetTensorTypeAndShapeInfo().GetElementCount());
    // // check _maskMemTposEnc
    // for (int i = 0; i < _maskDecoderOutputNodeDims[4][0]; i++) { // 7
    //     std::cout << "maskMemTposEnc[" << i << "]: ";
    //     for (int j = 0; j < _maskDecoderOutputNodeDims[4][3]; j++) { // 64
    //         std::cout << _maskMemTposEnc[i * _maskDecoderOutputNodeDims[4][3] + j] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    int lowResMaskHeight = _maskDecoderOutputNodeDims[0][2];
    int lowResMaskWidth  = _maskDecoderOutputNodeDims[0][3];
    auto lowResMask = lowResMultiMasks + bestIoUIndex * lowResMaskHeight * lowResMaskWidth;
    // auto highResMask = highResMultiMasks + bestIoUIndex * _imageSize * _imageSize;

    cv::Mat predMask(lowResMaskHeight, lowResMaskWidth, CV_32FC1, lowResMask);
    // cv::Mat predMask(_videoHeight, _videoWidth, CV_32FC1, lowResMask);

    // 3) memory_encoder 推理
    bool isMaskFromPts = frameIdx == 0;

    cv::Mat highResMaskMat;
    cv::resize(predMask, highResMaskMat, cv::Size(_imageSize, _imageSize));

    // std::vector<float> highResMask((float*)highResMaskMat.data, (float*)highResMaskMat.data + highResMaskMat.total());
    std::vector<float> highResMask(highResMaskMat.begin<float>(), highResMaskMat.end<float>());
    std::vector<int64_t> highResMaskDims = {1, 1, _imageSize, _imageSize};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value highResMaskForMemTensor = Ort::Value::CreateTensor<float>(memoryInfo, highResMask.data(), _imageSize * _imageSize,
                                                        highResMaskDims.data(), highResMaskDims.size());

    std::vector<Ort::Value> memoryEncoderOutputTensors;
    memoryEncoderInference(imageEncoderOutputTensors[2],
                           highResMaskForMemTensor,
                           maskDecoderOutputTensors[3],
                           isMaskFromPts,
                           memoryEncoderOutputTensors);

    auto maskmemFeatures = memoryEncoderOutputTensors[0].GetTensorMutableData<float>();
    auto maskmemPosEnc   = memoryEncoderOutputTensors[1].GetTensorMutableData<float>();
    
    // 4) save memory bank
    size_t maskmemFeaturesSize = memoryEncoderOutputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); // 262144=4096*1*64
    size_t maskmemPosEncSize   = memoryEncoderOutputTensors[1].GetTensorTypeAndShapeInfo().GetElementCount(); // 262144=4096*1*64
    size_t objPtrSize          = _maskDecoderOutputNodeDims[2][2]; // 1,3,256
    // std::cout << "maskmemFeaturesSize: " << maskmemFeaturesSize << std::endl;
    // std::cout << "maskmemPosEncSize: " << maskmemPosEncSize << std::endl;
    // std::cout << "objPtrSize: " << objPtrSize << std::endl;

    MemoryBankEntry entry;
    entry.maskmem_features = std::vector<float>(maskmemFeatures, maskmemFeatures + maskmemFeaturesSize);
    entry.maskmem_pos_enc  = std::vector<float>(maskmemPosEnc, maskmemPosEnc + maskmemPosEncSize);
    entry.obj_ptr          = std::vector<float>(objPtrs + bestIoUIndex * objPtrSize, objPtrs + (bestIoUIndex + 1) * objPtrSize);
    entry.best_iou_score   = bestIouScore;
    entry.obj_score_logits = objScoreLogits[0];
    entry.kf_score         = kfScore;

    _memoryBank[frameIdx] = entry;

    return predMask;
}

cv::Mat SAM2Tracker::trackStep(int frameIdx, const cv::Mat& frame) {
    std::vector<float> inputImage;
    preprocessImage(frame, inputImage);

    // 1) image_encoder 推理
    std::vector<Ort::Value> imageEncoderOutputTensors;
    imageEncoderInference(inputImage, imageEncoderOutputTensors);

    auto lowResFeatures = imageEncoderOutputTensors[2].GetTensorMutableData<float>(); // 下面要用到两次，保留副本，因为使用std::move后，原来的数据所有权转移
    size_t lowResFeaturesSize = imageEncoderOutputTensors[2].GetTensorTypeAndShapeInfo().GetElementCount();

    // 2) memory_attention 推理
    std::vector<Ort::Value> memoryAttentionOutputTensors;
    memoryAttentionInference(frameIdx, imageEncoderOutputTensors, memoryAttentionOutputTensors);

    // 3) mask_decoder 推理
    std::vector<float> inputPoints = {0, 0, 0, 0};
    std::vector<int32_t> inputLabels = {-1, -1};

    std::vector<Ort::Value> maskDecoderOutputTensors;
    maskDecoderInference(inputPoints, inputLabels,
                        imageEncoderOutputTensors,
                        memoryAttentionOutputTensors[0],
                        maskDecoderOutputTensors);

    PostprocessResult result = postprocessOutput(maskDecoderOutputTensors);
    int bestIoUIndex = result.bestIoUIndex;
    float bestIouScore = result.bestIouScore;
    float kfScore = result.kfScore;

    auto lowResMultiMasks  = maskDecoderOutputTensors[0].GetTensorMutableData<float>();
    // auto highResMultiMasks = maskDecoderOutputTensors[1].GetTensorMutableData<float>();
    // auto ious              = maskDecoderOutputTensors[1].GetTensorMutableData<float>();
    auto objPtrs           = maskDecoderOutputTensors[2].GetTensorMutableData<float>();
    auto objScoreLogits    = maskDecoderOutputTensors[3].GetTensorMutableData<float>();

    int lowResMaskHeight = _maskDecoderOutputNodeDims[0][2];
    int lowResMaskWidth  = _maskDecoderOutputNodeDims[0][3];

    auto lowResMask = lowResMultiMasks + bestIoUIndex * lowResMaskHeight * lowResMaskWidth;
    // auto highResMask = highResMultiMasks + bestIoUIndex * _imageSize * _imageSize;

    cv::Mat predMask(lowResMaskHeight, lowResMaskWidth, CV_32FC1, lowResMask);
    // cv::Mat predMask(_videoHeight, _videoWidth, CV_32FC1, lowResMask);

    // 4) memory_encoder 推理
#if 1
    bool isMaskFromPts = frameIdx == 0;

    cv::Mat highResMaskMat;
    cv::resize(predMask, highResMaskMat, cv::Size(_imageSize, _imageSize));

    // std::vector<float> highResMask((float*)highResMaskMat.data, (float*)highResMaskMat.data + highResMaskMat.total());
    std::vector<float> highResMask(highResMaskMat.begin<float>(), highResMaskMat.end<float>());
    std::vector<int64_t> highResMaskDims = {1, 1, _imageSize, _imageSize};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value visionFeaturesTensor = Ort::Value::CreateTensor<float>(memoryInfo, lowResFeatures, lowResFeaturesSize,
                                                            _memoryEncoderInputNodeDims[0].data(), _memoryEncoderInputNodeDims[0].size());
    Ort::Value highResMaskForMemTensor = Ort::Value::CreateTensor<float>(memoryInfo, highResMask.data(), _imageSize * _imageSize,
                                                            highResMaskDims.data(), highResMaskDims.size());
    std::vector<Ort::Value> memoryEncoderOutputTensors;
    memoryEncoderInference(visionFeaturesTensor,
                           highResMaskForMemTensor,
                           maskDecoderOutputTensors[3],
                           isMaskFromPts,
                           memoryEncoderOutputTensors);

    auto maskmemFeatures = memoryEncoderOutputTensors[0].GetTensorMutableData<float>();
    auto maskmemPosEnc   = memoryEncoderOutputTensors[1].GetTensorMutableData<float>();

    // 5) save memory bank
    size_t maskmemFeaturesSize = memoryEncoderOutputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); // 262144=4096*64
    size_t maskmemPosEncSize   = memoryEncoderOutputTensors[1].GetTensorTypeAndShapeInfo().GetElementCount(); // 262144=4096*64
    size_t objPtrSize          = _maskDecoderOutputNodeDims[2][2]; //1,3,256
    // std::cout << "maskmemFeaturesSize: " << maskmemFeaturesSize << std::endl;
    // std::cout << "maskmemPosEncSize: " << maskmemPosEncSize << std::endl;
    // std::cout << "objPtrSize: " << objPtrSize << std::endl;

    MemoryBankEntry entry;
    entry.maskmem_features = std::vector<float>(maskmemFeatures, maskmemFeatures + maskmemFeaturesSize);
    entry.maskmem_pos_enc  = std::vector<float>(maskmemPosEnc, maskmemPosEnc + maskmemPosEncSize);
    entry.obj_ptr          = std::vector<float>(objPtrs + bestIoUIndex * objPtrSize, objPtrs + (bestIoUIndex + 1) * objPtrSize);
    entry.best_iou_score   = bestIouScore;
    entry.obj_score_logits = objScoreLogits[0];
    entry.kf_score         = kfScore;

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
#endif

    return predMask;
}

void SAM2Tracker::preprocessImage(const cv::Mat& src, std::vector<float>& dest) {
    auto start = std::chrono::high_resolution_clock::now();

    // // 将图像转resize到指定大小, 并转换为float, 并减去均值, 除以方差
    // cv::Mat normalized;
    // src.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    // cv::subtract(normalized, _mean, normalized);
    // cv::divide(normalized, _std, normalized)
    // cv::Mat blob = cv::dnn::blobFromImage(normalized, 1.0, cv::Size(_imageSize, _imageSize), true, false);
    // dest.assign((float*)blob.data, (float*)blob.data + blob.total() * blob.channels());

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(_imageSize, _imageSize));
    cv::Mat rgbImage;
    cv::cvtColor(resized, rgbImage, cv::COLOR_BGR2RGB); // 转换为RGB
    rgbImage.convertTo(rgbImage, CV_32FC3); // 转换为float
    dest.assign((float*)rgbImage.data, (float*)rgbImage.data + rgbImage.total() * rgbImage.channels());

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "preprocessImage spent: " << duration.count() << " ms" << std::endl;
}

PostprocessResult SAM2Tracker::postprocessOutput(std::vector<Ort::Value>& maskDecoderOutputTensors) {
    // maskDecoderOutputTensors[0] : lowResMultiMasks，(3, videoW, videoH)
    // maskDecoderOutputTensors[1] : highResMultiMasks, (3, _imageSize, _imageSize)
    // maskDecoderOutputTensors[2] : ious, (3)
    // maskDecoderOutputTensors[3] : objPtr, (3, 256)
    // maskDecoderOutputTensors[4] : objScoreLogits, (1)
    // maskDecoderOutputTensors[5] : maskMemTposEnc, (7, 64)
    auto start = std::chrono::high_resolution_clock::now();

    auto lowResMultiMasks = maskDecoderOutputTensors[0].GetTensorMutableData<float>();
    // auto highResMultiMasks = maskDecoderOutputTensors[1].GetTensorMutableData<float>();
    auto ious = maskDecoderOutputTensors[1].GetTensorMutableData<float>();
    auto objPtr = maskDecoderOutputTensors[2].GetTensorMutableData<float>();
    auto objScoreLogits = maskDecoderOutputTensors[3].GetTensorMutableData<float>();
    auto maskMemTposEnc = maskDecoderOutputTensors[4].GetTensorMutableData<float>();

    int numMasks = _maskDecoderOutputNodeDims[1][1];

    // print ious
    std::cout << "ious: ";
    for (int i = 0; i < numMasks; i++) {
        std::cout << ious[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "objScoreLogits: " << *objScoreLogits << std::endl;

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

    if ((_kfMean.size() == 0 && _kfCovariance.size() == 0) || _stableFrameCount == 0) {
        bestIoUIndex = std::distance(ious, std::max_element(ious, ious + numMasks));
        bestIouScore = ious[bestIoUIndex];
        // auto highResMask = highResMultiMasks + bestIoUIndex * _imageSize * _imageSize;
        // cv::Mat predMask(_imageSize, _imageSize, CV_32FC1, highResMask);
        int lowResMaskSize = _maskDecoderOutputNodeDims[0][2];
        auto lowResMask = lowResMultiMasks + bestIoUIndex * lowResMaskSize * lowResMaskSize;
        cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1, lowResMask);

        cv::Mat binaryMask;
        cv::threshold(predMask, binaryMask, 0.01, 1.0, cv::THRESH_BINARY);

        cv::Rect bbox(0, 0, 0, 0);
        std::vector<cv::Point> nonZeroPoints;
        cv::findNonZero(binaryMask, nonZeroPoints);
        if (!nonZeroPoints.empty()) {
            bbox = cv::boundingRect(nonZeroPoints);
        }

        // std::cout << "bbox: [" << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]" << std::endl;

        std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.initiate(
                                                                        _kf.xyxy2xyah(Eigen::Vector4f(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height))
                                                                        );
        _kfMean = kfResult.first;
        _kfCovariance = kfResult.second;

        _stableFrameCount++;
    }
    else if (_stableFrameCount < _stableFramesThreshold)
    {
        std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.predict(_kfMean, _kfCovariance);
        _kfMean = kfResult.first;
        _kfCovariance = kfResult.second;

        bestIoUIndex = std::distance(ious, std::max_element(ious, ious + numMasks));
        bestIouScore = ious[bestIoUIndex];
        // auto highResMask = highResMultiMasks + bestIoUIndex * _imageSize * _imageSize;
        // cv::Mat predMask(_imageSize, _imageSize, CV_32FC1, highResMask);
        int lowResMaskSize = _maskDecoderOutputNodeDims[0][2];
        auto lowResMask = lowResMultiMasks + bestIoUIndex * lowResMaskSize * lowResMaskSize;
        cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1, lowResMask);

        cv::Mat binaryMask;
        cv::threshold(predMask, binaryMask, 0.01, 1.0, cv::THRESH_BINARY);

        cv::Rect bbox(0, 0, 0, 0);
        std::vector<cv::Point> nonZeroPoints;
        cv::findNonZero(binaryMask, nonZeroPoints);
        if (!nonZeroPoints.empty()) {
            bbox = cv::boundingRect(nonZeroPoints);
        }

        // std::cout << "bbox: [" << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]" << std::endl;

        if (bestIouScore > _stableIousThreshold) {
            std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.update(_kfMean, _kfCovariance,
                                                                            _kf.xyxy2xyah(Eigen::Vector4f(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height))
                                                                            );
            _kfMean = kfResult.first;
            _kfCovariance = kfResult.second;
            _stableFrameCount++;
        }
        else {
            _stableFrameCount = 0;
        }
    }
    else {
        std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.predict(_kfMean, _kfCovariance);
        _kfMean = kfResult.first;
        _kfCovariance = kfResult.second;

        std::vector<Eigen::Vector4f> predBboxs;
        for (int i = 0; i < numMasks; i++) {
            // auto highResMask = highResMultiMasks + i * _imageSize * _imageSize;
            // cv::Mat predMask(_imageSize, _imageSize, CV_32FC1, highResMask);
            int lowResMaskSize = _maskDecoderOutputNodeDims[0][2];
            auto lowResMask = lowResMultiMasks + i * lowResMaskSize * lowResMaskSize;
            cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1, lowResMask);

            cv::Mat binaryMask;
            cv::threshold(predMask, binaryMask, 0.01, 1.0, cv::THRESH_BINARY);

            cv::Rect bbox(0, 0, 0, 0);
            std::vector<cv::Point> nonZeroPoints;
            cv::findNonZero(binaryMask, nonZeroPoints);
            if (!nonZeroPoints.empty()) {
                bbox = cv::boundingRect(nonZeroPoints);
            }

            predBboxs.push_back(Eigen::Vector4f(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height));
        }

        std::vector<float> kfIousVec = _kf.computeIoUs(_kfMean.head(4), predBboxs);

        std::vector<float> weightedIous;
        for (int i = 0; i < numMasks; i++) {
            weightedIous.push_back(_kfScoreWeight * kfIousVec[i] + (1 - _kfScoreWeight) * ious[i]);
        }

        bestIoUIndex = std::distance(weightedIous.begin(), std::max_element(weightedIous.begin(), weightedIous.end()));
        bestIouScore = ious[bestIoUIndex];
        kfScore = kfIousVec[bestIoUIndex];

        if (bestIouScore > _stableIousThreshold) {
            std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.update(_kfMean, _kfCovariance,
                                                                                _kf.xyxy2xyah(predBboxs[bestIoUIndex])
                                                                                );
            _kfMean = kfResult.first;
            _kfCovariance = kfResult.second;
        }
        else {
            _stableFrameCount = 0;
        }
    }
    
#endif

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "postprocess spent: " << duration.count() << " ms" << std::endl;

    return {bestIoUIndex, bestIouScore, kfScore};
}