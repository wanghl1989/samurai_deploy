#ifndef SAM2_TRACKER_H
#define SAM2_TRACKER_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <chrono>
#include <omp.h>
#include <unistd.h>

#include "kalman_filter.h"

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

class SAM2Tracker {
public:
    SAM2Tracker() {}
    SAM2Tracker(const std::string& modelPath, bool useGPU, bool enableFp16);
    ~SAM2Tracker() {}

    void loadNetwork(const std::string& modelPath, bool useGPU, bool enableFp16);

    cv::Mat addFirstFrameBbox(int frameIdx, const cv::Mat& firstFrame, const cv::Rect& bbox);

    cv::Mat trackStep(int frameIdx, const cv::Mat& frame);

    void imageEncoderInference(std::vector<float>& frame, std::vector<Ort::Value>& imageEncoderOutputTensors);

    void memoryAttentionInference(int frameIdx, 
                                  std::vector<Ort::Value>& imageEncoderOutputTensors,
                                  std::vector<Ort::Value>& memoryAttentionOutputTensors);

    void maskDecoderInference(std::vector<float>& inputPoints,
                              std::vector<int32_t>& inputLabels,
                              std::vector<Ort::Value>& imageEncoderOutputTensors,
                              Ort::Value& pixFeatWithMem,
                              std::vector<Ort::Value>& maskDecoderOutputTensors);

    void memoryEncoderInference(Ort::Value& visionFeaturesTensor,
                                Ort::Value& highResMasksForMemTensor,
                                Ort::Value& objectScoreLogitsTensor,
                                bool isMaskFromPts,
                                std::vector<Ort::Value>& memoryEncoderOutputTensors);

    void preprocessImage(const cv::Mat& src, std::vector<float>& dest);

    PostprocessResult postprocessOutput(std::vector<Ort::Value>& maskDecoderOutputTensors);

private:
    void getModelInfo(const Ort::Session* session, const std::string& modelName,
                        std::vector<const char*>& inputNodeNames,
                        std::vector<const char*>& outputNodeNames,
                        std::vector<std::vector<int64_t>>& inputNodeDims,
                        std::vector<std::vector<int64_t>>& outputNodeDims);
    void printDataType(ONNXTensorElementDataType type);

    // ONNXRuntime related
    Ort::Env _env{ORT_LOGGING_LEVEL_WARNING, "SAM2Tracker"};
    Ort::SessionOptions _sessionOptions{nullptr};

    std::unique_ptr<Ort::Session> _imageEncoderSession{nullptr};
    std::unique_ptr<Ort::Session> _memoryAttentionSession{nullptr};
    std::unique_ptr<Ort::Session> _maskDecoderSession{nullptr};
    std::unique_ptr<Ort::Session> _memoryEncoderSession{nullptr};

    std::vector<Ort::AllocatedStringPtr> _inputNodeNameAllocatedStrings;
    std::vector<const char*> _imageEncoderInputNodeNames;
    std::vector<const char*> _memoryAttentionInputNodeNames;
    std::vector<const char*> _maskDecoderInputNodeNames;
    std::vector<const char*> _memoryEncoderInputNodeNames;

    std::vector<Ort::AllocatedStringPtr> _outputNodeNameAllocatedStrings;
    std::vector<const char*> _imageEncoderOutputNodeNames;
    std::vector<const char*> _memoryAttentionOutputNodeNames;
    std::vector<const char*> _maskDecoderOutputNodeNames;
    std::vector<const char*> _memoryEncoderOutputNodeNames;

    std::vector<std::vector<int64_t>> _imageEncoderInputNodeDims;
    std::vector<std::vector<int64_t>> _memoryAttentionInputNodeDims;
    std::vector<std::vector<int64_t>> _maskDecoderInputNodeDims;
    std::vector<std::vector<int64_t>> _memoryEncoderInputNodeDims;

    std::vector<std::vector<int64_t>> _imageEncoderOutputNodeDims;
    std::vector<std::vector<int64_t>> _memoryAttentionOutputNodeDims;
    std::vector<std::vector<int64_t>> _maskDecoderOutputNodeDims;
    std::vector<std::vector<int64_t>> _memoryEncoderOutputNodeDims;

    cv::Scalar _mean = cv::Scalar(0.485, 0.456, 0.406);
    cv::Scalar _std = cv::Scalar(0.229, 0.224, 0.225);
    int _imageSize = 512;
    int _videoWidth = 0;
    int _videoHeight = 0;

    // maskmem_tpos_enc
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
