#include <kalman_filter.h>

/**
 * @brief
 * 
 * 
 * A simple Kalman filter for tracking bounding boxes in image space.
 * 
 * The 8-dimensional state space
 * 
 *     x, y, a, h, vx, vy, va, vh
 * 
 * contains the bounding box center position (x, y), aspect ratio a, height h,
 * and their respective velocities.
 * 
 * Object motion follows a constant velocity model. The bounding box location
 * (x, y, a, h) is taken as direct observation of the state space (linear
 * observation model).
 * 
*/
KalmanFilter::KalmanFilter() {
    int ndim = 4;
    float dt = 1.f;

    // Initialize motion matrix
    _motionMat = Eigen::MatrixXf::Identity(2 * ndim, 2 * ndim);
    for (int i = 0; i < ndim; i++) {
        _motionMat(i, ndim + i) = dt;
    }

    // Initialize update matrix
    _updateMat = Eigen::MatrixXf::Identity(ndim, 2 * ndim);

    // Initialize uncertainty weights
    _stdWeightPosition = 1.0 / 20;
    _stdWeightVelocity = 1.0 / 160;
}

/**
 * @brief Create track from unassociated measurement.
 * 
 * @param measurement Bounding box coordinates (x, y, a, h) with center position (x, y),aspect ratio a, and height h.
 * @return std::pair<Eigen::VectorXf, Eigen::MatrixXf> 
 * Returns the mean vector (8 dimensional) and covariance matrix (8x8dimensional) of the new track. Unobserved velocities are initialized to 0 mean.
 */
std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::initiate(const Eigen::VectorXf& measurement) {
    Eigen::VectorXf meanPos = measurement;
    Eigen::VectorXf meanVel = Eigen::VectorXf::Zero(meanPos.size());
    Eigen::VectorXf mean(meanPos.size() + meanVel.size());
    mean << meanPos, meanVel;

    Eigen::VectorXf std(mean.size());

    std << 2 * _stdWeightPosition * measurement(3),
           2 * _stdWeightPosition * measurement(3),
           1e-2,
           2 * _stdWeightPosition * measurement(3),
           10 * _stdWeightVelocity * measurement(3),
           10 * _stdWeightVelocity * measurement(3),
           1e-5,
           10 * _stdWeightVelocity * measurement(3);

    Eigen::MatrixXf covariance = std.array().square().matrix().asDiagonal();
    return {mean, covariance};
}

/**
 * @brief Run Kalman filter prediction step.
 * 
 * @param mean The 8 dimensional mean vector of the object state at the previous time step.
 * @param covariance The 8x8 dimensional covariance matrix of the object state at the previous time step.
 * @return std::pair<Eigen::VectorXf, Eigen::MatrixXf> Returns the mean vector and covariance matrix of the predicted state. 
 *                                                     Unobserved velocities are initialized to 0 mean.
 */
std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::predict(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance) {
    Eigen::VectorXf stdPos(4);
    stdPos << _stdWeightPosition * mean(3),
              _stdWeightPosition * mean(3),
              1e-2,
              _stdWeightPosition * mean(3);

    Eigen::VectorXf stdVel(4);
    stdVel << _stdWeightVelocity * mean(3),
              _stdWeightVelocity * mean(3),
              1e-5,
              _stdWeightVelocity * mean(3);

    Eigen::VectorXf stdAll(stdPos.size() + stdVel.size());
    stdAll << stdPos, stdVel;
    Eigen::MatrixXf motionCov = stdAll.array().square().matrix().asDiagonal();

    Eigen::VectorXf predictedMean = _motionMat * mean;
    Eigen::MatrixXf predictedCov = _motionMat * covariance * _motionMat.transpose() + motionCov;

    return {predictedMean, predictedCov};
}

/**
 * @brief Project state distribution to measurement space.
 * 
 * @param mean The state's mean vector (8 dimensional array).
 * @param covariance The state's covariance matrix (8x8 dimensional).
 * @return std::pair<Eigen::VectorXf, Eigen::MatrixXf> Returns the projected mean and covariance matrix of the given state estimate.
 */
std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::project(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance) {
    Eigen::VectorXf std(4);
    std << _stdWeightPosition * mean(3),
           _stdWeightPosition * mean(3),
           1e-1,
           _stdWeightPosition * mean(3);

    Eigen::MatrixXf innovationCov = std.array().square().matrix().asDiagonal();

    Eigen::VectorXf projected_mean = _updateMat * mean;
    Eigen::MatrixXf projected_cov = _updateMat * covariance * _updateMat.transpose() + innovationCov;

    return {projected_mean, projected_cov};
}

/**
 * @brief Run Kalman filter correction step.
 * 
 * @param mean The predicted state's mean vector (8 dimensional).
 * @param covariance The state's covariance matrix (8x8 dimensional).
 * @param measurement The 4 dimensional measurement vector (x, y, a, h), where (x, y)
 *                    is the center position, a the aspect ratio, and h the height of the
 *                    bounding box.
 * @return std::pair<Eigen::VectorXf, Eigen::MatrixXf> Returns the measurement-corrected state distribution.
 */
std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::update(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance, const Eigen::VectorXf& measurement) {
    // auto [projected_mean, projected_cov] = project(mean, covariance);
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> projected = project(mean, covariance);
    Eigen::VectorXf projectedMean = projected.first;
    Eigen::MatrixXf projectedCov = projected.second;

    // Eigen::MatrixXf kalman_gain = covariance * _updateMat.transpose() * projected_cov.inverse();
    Eigen::LLT<Eigen::MatrixXf> lltOfS(projectedCov); // compute the Cholesky decomposition of projected_cov
    Eigen::MatrixXf kalmanGain = lltOfS.solve((covariance * _updateMat.transpose()).transpose()).transpose();

    Eigen::VectorXf innovation = measurement - projectedMean;

    Eigen::VectorXf newMean = mean + kalmanGain * innovation;
    Eigen::MatrixXf newCovariance = covariance - kalmanGain * projectedCov * kalmanGain.transpose();

    return {newMean, newCovariance};
}

/**
 * @brief Compute the IoU between the bbox and the bboxes
 * 
 * @param predBbox
 * @param bboxes 
 * @return std::vector<float> 
 */
std::vector<float> KalmanFilter::computeIoUs(const Eigen::Vector4f& predBbox, const std::vector<Eigen::Vector4f>& bboxes) {
    std::vector<float> ious;

    // transform the predicted bbox to xyxy format
    Eigen::Vector4f predBboxXyxy = xyah2xyxy(predBbox);

    for (int i = 0; i < bboxes.size(); i++) {
        float iou = computeIoU(predBboxXyxy, bboxes[i]);
        ious.push_back(iou);
    }

    return ious;
}

/**
 * @brief Compute the Intersection over Union (IoU) of two bounding boxes.
 * 
 * @param bbox1 The first bounding box in the format (xmin, ymin, xmax, ymax).
 * @param bbox2 The second bounding box in the format (xmin, ymin, xmax, ymax).
 * @return float The IoU of the two bounding boxes.
 */
float KalmanFilter::computeIoU(const Eigen::Vector4f& bbox1, const Eigen::Vector4f& bbox2) {
    // if (bbox1.size() != 4 || bbox2.size() != 4) {
    //     std::cout << "Bounding boxes must have 4 elements: [x1, y1, x2, y2]" << std::endl;
    //     return 0.f;
    // }
    // if bbox2 全为0, 则返回0
    if (bbox2(0) == 0 && bbox2(1) == 0 && bbox2(2) == 0 && bbox2(3) == 0) {
        return 0.f;
    }

    // Calculate the intersection area
    float x1 = std::max(bbox1(0), bbox2(0));
    float y1 = std::max(bbox1(1), bbox2(1));
    float x2 = std::min(bbox1(2), bbox2(2));
    float y2 = std::min(bbox1(3), bbox2(3));

    float intersection = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);

    // Calculate the union area
    float area1 = (bbox1(2) - bbox1(0)) * (bbox1(3) - bbox1(1));
    float area2 = (bbox2(2) - bbox2(0)) * (bbox2(3) - bbox2(1));
    float unionArea = area1 + area2 - intersection;

    float iou;
    if (unionArea == 0) {
        iou = 0;
    } else {
        iou = intersection / unionArea;
    }

    return iou;
}


/**
 * @brief Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (xc, yc, a, h).
 * 
 * @param xyxy The bounding box coordinates in format (xmin, ymin, xmax, ymax).
 * @return Eigen::VectorXf The bounding box coordinates in format (xc, yc, a, h).
 */
Eigen::Vector4f KalmanFilter::xyxy2xyah(const Eigen::Vector4f& xyxy) {
    // // check xyxy size
    // if (xyxy.size() != 4) {
    //     std::cout << "Input vector must have 4 elements: [x1, y1, x2, y2]" << std::endl;
    //     return Eigen::VectorXf::Zero(4);
    // }

    float x1 = xyxy(0);
    float y1 = xyxy(1);
    float x2 = xyxy(2);
    float y2 = xyxy(3);

    // 计算中心点坐标 (xc, yc)
    float xc = (x1 + x2) / 2.0f;
    float yc = (y1 + y2) / 2.0f;

    // 计算宽度和高度
    float w = x2 - x1;
    float h = y2 - y1;
    if (h == 0) {
        h = 1;
    }

    Eigen::Vector4f xyah;
    xyah << xc, yc, w / h, h;
    
    return xyah;
}

/**
 * @brief Convert bounding boxes from format (xc, yc, a, h) to (xmin, ymin, xmax, ymax).
 * 
 * @param xyah The bounding box coordinates in format (xc, yc, a, h).
 * @return Eigen::VectorXf The bounding box coordinates in format (xmin, ymin, xmax, ymax).
 */
Eigen::Vector4f KalmanFilter::xyah2xyxy(const Eigen::Vector4f& xyah) {
    // // check xywh size
    // if (xyah.size() != 4) {
    //     std::cout << "Input vector must have 4 elements: [xc, yc, a, h]" << std::endl;
    //     return Eigen::VectorXf::Zero(4);
    // }

    float xc = xyah(0);
    float yc = xyah(1);
    float a = xyah(2);
    float h = xyah(3);

    // 计算左上角和右下角坐标
    float w = h * a;
    float x1 = xc - w / 2.0f;
    float y1 = yc - h / 2.0f;
    float x2 = xc + w / 2.0f;
    float y2 = yc + h / 2.0f;

    Eigen::Vector4f xyxy;
    xyxy << x1, y1, x2, y2;

    return xyxy;
}