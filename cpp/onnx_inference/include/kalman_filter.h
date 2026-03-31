// kalman_filter.h
#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include <unordered_map>


// Chi-square distribution table for 0.95 quantile
const std::unordered_map<int, float> chi2inv95 = {
    {1, 3.8415}, {2, 5.9915}, {3, 7.8147}, {4, 9.4877},
    {5, 11.070}, {6, 12.592}, {7, 14.067}, {8, 15.507}, {9, 16.919}
};

class KalmanFilter {
public:
    KalmanFilter();
    ~KalmanFilter() {};

    // Initialize the state with a measurement
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> initiate(const Eigen::VectorXf& measurement);

    // Predict the next state
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> predict(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance);

    // Project the state to measurement space
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> project(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance);

    // Update the state with a measurement
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> update(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance, const Eigen::VectorXf& measurement);

    std::vector<float> computeIoUs(const Eigen::Vector4f& predBbox, const std::vector<Eigen::Vector4f>& bboxes);

    Eigen::Vector4f xyxy2xyah(const Eigen::Vector4f& xyxy);
    Eigen::Vector4f xyah2xyxy(const Eigen::Vector4f& xywh);

private:
    Eigen::MatrixXf _motionMat; // State transition matrix
    Eigen::MatrixXf _updateMat; // Observation matrix
    float _stdWeightPosition; // Position uncertainty weight
    float _stdWeightVelocity; // Velocity uncertainty weight
    
    float computeIoU(const Eigen::Vector4f& bbox1, const Eigen::Vector4f& bbox2);
};
#endif // KALMAN_FILTER_H