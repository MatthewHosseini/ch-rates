\
#pragma once
#include <Eigen/Dense>

// Minimal linear-Gaussian Kalman Filter:
// Measurement: y_t = H x_t + eps_t, eps_t ~ N(0, R)
// Transition:  x_t = C + F x_{t-1} + v_t, v_t ~ N(0, Q)
//
// Steps (innovation, gain, update) match the standard recursion.
class KalmanFilter {
public:
    KalmanFilter(const Eigen::MatrixXd& H,
                 const Eigen::MatrixXd& F,
                 const Eigen::VectorXd& C,
                 const Eigen::MatrixXd& Q,
                 const Eigen::MatrixXd& R,
                 const Eigen::VectorXd& x0,
                 const Eigen::MatrixXd& P0)
        : H_(H), F_(F), C_(C), Q_(Q), R_(R), x_(x0), P_(P0) {}

    // One update given observation y_t. Returns filtered state x_{t|t}.
    Eigen::VectorXd step(const Eigen::VectorXd& y) {
        // Predict
        Eigen::VectorXd x_pred = C_ + F_ * x_;
        Eigen::MatrixXd P_pred = F_ * P_ * F_.transpose() + Q_;

        // Innovation
        Eigen::VectorXd nu = y - H_ * x_pred;
        Eigen::MatrixXd S = H_ * P_pred * H_.transpose() + R_;

        // Gain
        Eigen::MatrixXd K = P_pred * H_.transpose() * S.inverse();

        // Update
        x_ = x_pred + K * nu;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P_.rows(), P_.cols());
        P_ = (I - K * H_) * P_pred;

        return x_;
    }

    const Eigen::VectorXd& state() const { return x_; }
    const Eigen::MatrixXd& cov() const { return P_; }

private:
    Eigen::MatrixXd H_, F_, Q_, R_;
    Eigen::VectorXd C_;
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
};
