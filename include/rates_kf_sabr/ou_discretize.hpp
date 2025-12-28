\
#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// Exact OU/Hull–White discretization using Van Loan block matrix exponential.
// For VAR(1): X_t = C + F X_{t-1} + v_t,  v_t ~ N(0,Q)
//
// F = exp(-kappa * dt)
// C = (I - F) * theta
// Q extracted via Van Loan: build M = [[-kappa, S],[0, kappa^T]] where S = Sigma Sigma^T,
// E = exp(M dt) = [[A, G],[0, A^{-T}]], and Q = A * G * A^T.
//
// See the derivation and implementation sketch in the attached note. 
// (In particular the block construction and read-off of F and Q.)
struct DiscreteOU {
    Eigen::MatrixXd F;  // (n x n)
    Eigen::VectorXd C;  // (n)
    Eigen::MatrixXd Q;  // (n x n)
};

inline DiscreteOU discretize_ou_van_loan(const Eigen::MatrixXd& kappa,
                                        const Eigen::VectorXd& theta,
                                        const Eigen::MatrixXd& Sigma,
                                        double dt) {
    const int n = static_cast<int>(kappa.rows());
    DiscreteOU out;
    const Eigen::MatrixXd S = Sigma * Sigma.transpose();

    out.F = (-kappa * dt).exp();
    out.C = (Eigen::MatrixXd::Identity(n,n) - out.F) * theta;

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(2*n, 2*n);
    M.block(0,0,n,n) = -kappa;
    M.block(0,n,n,n) = S;
    M.block(n,n,n,n) = kappa.transpose();

    Eigen::MatrixXd E = (M * dt).exp();
    Eigen::MatrixXd A = E.block(0,0,n,n);      // == out.F
    Eigen::MatrixXd G = E.block(0,n,n,n);

    out.Q = A * G * A.transpose();
    out.Q = 0.5 * (out.Q + out.Q.transpose()); // symmetrize
    return out;
}
