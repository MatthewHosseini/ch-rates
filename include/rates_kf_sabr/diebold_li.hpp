\
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>

// Diebold–Li (time-varying Nelson–Siegel) measurement loading matrix H(lambda)
// y(tau) = [1, (1-exp(-tau/l))/ (tau/l), (1-exp(-tau/l))/(tau/l) - exp(-tau/l)] * [beta1 beta2 beta3]^T
inline Eigen::MatrixXd diebold_li_loadings(const std::vector<double>& taus, double lambda) {
    const int m = static_cast<int>(taus.size());
    Eigen::MatrixXd H(m, 3);
    for (int i = 0; i < m; ++i) {
        const double tau = taus[i];
        const double x = tau / lambda;
        const double e = std::exp(-x);
        const double f = (x == 0.0) ? 1.0 : (1.0 - e) / x;
        H(i,0) = 1.0;
        H(i,1) = f;
        H(i,2) = f - e;
    }
    return H;
}
