\
#pragma once
#include <cmath>
#include <algorithm>
#include <vector>

// Hagan–Kumar–Lesniewski–Woodward (HKLW) implied vol approximation.
// Handles lognormal SABR by default; for near/through-zero rates you can apply a shift and/or set beta=0.
inline double sabr_iv_hagan(double F, double K, double T,
                           double alpha, double beta, double rho, double nu,
                           double shift = 0.0) {
    // Apply shift for negative rates if desired
    const double Fs = F + shift;
    const double Ks = K + shift;

    const double eps = 1e-12;
    const double one_minus_beta = 1.0 - beta;

    // ATM special case
    if (std::fabs(Fs - Ks) < 1e-10) {
        const double FK = std::max(Fs, eps);
        const double FK_pow = std::pow(FK, one_minus_beta);
        const double term1 = (one_minus_beta * one_minus_beta / 24.0) * (alpha * alpha) / (FK_pow * FK_pow);
        const double term2 = (rho * beta * nu * alpha) / (4.0 * FK_pow);
        const double term3 = (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0;
        return (alpha / FK_pow) * (1.0 + (term1 + term2 + term3) * T);
    }

    const double logFK = std::log(Fs / Ks);
    const double FK = Fs * Ks;
    const double FK_pow = std::pow(FK, one_minus_beta / 2.0);

    const double z = (nu / std::max(alpha, eps)) * FK_pow * logFK;
    const double xz = std::log( (std::sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho) );

    const double A = 1.0 + (one_minus_beta * one_minus_beta / 24.0) * (logFK * logFK)
                           + (std::pow(one_minus_beta, 4) / 1920.0) * std::pow(logFK, 4);

    const double term1 = (one_minus_beta * one_minus_beta / 24.0) * (alpha * alpha) / (FK_pow * FK_pow);
    const double term2 = (rho * beta * nu * alpha) / (4.0 * FK_pow);
    const double term3 = (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0;

    const double B = 1.0 + (term1 + term2 + term3) * T;

    return (alpha / (FK_pow)) * (z / std::max(xz, eps)) * A * B;
}

// Simple calibration: beta fixed; fit (alpha, rho, nu) to a set of (K, vol) at given (F,T).
// This is demonstrative and intentionally lightweight; extend with robust constraints/optimizers as needed.
struct SabrFitResult {
    double alpha;
    double rho;
    double nu;
    int iters;
    double rmse;
};

inline SabrFitResult sabr_calibrate_beta_fixed(double F, double T, double beta,
                                               const std::vector<double>& strikes,
                                               const std::vector<double>& vols,
                                               double alpha0, double rho0, double nu0,
                                               double shift = 0.0,
                                               int max_iters = 3000,
                                               double step = 5e-3) {
    // Basic coordinate search with shrinking step (robust enough for a demo; replace in production).
    auto loss = [&](double a, double r, double n) {
        r = std::min(0.999, std::max(-0.999, r));
        a = std::max(1e-8, a);
        n = std::max(1e-8, n);
        double sse = 0.0;
        for (size_t i = 0; i < strikes.size(); ++i) {
            const double model = sabr_iv_hagan(F, strikes[i], T, a, beta, r, n, shift);
            const double e = model - vols[i];
            sse += e * e;
        }
        return sse / std::max<size_t>(1, strikes.size());
    };

    double a = alpha0, r = rho0, n = nu0;
    double best = loss(a,r,n);
    int it = 0;
    double h = step;

    while (it < max_iters) {
        bool improved = false;
        // try +/- in each coordinate
        const double candidates[6][3] = {
            {a*(1+h), r, n}, {a*(1-h), r, n},
            {a, r+h, n},     {a, r-h, n},
            {a, r, n*(1+h)}, {a, r, n*(1-h)},
        };
        for (auto &c : candidates) {
            double a1=c[0], r1=c[1], n1=c[2];
            double v = loss(a1,r1,n1);
            if (v < best) {
                a=a1; r=r1; n=n1; best=v;
                improved = true;
            }
        }
        if (!improved) {
            h *= 0.7; // shrink
            if (h < 1e-6) break;
        }
        ++it;
    }

    return SabrFitResult{a, std::min(0.999, std::max(-0.999, r)), n, it, std::sqrt(best)};
}
