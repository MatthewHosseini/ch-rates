\
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "rates_kf_sabr/diebold_li.hpp"
#include "rates_kf_sabr/ou_discretize.hpp"
#include "rates_kf_sabr/kalman.hpp"
#include "rates_kf_sabr/sabr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rates_kf_sabr, m) {
    m.doc() = "C++ core for Diebold–Li + Kalman + OU discretization + SABR.";

    m.def("diebold_li_loadings", &diebold_li_loadings,
          py::arg("taus"), py::arg("lambda"),
          "Return Diebold–Li loading matrix H(lambda) for a vector of maturities (taus).");

    m.def("discretize_ou_van_loan",
          [](const Eigen::MatrixXd& kappa,
             const Eigen::VectorXd& theta,
             const Eigen::MatrixXd& Sigma,
             double dt) {
                DiscreteOU d = discretize_ou_van_loan(kappa, theta, Sigma, dt);
                return py::make_tuple(d.F, d.C, d.Q);
          },
          py::arg("kappa"), py::arg("theta"), py::arg("Sigma"), py::arg("dt"),
          "Exact OU discretization (F,C,Q) via Van Loan block matrix exponential.");

    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&,
                      const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&>(),
             py::arg("H"), py::arg("F"), py::arg("C"), py::arg("Q"), py::arg("R"), py::arg("x0"), py::arg("P0"))
        .def("step", &KalmanFilter::step, py::arg("y"))
        .def_property_readonly("state", &KalmanFilter::state)
        .def_property_readonly("cov", &KalmanFilter::cov);

    m.def("sabr_iv_hagan", &sabr_iv_hagan,
          py::arg("F"), py::arg("K"), py::arg("T"),
          py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"),
          py::arg("shift") = 0.0,
          "HKLW SABR implied volatility approximation (optionally shifted).");

    m.def("sabr_calibrate_beta_fixed",
          [](double F, double T, double beta,
             const std::vector<double>& strikes,
             const std::vector<double>& vols,
             double alpha0, double rho0, double nu0,
             double shift,
             int max_iters,
             double step) {
                auto res = sabr_calibrate_beta_fixed(F, T, beta, strikes, vols, alpha0, rho0, nu0, shift, max_iters, step);
                return py::make_tuple(res.alpha, res.rho, res.nu, res.iters, res.rmse);
          },
          py::arg("F"), py::arg("T"), py::arg("beta"),
          py::arg("strikes"), py::arg("vols"),
          py::arg("alpha0")=0.02, py::arg("rho0")=-0.2, py::arg("nu0")=0.5,
          py::arg("shift")=0.0, py::arg("max_iters")=3000, py::arg("step")=5e-3,
          "Calibrate SABR to (K,vol) with beta fixed; returns (alpha,rho,nu,iters,rmse).");
}
