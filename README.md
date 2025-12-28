# Rates Term Structure (Diebold–Li + Kalman + OU) and SABR 

This repository is a **front-office–style model development artifact**:
- **Diebold–Li (time-varying Nelson–Siegel) measurement** and **Kalman filtering** for yield curve factors
- **OU/Hull–White factor dynamics** with **exact discretization via Van Loan**
- **SABR implied vol (HKLW)** and a simple calibration routine (beta fixed by default)
- **C++ library** (Eigen) exposed to Python via **pybind11**, demonstrated in Jupyter

The design mirrors a common FO quant workflow: **compiled numerics + Python orchestration**.

## (1) Quick start (local)
### Prereqs
- CMake >= 3.20
- A C++17 compiler
- Python 3.10+
- Eigen3
- pybind11

### Build + install the Python module (editable)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Run the demo notebook
```bash
jupyter lab
# open notebooks/01_kf_ou_sabr_demo.ipynb
```

## (2) Repo structure
- `include/` C++ headers (Diebold–Li loadings, OU discretization, Kalman, SABR)
- `src/` pybind11 bindings
- `python/` packaging for `pip install -e .`
- `notebooks/` demo notebook (KF factors + OU scenario + SABR fit)
- `tests/` lightweight pytest checks
- `.github/workflows/` CI template

## (3) The Idea

This project mirrors a common **front-office rates quant** stack: a **curve model** to generate consistent forwards/discounting, a **state-space dynamics model** for time evolution and scenario generation, and a **volatility smile model** for options. In practice, **term structure** and **volatility** are distinct but coupled components of the same pricing system.

### 3.1 Curve: Nelson–Siegel / Diebold–Li as a state-space measurement model

We represent the curve (zero yields or par yields by maturity) using **Nelson–Siegel** factor loadings. Diebold–Li treats the Nelson–Siegel betas as **time-varying latent factors**.

For maturities $ \tau_1,\dots,\tau_n $ and time $t$, the measurement equation is:
$$
\mathbf{y}_t = \mathbf{H}(\lambda)\,\mathbf{x}_t + \boldsymbol{\varepsilon}_t,
\qquad
\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}),
$$
where:
- $\mathbf{y}_t \in \mathbb{R}^{n}$ are observed yields at maturities $\tau$,
- $\mathbf{x}_t = [\beta_{0,t}, \beta_{1,t}, \beta_{2,t}]^\top$ are latent **level/slope/curvature** factors,
- $\lambda$ controls the factor loading decay.

The (Nelson–Siegel / Diebold–Li) loading matrix $\mathbf{H}(\lambda) \in \mathbb{R}^{n \times 3}$ has rows:
$$
\mathbf{H}_i(\lambda) =
\begin{bmatrix}
1 &
\frac{1 - e^{-\lambda \tau_i}}{\lambda \tau_i} &
\frac{1 - e^{-\lambda \tau_i}}{\lambda \tau_i} - e^{-\lambda \tau_i}
\end{bmatrix}.
$$
This gives a compact, smooth curve representation while maintaining a clear factor interpretation.

### 3.2 Dynamics: OU / Hull–White-style factor evolution and Kalman filtering

We model factors as mean-reverting (OU) processes in state space. In continuous time:
$$
d\mathbf{x}_t = \mathbf{K}(\boldsymbol{\theta} - \mathbf{x}_t)\,dt + \mathbf{\Sigma}\, d\mathbf{W}_t,
$$
where $\mathbf{K}$ controls mean-reversion and $\mathbf{\Sigma}$ controls factor shocks.

The discrete-time state transition used by the Kalman filter is:
$$
\mathbf{x}_{t+\Delta} = \mathbf{A}\,\mathbf{x}_t + \mathbf{c} + \boldsymbol{\eta}_t,
\qquad
\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}),
$$
with
$$
\mathbf{A} = e^{-\mathbf{K}\Delta},
\qquad
\mathbf{c} = (\mathbf{I}-\mathbf{A})\,\boldsymbol{\theta}.
$$
The process covariance $\mathbf{Q}$ is computed exactly via the **Van Loan** method. Define the block matrix:
$$
\mathbf{M} =
\begin{bmatrix}
-\mathbf{K} & \mathbf{\Sigma}\mathbf{\Sigma}^\top \\
\mathbf{0} & \mathbf{K}^\top
\end{bmatrix},
\qquad
e^{\mathbf{M}\Delta} =
\begin{bmatrix}
\mathbf{E}_{11} & \mathbf{E}_{12} \\
\mathbf{0} & \mathbf{E}_{22}
\end{bmatrix}.
$$
Then:
$$
\mathbf{A} = \mathbf{E}_{22}^\top,
\qquad
\mathbf{Q} = \mathbf{A}\,\mathbf{E}_{12}.
$$
This gives an exact OU discretization, which is important for stable inference and scenario generation.

### 3.3 Why NS/NSS (curve) and SABR (smile) belong together

**NS/NSS (curve) and SABR (volatility) solve different components of the same pricing problem**:
- The curve determines **discounting** and the relevant **forward** underlying for options (e.g., forward swap rate or forward rate).
- SABR models the **implied volatility smile** across strikes (skew/convexity) required for nonlinear rates products.

Concretely, for an option with expiry $T$, SABR is parameterized in terms of a forward $F$ and strike $K$. In a rates context:
- $F$ is derived from the curve (e.g., forward swap rate $S(0)$ or forward rate $L(0)$),
- PV requires discounting via curve-implied discount factors.

A practical FO workflow therefore looks like:
$$
\text{Curve build} \Rightarrow (P(0,T),\,F) \Rightarrow \text{Vol smile calibration} \Rightarrow \text{Option price/risk}.
$$

NS/NSS is particularly convenient here as it provides:
- a **smooth** curve representation,
- a **low-dimensional factorization** (good for filtering/scenarios),
- a natural link to the Diebold–Li state-space formulation.

### 3.4 SABR implied vol (HKLW / Hagan approximation) and calibration

For a given expiry $T$, forward $F$, strike $K$, and parameters $(\alpha,\beta,\rho,\nu)$, the Hagan (HKLW) approximation defines an implied volatility surface.

Define:
$$
z = \frac{\nu}{\alpha}\,(F K)^{\frac{1-\beta}{2}} \ln\!\left(\frac{F}{K}\right),
\qquad
\chi(z) = \ln\!\left(\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1-\rho}\right).
$$
Then the implied vol (non-ATM form) is:
$$
\sigma_{\text{impl}}(F,K) \approx
\frac{\alpha}{(F K)^{\frac{1-\beta}{2}}}
\frac{z}{\chi(z)}
\left[1 + \left(
\frac{(1-\beta)^2}{24}\frac{\alpha^2}{(F K)^{1-\beta}}
+ \frac{\rho\beta\nu\alpha}{4(F K)^{\frac{1-\beta}{2}}}
+ \frac{2-3\rho^2}{24}\nu^2
\right)T\right].
$$
At-the-money ($K=F$) is handled with the standard ATM expansion to avoid numerical issues.

In this repo, we **fix $\beta$ by default** (common in practice) and calibrate $(\alpha,\rho,\nu)$ to observed implied vols by minimizing a simple least-squares objective:
$$
\min_{\alpha,\rho,\nu}\;\sum_{j}\left(\sigma_{\text{impl}}(F,K_j; \alpha,\beta,\rho,\nu) - \sigma^{\text{mkt}}_j\right)^2.
$$

### 3.5 What the demo notebook is demonstrating

The demo notebook (`notebooks/01_kf_ou_sabr_demo.ipynb`) ties the pieces into an end-to-end workflow:

1) **Loadings + Kalman filter** infer time-varying curve factors $\mathbf{x}_t$.  
2) **OU/Van Loan discretization** produces consistent factor scenarios.  
3) **SABR IV + calibration** fits a smile on a chosen expiry using curve-consistent forwards.  

The result is a coherent “FO-style” pipeline: **curve factors + dynamics + vol smile**, implemented with a **C++ numerical core** and Python orchestration.
