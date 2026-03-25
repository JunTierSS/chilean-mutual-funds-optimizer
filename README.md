# 🇨🇱 Chilean Mutual Funds — Portfolio Optimizer

> **Interactive Streamlit app for quantitative analysis and optimization of Chilean mutual funds. Combines Modern Portfolio Theory, robust covariance estimation, market regime detection, and Monte Carlo simulation in a single dashboard.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-6.6-3F4F75?logo=plotly)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Funds](https://img.shields.io/badge/Funds-42-orange)]()

---

## Abstract

This project implements a rigorous quantitative framework for optimizing portfolios of Chilean mutual funds. Using 42 funds from Santander Asset Management and LarrainVial (2020–2026), we apply six advanced techniques that go beyond standard Markowitz optimization: **Ledoit-Wolf covariance shrinkage** to address the curse of dimensionality, **robust min-max optimization** for worst-case Sharpe maximization, **K-Means and hierarchical clustering** to reveal behavioral fund groupings, **Hidden Markov Models** for market regime detection, **block bootstrap** as a non-parametric alternative to Monte Carlo, and **walk-forward validation** for honest out-of-sample evaluation.

**Key results:**
- Optimal global portfolio: **Sharpe 0.519** | Return **11.45%** | Volatility **11.28%**
- Ledoit-Wolf reduces condition number from **4×10⁸ → 130** (shrinkage α = 0.17)
- K-Means identifies **8 behavioral clusters** that cut across official fund categories
- Walk-forward OOS Sharpe: **0.57 average** across 4 splits (2022–2026)
- Block bootstrap produces **35% wider tails** than parametric Monte Carlo

---

## Live Demo

```bash
git clone https://github.com/JunTierSS/chilean-mutual-funds-optimizer
cd chilean-mutual-funds-optimizer
pip install -r requirements.txt
python3 -m streamlit run app.py
```

---

## Dashboard — 10 Tabs + 6 Simulators

### Tabs

| Tab | Content |
|-----|---------|
| 📊 **Mercado** | Correlation heatmap, return distribution, seasonality by fund |
| 🔍 **Fondos** | Fund screener: Sharpe, Sortino, VaR, CVaR, drawdown table |
| 🎯 **Portafolio** | Markowitz optimization, efficient frontier, HRP comparison |
| 🌊 **Regímenes** | Hidden Markov Model — bull/bear/crisis regimes with BIC selection |
| 🚀 **Proyección** | Block bootstrap vs Monte Carlo fan charts, multi-profile comparison |
| 🔄 **Backtesting** | Walk-forward validation (4 splits), Sharpe degradation IS→OOS |
| 📉 **Dinámico** | GARCH(1,1) volatility, rolling Sharpe/beta/correlation |
| ⚡ **Stress Test** | 7 hypothetical stress scenarios (COVID, rate shock, CLP crash…) |
| 🔬 **Clustering** | K-Means + hierarchical clustering, dendrogram, cluster profiles |
| 🎮 **Simulador** | 6 interactive simulators (see below) |

### Simulators

| Simulator | Description |
|-----------|-------------|
| 🎯 Meta financiera | Required monthly savings to reach a financial goal |
| ⚖️ Rebalanceo | Buy & hold vs monthly/annual/threshold rebalancing |
| 📅 ¿Qué habría pasado? | Actual historical returns from any start date |
| 💸 Retiro periódico | Capital survival curve with periodic withdrawals |
| 🔀 Cambio de perfil | Two-phase fan chart: stay A, stay B, switch A→B |
| 🏝️ Independencia Financiera | Accumulation phase + retirement phase, back-solve capital needed |

---

## Methodology

### 1. Ledoit-Wolf Covariance Shrinkage

The Ledoit-Wolf (2004) estimator shrinks the sample covariance Σ̂ toward a structured target T:

```
Σ_LW = (1 - α) · Σ̂ + α · T
```

With our dataset, α = 0.169 reduces the condition number from **4.0×10⁸ to 130**, making portfolio weights numerically stable.

### 2. Robust Optimization (Min-Max Sharpe)

```
max_w  min_{Σ ∈ U} Sharpe(w, Σ)
```

Implemented via Σ_worst = Σ_LW + ε·diag(Σ_LW), with ε = 0.10.

### 3. Hierarchical Risk Parity (HRP) — López de Prado (2016)

Three-step algorithm: (1) Mantegna distance clustering, (2) quasi-diagonalization, (3) recursive bisection. No matrix inversion required.

### 4. Hidden Markov Model — Market Regimes

```
P(r_t | s_t = k) ~ N(μ_k, σ_k²)
```

Number of regimes selected by minimizing BIC = −2·log L + k·log(n).

### 5. Block Bootstrap — Kunsch (1989)

```python
indices = block_bootstrap_indices(n_obs, block_size=6, n_periods=60)
R_sim   = R_historical[indices]   # preserves autocorrelation
```

Produces **35% wider confidence intervals** vs parametric Monte Carlo.

### 6. Walk-Forward Validation

```
|── Train 1 (70%) ──|── Test 1 ──|
     |── Train 2 (70%) ──|── Test 2 ──|
          |── Train 3 (70%) ──|── Test 3 ──|
```

4 splits over 2022–2026. Average IS→OOS Sharpe degradation: **+0.36** (no overfitting).

### 7. GARCH(1,1) — Bollerslev (1986)

```
σ_t² = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Implemented via `arch` library with manual MLE fallback when not available.

---

## Results

### Portfolio Performance (Jan 2020 – Mar 2026)

| Portfolio | Sharpe | Return | Volatility | Max DD | Sortino |
|-----------|--------|--------|------------|--------|---------|
| Conservador (LW) | 0.128 | 6.10% | 3.87% | -4.2% | 0.156 |
| Moderado (LW) | 0.245 | 8.28% | 10.91% | -11.3% | 0.271 |
| Agresivo (LW) | 0.353 | 10.05% | 12.61% | -16.8% | 0.362 |
| **Óptimo Global** | **0.519** | **11.45%** | **11.28%** | -14.1% | **0.481** |
| HRP | 0.441 | 9.87% | 10.54% | -13.2% | 0.398 |

### 5-Year Projection of CLP 10M (Bootstrap vs Monte Carlo)

| Method | P5 | Median | P95 |
|--------|----|--------|-----|
| Block Bootstrap | CLP 13.4M | CLP 18.6M | CLP 24.8M |
| Monte Carlo Normal | CLP 11.5M | CLP 17.3M | CLP 26.1M |

---

## Project Structure

```
chilean-mutual-funds-optimizer/
├── app.py                    # Streamlit dashboard (10 tabs, 6 simulators)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml           # Dark theme + server config
├── src/
│   ├── loader.py             # Data loading + USD/CLP normalization
│   ├── metrics.py            # Sharpe, Sortino, VaR, CVaR, Calmar, Omega
│   ├── covariance.py         # Ledoit-Wolf + OAS shrinkage
│   ├── optimizer.py          # Markowitz + Robust + Global + HRP
│   ├── hrp.py                # Hierarchical Risk Parity (López de Prado 2016)
│   ├── clustering.py         # K-Means + Hierarchical clustering
│   ├── regimes.py            # Hidden Markov Model (BIC selection)
│   ├── backtesting.py        # Walk-forward validation
│   ├── bootstrap.py          # Block bootstrap + Monte Carlo simulation
│   ├── garch.py              # GARCH(1,1) conditional volatility
│   ├── rolling.py            # Rolling Sharpe, beta, correlation
│   ├── stress_hipotetico.py  # Hypothetical stress test scenarios
│   └── benchmarks.py         # SP500/IPSA comparison, Jensen's alpha
├── data/
│   ├── santander/            # 20 funds — Santander AM (CSV)
│   ├── larrain_vial/         # 22 funds — LarrainVial AM (CSV)
│   └── raw/
│       └── sp500.csv         # S&P 500 CLP-adjusted returns
├── outputs/                  # Generated charts
└── analisis_completo.ipynb   # Full analysis notebook
```

---

## Data

| Source | Funds | Period | Frequency |
|--------|-------|--------|-----------|
| Santander Asset Management | 20 | Jul 2017 – Mar 2026 | Monthly |
| LarrainVial Asset Management | 22 | Jul 2017 – Mar 2026 | Monthly |
| S&P 500 (CLP-adjusted) | 1 | 2010 – Mar 2026 | Monthly |

**Currency normalization:** 3 LarrainVial funds denominated in USD are converted to CLP using approximate historical Banco Central de Chile exchange rates.

---

## Installation

```bash
# Clone
git clone https://github.com/JunTierSS/chilean-mutual-funds-optimizer
cd chilean-mutual-funds-optimizer

# Install dependencies
pip install -r requirements.txt

# Run app
python3 -m streamlit run app.py
```

**Optional packages for full functionality:**
```bash
pip install hmmlearn arch   # HMM regimes + GARCH (graceful fallback if absent)
```

---

## Author

**YOUR NAME** — Industrial Civil Engineer + MSc Data Science

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/JunTierSS)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/JunTierSS)

---

## References

- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*.
- López de Prado, M. (2016). Building diversified portfolios that outperform out-of-sample. *Journal of Portfolio Management*.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*.
- Kunsch, H.R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*.
- Mantegna, R.N. (1999). Hierarchical structure in financial markets. *European Physical Journal B*.

---

## License

MIT License — free to use, modify and distribute with attribution.

---

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. Past performance does not guarantee future results. This is not financial advice.
