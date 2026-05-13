# Commercial Bank Risk Real-Time Dashboard

A mock real-time risk management dashboard for a commercial bank, built with Python + Streamlit and fully synthetic data. Built for an AI Finance course assignment.

## Features

| Module | Metrics |
|---|---|
| **Market Risk** | Parametric VaR, Historical VaR, CVaR/ES, Rolling 30-day VaR |
| **Credit Risk** | Expected Loss (PD × LGD × EAD), NPL Ratio, Unexpected Loss, Segment breakdown |
| **Liquidity Risk** | LCR, NSFR, 30-day Liquidity Runway, Waterfall chart |
| **Operational Risk** | Loss Distribution (Poisson frequency × LogNormal severity), Event heatmap |
| **Risk Heatmap** | Enterprise risk register, Likelihood × Impact matrix, Radar chart |

## Math

- **VaR**: `VaR_α = −(μ − z_α · σ)` (parametric) or empirical percentile (historical)
- **Expected Loss**: `EL = PD × LGD × EAD`
- **LCR**: `HQLA / Net Cash Outflows (30d) ≥ 100%`
- **NSFR**: `Available Stable Funding / Required Stable Funding ≥ 100%`
- **Op Risk Frequency**: `N ~ Poisson(λ)` | **Severity**: `X ~ LogNormal(μ, σ²)`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Prompt Used (AI-Assisted)

> "You are a senior risk analyst at a commercial bank. Help me build a mock real-time risk dashboard in Python (Streamlit) using synthetic data. The dashboard must cover the four Basel III risk pillars: Credit Risk, Market Risk, Liquidity Risk, and Operational Risk. For each risk, generate realistic synthetic data, compute the standard industry metrics (NPL ratio & Expected Loss for credit; 95%/99% VaR for market; LCR & NSFR for liquidity; loss frequency & severity for operational), show the math behind each calculation, and display color-coded KPI cards with live-updating charts. Simulate real-time data refresh every few seconds. Include a risk heatmap summary. Explain every formula used."

## Framework

Basel III aligned: VaR (FRTB), EL/UL (IRB), LCR/NSFR (LCR standard), Op Risk LDA/BIA.

## Tech Stack

- Python 3.11+
- Streamlit (dashboard framework)
- Plotly (interactive charts)
- NumPy / SciPy (risk math)
- Pandas (data manipulation)
