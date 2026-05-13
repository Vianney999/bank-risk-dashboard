"""
Commercial Bank Risk Real-Time Dashboard
AI Finance Course - Risk Management Assignment

Covers: Credit Risk, Market Risk, Liquidity Risk, Operational Risk
Framework: Basel III aligned metrics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import time
from datetime import datetime, timedelta

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commercial Bank Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px;
    border-left: 4px solid #7c7cff;
    margin-bottom: 8px;
}
.risk-high   { border-left-color: #ff4b4b !important; }
.risk-medium { border-left-color: #ffa500 !important; }
.risk-low    { border-left-color: #21c354 !important; }
.math-box {
    background: #0e1117;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 12px;
    font-family: monospace;
    font-size: 0.85em;
    color: #cdd6f4;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.title("Dashboard Controls")
auto_refresh = st.sidebar.toggle("Live Refresh (5s)", value=True)
confidence_level = st.sidebar.selectbox("VaR Confidence Level", [0.95, 0.99], index=0)
portfolio_size = st.sidebar.slider("Portfolio Size ($M)", 100, 2000, 500, step=50)
show_math = st.sidebar.checkbox("Show Math Explanations", value=True)
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ── Seed for reproducibility with drift ──────────────────────────────────────
seed = int(time.time()) % 10000 if auto_refresh else 42
rng = np.random.default_rng(seed)

# ═════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def gen_credit_data(rng, portfolio_m):
    """Generate synthetic loan portfolio data."""
    n_loans = 1000
    # Loan categories with different PD profiles
    segments = {
        "Corporate":      {"weight": 0.40, "pd_mean": 0.015, "lgd": 0.45, "ead_range": (1, 50)},
        "SME":            {"weight": 0.30, "pd_mean": 0.035, "lgd": 0.55, "ead_range": (0.1, 5)},
        "Retail Mortgage":{"weight": 0.20, "pd_mean": 0.010, "lgd": 0.20, "ead_range": (0.1, 2)},
        "Consumer":       {"weight": 0.10, "pd_mean": 0.060, "lgd": 0.75, "ead_range": (0.01, 0.5)},
    }
    loans = []
    for seg, params in segments.items():
        n = int(n_loans * params["weight"])
        pd_vals  = np.clip(rng.normal(params["pd_mean"], params["pd_mean"]*0.3, n), 0.001, 0.30)
        lgd_vals = np.clip(rng.normal(params["lgd"], 0.05, n), 0.05, 0.95)
        ead_vals = rng.uniform(*params["ead_range"], n)
        el_vals  = pd_vals * lgd_vals * ead_vals
        defaulted = rng.binomial(1, pd_vals)
        loans.append(pd.DataFrame({
            "segment": seg, "PD": pd_vals, "LGD": lgd_vals,
            "EAD": ead_vals, "EL": el_vals, "defaulted": defaulted
        }))
    df = pd.concat(loans, ignore_index=True)
    # Scale to portfolio size
    scale = portfolio_m / df["EAD"].sum()
    df["EAD"] *= scale
    df["EL"]  *= scale
    return df

def gen_market_data(rng, n_days=252):
    """Generate synthetic daily P&L series (in $M)."""
    mu    = 0.0002
    sigma = rng.uniform(0.008, 0.015)
    returns = rng.normal(mu, sigma, n_days)
    # Add occasional fat tails
    shocks = rng.choice([0, 1], n_days, p=[0.97, 0.03])
    returns += shocks * rng.normal(0, sigma * 4, n_days)
    pnl = returns * portfolio_size  # $M
    return pnl, sigma

def gen_liquidity_data(rng):
    """Generate HQLA, outflows, inflows, NSF stable/required funding."""
    hqla         = rng.uniform(120, 180)   # $M  High Quality Liquid Assets
    outflows_30d = rng.uniform(80, 130)    # $M  Expected 30-day outflows
    inflows_30d  = min(rng.uniform(20, 60), 0.75 * outflows_30d)
    net_outflows = outflows_30d - inflows_30d
    lcr          = hqla / net_outflows * 100

    available_sf = rng.uniform(400, 600)   # $M  Available Stable Funding
    required_sf  = rng.uniform(350, 550)   # $M  Required Stable Funding
    nsfr         = available_sf / required_sf * 100

    # 30-day liquidity runway curve
    daily_burn = net_outflows / 30
    buffer = hqla
    runway = []
    for d in range(31):
        runway.append(max(buffer - daily_burn * d, 0))

    return {
        "hqla": hqla, "outflows": outflows_30d, "inflows": inflows_30d,
        "net_outflows": net_outflows, "lcr": lcr,
        "available_sf": available_sf, "required_sf": required_sf,
        "nsfr": nsfr, "runway": runway
    }

def gen_operational_data(rng, n_months=12):
    """Generate synthetic op-risk loss events."""
    event_types = {
        "Fraud / Theft":        {"freq_lambda": 3.5, "sev_mu": 0.8, "sev_sigma": 0.9},
        "IT System Failure":    {"freq_lambda": 2.0, "sev_mu": 1.2, "sev_sigma": 0.7},
        "Process Error":        {"freq_lambda": 5.0, "sev_mu": 0.3, "sev_sigma": 0.8},
        "Legal / Compliance":   {"freq_lambda": 1.0, "sev_mu": 2.5, "sev_sigma": 1.0},
        "External Cyber Attack":{"freq_lambda": 0.8, "sev_mu": 3.0, "sev_sigma": 1.1},
    }
    records = []
    for evt, params in event_types.items():
        for month in range(n_months):
            n_events = rng.poisson(params["freq_lambda"])
            if n_events > 0:
                losses = np.exp(rng.normal(params["sev_mu"], params["sev_sigma"], n_events))
                for loss in losses:
                    records.append({"event_type": evt, "month": month + 1, "loss_$K": loss})
    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════════
#  RISK CALCULATORS
# ═════════════════════════════════════════════════════════════════════════════

def calc_var(pnl_series, confidence):
    """
    Parametric VaR  : VaR = -( mu - z * sigma ) * sqrt(1)
    Historical VaR  : empirical percentile of loss distribution
    CVaR (ES)       : mean of losses beyond VaR threshold
    """
    mu    = np.mean(pnl_series)
    sigma = np.std(pnl_series)
    z     = norm.ppf(confidence)

    var_param = -(mu - z * sigma)
    var_hist  = -np.percentile(pnl_series, (1 - confidence) * 100)
    cvar      = -np.mean(pnl_series[pnl_series <= -var_hist])
    return var_param, var_hist, cvar, mu, sigma

def calc_credit_metrics(df):
    total_ead  = df["EAD"].sum()
    total_el   = df["EL"].sum()
    n_defaults = df["defaulted"].sum()
    npl_ratio  = n_defaults / len(df) * 100
    el_rate    = total_el / total_ead * 100
    # Unexpected Loss (simplified UL = k * sigma of EL)
    ul = 2.33 * np.sqrt((df["PD"] * (1 - df["PD"]) * (df["LGD"] * df["EAD"])**2).sum())
    return total_ead, total_el, npl_ratio, el_rate, ul, n_defaults

def risk_rating(value, thresholds, labels=("LOW","MEDIUM","HIGH")):
    """Return (label, color) based on thresholds [low_max, medium_max]."""
    if value <= thresholds[0]:
        return labels[0], "#21c354"
    elif value <= thresholds[1]:
        return labels[1], "#ffa500"
    else:
        return labels[2], "#ff4b4b"


# ═════════════════════════════════════════════════════════════════════════════
#  GENERATE ALL DATA
# ═════════════════════════════════════════════════════════════════════════════
credit_df    = gen_credit_data(rng, portfolio_size)
pnl_series, sigma_daily = gen_market_data(rng)
liq          = gen_liquidity_data(rng)
op_df        = gen_operational_data(rng)

var_param, var_hist, cvar, pnl_mu, pnl_sigma = calc_var(pnl_series, confidence_level)
total_ead, total_el, npl_ratio, el_rate, ul, n_defaults = calc_credit_metrics(credit_df)


# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("🏦 Commercial Bank — Real-Time Risk Dashboard")
st.caption(f"Portfolio: ${portfolio_size}M  |  Confidence: {confidence_level*100:.0f}%  |  "
           f"Basel III Framework  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
#  TOP KPI ROW
# ═════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)

var_rating, var_color   = risk_rating(var_hist, [portfolio_size*0.02, portfolio_size*0.04])
npl_rating, npl_color   = risk_rating(npl_ratio, [3, 7])
lcr_rating, lcr_color   = risk_rating(liq["lcr"], [100, 120], ("HIGH","MEDIUM","LOW"))
lcr_rating_inv, lcr_color_inv = ("LOW","#21c354") if liq["lcr"] >= 120 else \
                                 ("MEDIUM","#ffa500") if liq["lcr"] >= 100 else ("HIGH","#ff4b4b")

with k1:
    st.metric("Market VaR (1-day)", f"${var_hist:.2f}M",
              delta=f"CVaR: ${cvar:.2f}M", delta_color="inverse")
with k2:
    st.metric("NPL Ratio", f"{npl_ratio:.2f}%",
              delta=f"EL: ${total_el:.1f}M", delta_color="inverse")
with k3:
    st.metric("LCR", f"{liq['lcr']:.1f}%",
              delta=f"Min 100% (Basel III)", delta_color="off")
with k4:
    op_total_loss = op_df["loss_$K"].sum() / 1000
    st.metric("Op. Risk Losses (12M)", f"${op_total_loss:.2f}M",
              delta=f"{len(op_df)} events", delta_color="inverse")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Market Risk", "💳 Credit Risk", "💧 Liquidity Risk",
    "⚙️ Operational Risk", "🗺️ Risk Heatmap"
])

# ─── TAB 1: MARKET RISK ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Market Risk — Value at Risk")

    if show_math:
        st.markdown("""
<div class='math-box'>
<b>Parametric VaR</b> (Variance-Covariance Method)<br>
&nbsp;&nbsp;VaR<sub>α</sub> = −(μ − z<sub>α</sub> · σ) · √h<br>
&nbsp;&nbsp;where z<sub>α</sub> = N⁻¹(α), h = holding period (days)<br><br>
<b>Historical Simulation VaR</b><br>
&nbsp;&nbsp;VaR<sub>α</sub> = −Percentile(P&L, (1−α)·100)<br><br>
<b>Expected Shortfall (CVaR)</b><br>
&nbsp;&nbsp;CVaR<sub>α</sub> = −E[P&L | P&L ≤ −VaR<sub>α</sub>]<br><br>
<b>Current inputs:</b>&nbsp;
μ = {:.4f}M, σ = {:.4f}M, z = {:.3f}, α = {:.0f}%<br>
<b>Parametric VaR</b> = ${:.3f}M&nbsp;&nbsp;
<b>Historical VaR</b> = ${:.3f}M&nbsp;&nbsp;
<b>CVaR</b> = ${:.3f}M
</div>
""".format(pnl_mu, pnl_sigma * portfolio_size, norm.ppf(confidence_level),
           confidence_level * 100, var_param, var_hist, cvar), unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])

    with c1:
        # P&L Histogram with VaR lines
        fig_var = go.Figure()
        fig_var.add_trace(go.Histogram(
            x=pnl_series, nbinsx=60, name="Daily P&L",
            marker_color="#7c7cff", opacity=0.7
        ))
        fig_var.add_vline(x=-var_hist, line_color="#ff4b4b", line_dash="dash",
                          annotation_text=f"Hist VaR {confidence_level*100:.0f}%: −${var_hist:.2f}M",
                          annotation_position="top right")
        fig_var.add_vline(x=-var_param, line_color="#ffa500", line_dash="dot",
                          annotation_text=f"Param VaR: −${var_param:.2f}M")
        fig_var.add_vline(x=-cvar, line_color="#ff0000", line_dash="longdash",
                          annotation_text=f"CVaR: −${cvar:.2f}M")
        fig_var.update_layout(title="P&L Distribution (252 trading days)",
                               xaxis_title="Daily P&L ($M)", yaxis_title="Frequency",
                               template="plotly_dark", height=350)
        st.plotly_chart(fig_var, use_container_width=True)

    with c2:
        # Rolling 30-day VaR
        rolling_var = []
        for i in range(30, len(pnl_series)):
            window = pnl_series[i-30:i]
            rolling_var.append(-np.percentile(window, (1 - confidence_level) * 100))
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            y=rolling_var, mode="lines", name="Rolling 30d VaR",
            line=dict(color="#7c7cff", width=2)
        ))
        fig_roll.add_hline(y=var_hist, line_color="#ff4b4b", line_dash="dash")
        fig_roll.update_layout(title="Rolling 30-Day VaR ($M)", template="plotly_dark",
                                yaxis_title="VaR ($M)", height=350)
        st.plotly_chart(fig_roll, use_container_width=True)

    # VaR Summary Table
    df_var_summary = pd.DataFrame({
        "Metric": ["Parametric VaR", "Historical VaR", "CVaR / ES",
                   "Daily σ ($M)", "10-Day VaR (√10 rule)"],
        "Value": [f"${var_param:.3f}M", f"${var_hist:.3f}M", f"${cvar:.3f}M",
                  f"${pnl_sigma*portfolio_size:.3f}M",
                  f"${var_hist * np.sqrt(10):.3f}M"],
        "Notes": [
            "Normal distribution assumption",
            "Empirical percentile of P&L history",
            "Average loss beyond VaR threshold",
            "Annualized: {:.1f}%".format(pnl_sigma * np.sqrt(252) * 100),
            "Basel III standard scaling"
        ]
    })
    st.dataframe(df_var_summary, use_container_width=True, hide_index=True)


# ─── TAB 2: CREDIT RISK ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Credit Risk — Loan Portfolio Analysis")

    if show_math:
        st.markdown("""
<div class='math-box'>
<b>Expected Loss (EL)</b><br>
&nbsp;&nbsp;EL = PD × LGD × EAD<br>
&nbsp;&nbsp;PD = Probability of Default &nbsp;|&nbsp; LGD = Loss Given Default &nbsp;|&nbsp; EAD = Exposure at Default<br><br>
<b>Unexpected Loss (UL)</b> — simplified single-asset approximation:<br>
&nbsp;&nbsp;UL = z<sub>0.99</sub> · √[ Σᵢ PD<sub>i</sub>(1−PD<sub>i</sub>) · (LGD<sub>i</sub>·EAD<sub>i</sub>)² ]<br><br>
<b>NPL Ratio</b> = Number of Defaulted Loans / Total Loans × 100%<br><br>
<b>Portfolio:</b> EAD=${:.1f}M  EL=${:.2f}M ({:.2f}%)  NPL={:.2f}%  UL=${:.2f}M
</div>
""".format(total_ead, total_el, el_rate, npl_ratio, ul), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Exposure (EAD)", f"${total_ead:.1f}M")
    with c2:
        st.metric("Expected Loss (EL)", f"${total_el:.2f}M", delta=f"{el_rate:.2f}% of EAD",
                  delta_color="inverse")
    with c3:
        st.metric("Unexpected Loss (UL)", f"${ul:.2f}M", delta="99% conf.",
                  delta_color="inverse")

    c1, c2 = st.columns([3, 2])

    with c1:
        # EL by segment
        seg_summary = credit_df.groupby("segment").agg(
            EAD=("EAD","sum"), EL=("EL","sum"),
            Avg_PD=("PD","mean"), Defaults=("defaulted","sum"),
            Count=("PD","count")
        ).reset_index()
        seg_summary["EL_Rate"] = seg_summary["EL"] / seg_summary["EAD"] * 100
        seg_summary["NPL_%"]   = seg_summary["Defaults"] / seg_summary["Count"] * 100

        fig_seg = px.bar(seg_summary, x="segment", y=["EAD","EL"],
                         barmode="group", title="EAD vs Expected Loss by Segment ($M)",
                         template="plotly_dark",
                         color_discrete_map={"EAD":"#7c7cff","EL":"#ff4b4b"})
        st.plotly_chart(fig_seg, use_container_width=True)

    with c2:
        fig_pie = px.pie(seg_summary, names="segment", values="EAD",
                         title="Portfolio Concentration by EAD",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig_pie, use_container_width=True)

    # PD distribution
    fig_pd = px.histogram(credit_df, x="PD", color="segment", nbins=50,
                          title="PD Distribution Across Loan Segments",
                          template="plotly_dark", opacity=0.75,
                          labels={"PD": "Probability of Default"})
    st.plotly_chart(fig_pd, use_container_width=True)

    # Segment detail table
    seg_summary_display = seg_summary.copy()
    seg_summary_display["EAD"]     = seg_summary_display["EAD"].map("${:.1f}M".format)
    seg_summary_display["EL"]      = seg_summary_display["EL"].map("${:.2f}M".format)
    seg_summary_display["Avg_PD"]  = seg_summary_display["Avg_PD"].map("{:.2%}".format)
    seg_summary_display["EL_Rate"] = seg_summary_display["EL_Rate"].map("{:.2f}%".format)
    seg_summary_display["NPL_%"]   = seg_summary_display["NPL_%"].map("{:.2f}%".format)
    st.dataframe(seg_summary_display[["segment","EAD","EL","Avg_PD","EL_Rate","NPL_%","Defaults"]],
                 use_container_width=True, hide_index=True)


# ─── TAB 3: LIQUIDITY RISK ───────────────────────────────────────────────────
with tab3:
    st.subheader("Liquidity Risk — LCR & NSFR")

    if show_math:
        st.markdown("""
<div class='math-box'>
<b>Liquidity Coverage Ratio (LCR)</b> — Basel III §30-day stress horizon:<br>
&nbsp;&nbsp;LCR = HQLA / Net Cash Outflows (30 days) ≥ 100%<br>
&nbsp;&nbsp;HQLA = High-Quality Liquid Assets (Level 1 + Level 2)<br>
&nbsp;&nbsp;Net Outflows = Total Outflows − min(Inflows, 0.75 × Outflows)<br><br>
<b>Net Stable Funding Ratio (NSFR)</b> — Basel III structural liquidity:<br>
&nbsp;&nbsp;NSFR = Available Stable Funding / Required Stable Funding ≥ 100%<br><br>
<b>Current:</b>&nbsp;
HQLA=${:.1f}M  Net Outflows=${:.1f}M  LCR={:.1f}%  |
ASF=${:.1f}M  RSF=${:.1f}M  NSFR={:.1f}%
</div>
""".format(liq["hqla"], liq["net_outflows"], liq["lcr"],
           liq["available_sf"], liq["required_sf"], liq["nsfr"]), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    lcr_delta = "PASS ✓" if liq["lcr"] >= 100 else "BREACH ✗"
    nsfr_delta = "PASS ✓" if liq["nsfr"] >= 100 else "BREACH ✗"
    with c1: st.metric("HQLA", f"${liq['hqla']:.1f}M")
    with c2: st.metric("Net 30d Outflows", f"${liq['net_outflows']:.1f}M")
    with c3: st.metric("LCR", f"{liq['lcr']:.1f}%", delta=lcr_delta,
                        delta_color="normal" if liq["lcr"] >= 100 else "inverse")
    with c4: st.metric("NSFR", f"{liq['nsfr']:.1f}%", delta=nsfr_delta,
                        delta_color="normal" if liq["nsfr"] >= 100 else "inverse")

    c1, c2 = st.columns(2)

    with c1:
        # LCR Gauge
        fig_lcr = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=liq["lcr"],
            delta={"reference": 100, "increasing": {"color": "#21c354"},
                   "decreasing": {"color": "#ff4b4b"}},
            gauge={
                "axis": {"range": [0, 250]},
                "bar":  {"color": "#7c7cff"},
                "steps": [
                    {"range": [0, 100],   "color": "#ff4b4b"},
                    {"range": [100, 120], "color": "#ffa500"},
                    {"range": [120, 250], "color": "#21c354"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 100}
            },
            title={"text": "LCR (%) — Min 100%"}
        ))
        fig_lcr.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_lcr, use_container_width=True)

    with c2:
        # NSFR Gauge
        fig_nsfr = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=liq["nsfr"],
            delta={"reference": 100},
            gauge={
                "axis": {"range": [0, 200]},
                "bar":  {"color": "#7c7cff"},
                "steps": [
                    {"range": [0, 100],   "color": "#ff4b4b"},
                    {"range": [100, 130], "color": "#ffa500"},
                    {"range": [130, 200], "color": "#21c354"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 100}
            },
            title={"text": "NSFR (%) — Min 100%"}
        ))
        fig_nsfr.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_nsfr, use_container_width=True)

    # Liquidity runway
    fig_runway = go.Figure()
    fig_runway.add_trace(go.Scatter(
        x=list(range(31)), y=liq["runway"], fill="tozeroy",
        name="Liquidity Buffer", line=dict(color="#7c7cff", width=2)
    ))
    fig_runway.add_hline(y=0, line_color="#ff4b4b", line_dash="dash",
                          annotation_text="Buffer Exhausted")
    fig_runway.update_layout(title="30-Day Liquidity Runway ($M)",
                              xaxis_title="Days", yaxis_title="Buffer ($M)",
                              template="plotly_dark", height=280)
    st.plotly_chart(fig_runway, use_container_width=True)

    # Waterfall: outflows breakdown
    fig_wf = go.Figure(go.Waterfall(
        name="Liquidity",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["HQLA", "Gross Outflows", "Inflows", "Net Buffer"],
        y=[liq["hqla"], -liq["outflows"], liq["inflows"],
           liq["hqla"] - liq["net_outflows"]],
        connector={"line": {"color": "#555"}},
        decreasing={"marker": {"color": "#ff4b4b"}},
        increasing={"marker": {"color": "#21c354"}},
        totals={"marker": {"color": "#7c7cff"}},
    ))
    fig_wf.update_layout(title="Liquidity Waterfall — 30-Day Stress ($M)",
                          template="plotly_dark", height=320)
    st.plotly_chart(fig_wf, use_container_width=True)


# ─── TAB 4: OPERATIONAL RISK ─────────────────────────────────────────────────
with tab4:
    st.subheader("Operational Risk — Loss Event Analysis")

    if show_math:
        st.markdown("""
<div class='math-box'>
<b>Operational Risk — Loss Distribution Approach (LDA)</b><br>
&nbsp;&nbsp;Frequency: N ~ Poisson(λ)  →  P(N=k) = e⁻λ λᵏ / k!<br>
&nbsp;&nbsp;Severity:  X ~ LogNormal(μ, σ²) →  E[X] = e^(μ + σ²/2)<br>
&nbsp;&nbsp;Annual Loss = Σᵢ Xᵢ  (compound distribution)<br><br>
<b>Basic Indicator Approach (BIA) Capital Charge:</b><br>
&nbsp;&nbsp;K_BIA = α × GNI_avg  where α = 15%, GNI = Gross Net Income<br><br>
<b>Advanced Measurement Approach (AMA):</b><br>
&nbsp;&nbsp;Capital = VaR₉₉.₉%(Annual Loss Distribution)<br>
</div>
""", unsafe_allow_html=True)

    monthly = op_df.groupby(["month","event_type"])["loss_$K"].sum().reset_index()
    total_by_type = op_df.groupby("event_type")["loss_$K"].agg(["sum","count","mean"]).reset_index()
    total_by_type.columns = ["Event Type", "Total Loss $K", "Count", "Avg Loss $K"]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Op Loss (12M)", f"${op_df['loss_$K'].sum()/1000:.2f}M")
    with c2: st.metric("Total Events", f"{len(op_df)}")
    with c3: st.metric("Worst Event", f"${op_df['loss_$K'].max():.1f}K")

    c1, c2 = st.columns(2)

    with c1:
        fig_op_bar = px.bar(total_by_type, x="Event Type", y="Total Loss $K",
                             color="Event Type", title="Total Losses by Event Type ($K)",
                             template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Bold)
        fig_op_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_op_bar, use_container_width=True)

    with c2:
        fig_op_scatter = px.scatter(op_df, x="month", y="loss_$K",
                                     color="event_type", size="loss_$K",
                                     title="Loss Events over Time",
                                     template="plotly_dark",
                                     labels={"loss_$K":"Loss ($K)","month":"Month"},
                                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig_op_scatter.update_layout(height=350)
        st.plotly_chart(fig_op_scatter, use_container_width=True)

    # Monthly heatmap
    pivot = monthly.pivot_table(index="event_type", columns="month",
                                  values="loss_$K", fill_value=0)
    fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                          title="Monthly Loss Heatmap by Event Type ($K)",
                          labels={"x":"Month","y":"Event Type","color":"Loss $K"},
                          template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Severity histogram
    fig_sev = px.histogram(op_df, x="loss_$K", color="event_type", nbins=50,
                            log_y=True, title="Loss Severity Distribution (Log Scale)",
                            template="plotly_dark",
                            labels={"loss_$K":"Loss ($K)"},
                            color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_sev, use_container_width=True)

    total_by_type["Total Loss $K"] = total_by_type["Total Loss $K"].map("{:.1f}".format)
    total_by_type["Avg Loss $K"]   = total_by_type["Avg Loss $K"].map("{:.1f}".format)
    st.dataframe(total_by_type, use_container_width=True, hide_index=True)


# ─── TAB 5: RISK HEATMAP ─────────────────────────────────────────────────────
with tab5:
    st.subheader("Enterprise Risk Heatmap — Summary View")

    # Build risk score matrix (likelihood x impact)
    risks = {
        "Credit Default Contagion":     {"likelihood": 3, "impact": 5, "cat": "Credit"},
        "Interest Rate Shock":          {"likelihood": 4, "impact": 4, "cat": "Market"},
        "Equity Market Crash":          {"likelihood": 2, "impact": 5, "cat": "Market"},
        "Liquidity Stress (30d)":       {"likelihood": 2, "impact": 5, "cat": "Liquidity"},
        "Funding Cliff":                {"likelihood": 2, "impact": 4, "cat": "Liquidity"},
        "Cyber Attack":                 {"likelihood": 3, "impact": 4, "cat": "Operational"},
        "Rogue Trader":                 {"likelihood": 1, "impact": 5, "cat": "Operational"},
        "Regulatory Breach":            {"likelihood": 2, "impact": 3, "cat": "Operational"},
        "SME Loan Defaults Spike":      {"likelihood": 4, "impact": 3, "cat": "Credit"},
        "FX Volatility":                {"likelihood": 4, "impact": 3, "cat": "Market"},
    }
    df_risk = pd.DataFrame(risks).T.reset_index()
    df_risk.columns = ["Risk", "Likelihood", "Impact", "Category"]
    df_risk["Likelihood"] = df_risk["Likelihood"].astype(int)
    df_risk["Impact"]     = df_risk["Impact"].astype(int)
    df_risk["Score"]      = df_risk["Likelihood"] * df_risk["Impact"]
    df_risk["Color"] = df_risk["Score"].apply(
        lambda s: "#ff4b4b" if s >= 12 else ("#ffa500" if s >= 6 else "#21c354")
    )
    df_risk["Rating"] = df_risk["Score"].apply(
        lambda s: "Critical" if s >= 12 else ("High" if s >= 8 else ("Medium" if s >= 4 else "Low"))
    )

    fig_hm = px.scatter(df_risk, x="Likelihood", y="Impact",
                         size="Score", color="Category",
                         text="Risk", title="Risk Heatmap (Likelihood × Impact)",
                         template="plotly_dark",
                         range_x=[0, 6], range_y=[0, 6],
                         size_max=40,
                         color_discrete_sequence=px.colors.qualitative.Bold)
    # Add quadrant backgrounds
    for (x0, x1, y0, y1, col) in [
        (0,2,0,3,"#21c35420"), (2,4,0,3,"#ffa50020"),
        (0,2,3,6,"#ffa50020"), (2,4,3,6,"#ff4b4b20"), (4,6,0,6,"#ff4b4b30")
    ]:
        fig_hm.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                          fillcolor=col, line_width=0, layer="below")
    fig_hm.update_traces(textposition="top center")
    fig_hm.update_layout(height=500, xaxis_title="Likelihood (1=Rare, 5=Almost Certain)",
                          yaxis_title="Impact (1=Negligible, 5=Catastrophic)")
    st.plotly_chart(fig_hm, use_container_width=True)

    # Risk register table
    df_display = df_risk[["Risk","Category","Likelihood","Impact","Score","Rating"]].sort_values(
        "Score", ascending=False)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Radar chart — risk scores by category
    cat_scores = df_risk.groupby("Category")["Score"].max().reset_index()
    fig_radar = go.Figure(go.Scatterpolar(
        r=cat_scores["Score"].tolist() + [cat_scores["Score"].iloc[0]],
        theta=cat_scores["Category"].tolist() + [cat_scores["Category"].iloc[0]],
        fill="toself", fillcolor="rgba(124,124,255,0.2)",
        line=dict(color="#7c7cff", width=2), name="Max Risk Score"
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,25])),
                             title="Risk Radar — Max Score per Category",
                             template="plotly_dark", height=400)
    st.plotly_chart(fig_radar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH
# ═════════════════════════════════════════════════════════════════════════════
if auto_refresh:
    time.sleep(5)
    st.rerun()
