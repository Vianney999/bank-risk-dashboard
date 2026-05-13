# SUMMARY MEMO
## AI-Built Risk Dashboard — Comparison to Lecture 5 Framework

**TO:** Professor / AI Finance Class
**FROM:** Nengoueye Takam Brondol Vianne
**DATE:** 13 May 2026
**RE:** Commercial Bank Risk Dashboard — Build, Findings & Gap Analysis vs. Lecture 5

---

## 1. What I Built

Using Claude (Anthropic) as my AI co-pilot, I designed and built a **mock real-time risk dashboard** for a commercial bank in Python (Streamlit). The dashboard covers the four Basel III risk pillars across five interactive tabs, with synthetic data refreshing every five seconds.

| Module | Key Metrics Implemented |
|---|---|
| **Market Risk** | Parametric VaR, Historical VaR, CVaR/ES (95% & 99%), Rolling 30-day VaR |
| **Credit Risk** | Expected Loss (PD × LGD × EAD), NPL Ratio, Unexpected Loss, 4-segment loan portfolio |
| **Liquidity Risk** | LCR (30-day), NSFR, Liquidity Runway, Waterfall chart |
| **Operational Risk** | Poisson frequency + LogNormal severity (LDA), monthly loss heatmap |
| **Risk Heatmap** | Likelihood × Impact register, radar chart across all risk categories |

The AI process was iterative: I first asked Claude to help craft the prompt by answering questions about my bank type, course context, risk focus, and tech stack. Claude then generated all the code, explained every formula, and helped push the project to GitHub.

**GitHub:** https://github.com/Vianney999/bank-risk-dashboard

---

## 2. The Math — What the Dashboard Shows

**Expected Loss:**
> EL = PD × LGD × EAD

Where PD = Probability of Default, LGD = Loss Given Default, EAD = Exposure at Default. The dashboard models four loan segments (Corporate, SME, Retail Mortgage, Consumer) each with distinct PD distributions.

**Value at Risk (Parametric):**
> VaR_α = −(μ − z_α · σ)  where z_α = N⁻¹(α)

Both parametric (assumes normality) and historical (empirical percentile) VaR are computed. CVaR/Expected Shortfall captures the average loss in the tail beyond VaR.

**Liquidity Coverage Ratio:**
> LCR = HQLA / Net Cash Outflows (30 days) ≥ 100%

Net outflows = Gross outflows − min(inflows, 75% × gross outflows), per Basel III.

---

## 3. Comparison to Lecture 5 — Where My Dashboard Falls Short

Lecture 5 ("Closing the Intraday Blind Spot") made a precise and important argument: **a daily VaR dashboard that reports 30-day LCR and end-of-day positions has a structural blind spot — intraday liquidity.** This is exactly the gap present in what I built.

| Dimension | My Dashboard | Lecture 5 Framework |
|---|---|---|
| **Liquidity horizon** | 30-day LCR + NSFR (regulatory, end-of-day) | iLCR: live intraday ratio updated continuously |
| **VaR + liquidity link** | Separate tabs, no co-movement signal | Unified view: market stress and intraday compression shown together |
| **Data frequency** | Synthetic, refreshes every 5s (simulated) | RTGS feed, BEAC collateral data — real institutional feeds |
| **Governance layer** | Not modeled | 1st/2nd/3rd line owners, breach protocols (amber/red), ALCO triggers |
| **Geographic/currency risk** | Single generic bank | Multi-jurisdiction CFA franc (Cameroon, Gabon, Tchad), BEAC-specific |
| **Buffer optimization** | Static synthetic HQLA | Dynamic BEAC collateral pool sizing; 15-25% optimization target |

The Lecture 5 briefing identified three cost channels from not having iLCR:
1. **Idle precautionary buffers** — over-collateralisation at BEAC costs XAF 280–525M/year
2. **Emergency intraday funding** — 30–80 bps premium in shallow CFA franc markets
3. **Regulatory exposure** — non-alignment with BCBS (2013) intraday monitoring tools

My dashboard captures none of these. The 30-day LCR I implemented satisfies the regulatory minimum but offers no signal during the trading day — exactly the scenario Lecture 5 describes as the "single category of risk most likely to translate a market stress event into a same-day liquidity event."

---

## 4. What I Would Add Next (Improvements)

Based on Lecture 5, three enhancements would close the gap:

**1. Intraday Liquidity Coverage Ratio (iLCR) tab**
> iLCR = Available Intraday Liquidity / Peak Projected Intraday Obligation

Display as a live gauge with an amber threshold (e.g., 110%) and a red breach alert — mirroring the tiered breach protocol from Lecture 5.

**2. Joint VaR + iLCR panel**
A single chart showing VaR (market stress) alongside intraday liquidity headroom on the same time axis — surfacing the co-movement that currently exists in silos.

**3. Governance overlay**
A status panel showing which governance layer is active (1st line: Treasury computing iLCR; 2nd line: Market Risk validating; breach status) — making the dashboard a management tool, not just a monitoring screen.

---

## 5. Reflections on the AI-Assisted Process

Using Claude to build this dashboard was faster than writing from scratch, but the AI required clear, specific prompting. The key insight from the prompting exercise: **the quality of the output matched the specificity of the input.** Vague prompts ("build a risk dashboard") produced generic outputs. Structured prompts specifying risk pillars, formulas, and display requirements produced production-quality code.

What AI did well: code generation, formula explanation, chart design, and iteration speed.
What required human judgment: identifying which metrics matter for this specific bank type, recognizing the gap that Lecture 5 highlighted, and framing the governance context.

The Lecture 5 gap — iLCR — is precisely the kind of nuance that comes from institutional knowledge, not from asking an AI to "build a bank dashboard." The AI builds what you ask for; it takes a risk professional to know what to ask for.

---

*Dashboard source code and full prompt documentation: https://github.com/Vianney999/bank-risk-dashboard*
