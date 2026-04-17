import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (FEATURES, FEATURE_LABELS, INDUSTRY_PM, RED_FLAGS,
                   compute_features_all_years, compute_beneish)

st.set_page_config(page_title="Risk Interpretation", page_icon="📋", layout="wide")
st.title("📋 Risk Interpretation")
st.markdown("Plain-language explanation of what the fraud risk results mean for this company.")

@st.cache_resource
def load_models():
    rf = joblib.load("fraud_model.pkl")
    try:    lr = joblib.load("logistic_model.pkl")
    except: lr = None
    return rf, lr

try:
    rf_model, lr_model = load_models()
except Exception:
    st.error("⚠️ `fraud_model.pkl` not found."); st.stop()

SECTOR_MAP = {
    "Technology": "Technology", "Healthcare": "Healthcare",
    "Financial Services": "Finance", "Consumer Defensive": "Consumer Goods",
    "Consumer Cyclical": "Consumer Goods", "Energy": "Energy",
    "Industrials": "Industrials", "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}

FRAUD_BENCHMARKS = {
    "Enron (2001)":     0.91,
    "WorldCom (2002)":  0.87,
    "Wirecard (2020)":  0.83,
    "Healthy Avg.":     0.12,
}

FLAG_EXPLANATIONS = {
    'accrual_ratio': (
        "📌 High Accrual Ratio",
        "The company's reported profit is significantly higher than its actual cash received. "
        "This gap — known as accruals — is one of the most reliable warning signs of earnings manipulation. "
        "Companies sometimes inflate profits on paper without the cash to back it up."
    ),
    'cfo_to_income': (
        "📌 Low Cash Flow vs. Reported Income",
        "For every dollar of profit the company reports, it is generating very little actual cash. "
        "Healthy companies typically collect most of their reported income as real cash. "
        "A large gap here often precedes financial restatements."
    ),
    'receivable_ratio': (
        "📌 High Accounts Receivable",
        "The company has a large and growing amount of money owed to it by customers, "
        "relative to its revenue. This could mean it is booking sales before customers have actually paid — "
        "a common technique used to inflate revenue figures."
    ),
    'revenue_growth': (
        "📌 Unusually High Revenue Growth",
        "The company's revenue has grown at an unusually fast pace. "
        "While growth is generally positive, sudden and extreme revenue spikes "
        "have historically been associated with aggressive or fraudulent revenue recognition."
    ),
    'asset_growth': (
        "📌 Rapid Asset Growth",
        "The company's total assets have grown much faster than expected. "
        "Unexplained asset inflation can signal that the company is capitalizing expenses "
        "it should be writing off, making its balance sheet appear stronger than it is."
    ),
    'debt_ratio': (
        "📌 High Debt Burden",
        "The company carries a heavy debt load relative to its total assets. "
        "High leverage creates financial pressure on management, "
        "which increases the incentive to manipulate earnings to meet targets or debt covenants."
    ),
    'profit_margin_vs_industry': (
        "📌 Margin Significantly Above Industry Peers",
        "The company reports profit margins notably higher than its industry peers. "
        "While this can reflect genuine competitive advantage, it can also indicate "
        "that costs are being understated or revenues overstated."
    ),
    'sga_ratio': (
        "📌 High SG&A Expense Ratio",
        "Selling, general and administrative expenses are consuming a large share of revenue. "
        "A rising SG&A ratio can signal operational stress and may create pressure "
        "to manipulate other parts of the financial statements."
    ),
    'ocf_ratio': (
        "📌 Low Operating Cash Flow",
        "The company is generating very little operating cash flow relative to its asset base. "
        "This suggests the business may not be as profitable in cash terms as the income statement implies."
    ),
}

RISK_MESSAGES = {
    "LOW": (
        "This company shows **low signs of earnings manipulation**.",
        "Its financial ratios are broadly consistent with normal reporting behavior. "
        "The model does not detect significant anomalies in profitability, cash flow quality, "
        "or growth patterns. While no model can guarantee accuracy, "
        "this company does not exhibit the typical warning signs seen in historical fraud cases."
    ),
    "MODERATE": (
        "This company shows **some indicators that warrant attention**.",
        "A few financial ratios deviate from typical patterns. This does not mean fraud is occurring — "
        "but it suggests investors and analysts should look more closely at the quality of earnings, "
        "cash flow conversion, and revenue recognition practices before making decisions."
    ),
    "HIGH": (
        "This company shows **multiple red flags associated with earnings manipulation**.",
        "Several key financial indicators — including cash flow quality, accruals, and growth patterns — "
        "deviate significantly from normal ranges. Historically, companies with similar profiles "
        "have had a higher likelihood of financial restatements or regulatory scrutiny."
    ),
    "VERY HIGH": (
        "This company shows **strong signals consistent with historical fraud cases**.",
        "The combination of indicators detected closely resembles companies that were later found "
        "to have engaged in material earnings manipulation. Extreme caution is advised."
    ),
}

def get_risk_level(score):
    if score < 0.30:   return "LOW",       "🟢", "#2ecc71", "#d5f5e3"
    elif score < 0.50: return "MODERATE",  "🟡", "#f1c40f", "#fef9e7"
    elif score < 0.70: return "HIGH",      "🟠", "#e67e22", "#fdebd0"
    else:              return "VERY HIGH", "🔴", "#e74c3c", "#fadbd8"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    run_btn = st.button("🔍 Interpret Results", use_container_width=True)
    st.divider()
    st.caption("Try a known fraud case like ENRN (Enron) to see high-risk output.")

if not run_btn:
    st.info("👈 Enter a ticker and click **Interpret Results**."); st.stop()

# ── Auto-detect industry ──────────────────────────────────────────────────────
with st.spinner(f"Analyzing **{ticker_input}**…"):
    try:
        info         = yf.Ticker(ticker_input).info
        sector       = info.get("sector", "Other")
        industry_sel = SECTOR_MAP.get(sector, "Other")
        industry_pm  = INDUSTRY_PM[industry_sel]
        feat_all     = compute_features_all_years(ticker_input, industry_pm)
        m_score, _   = compute_beneish(ticker_input)
    except Exception as e:
        st.error(str(e)); st.stop()

st.caption(f"🏭 Auto-detected industry: **{industry_sel}** (sector: {sector})")

feat_latest  = feat_all.iloc[-1][FEATURES].fillna(0)
feat_df      = pd.DataFrame([feat_latest])
rf_prob      = rf_model.predict_proba(feat_df)[0][1]
lr_prob      = lr_model.predict_proba(feat_df)[0][1] if lr_model else None
beneish_flag = int(m_score > -1.78)
scores       = [rf_prob, beneish_flag] + ([lr_prob] if lr_prob else [])
ensemble     = float(np.mean(scores))

risk_label, risk_icon, risk_color, risk_bg = get_risk_level(ensemble)

# ── Risk verdict card ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{risk_bg}; border-left:6px solid {risk_color};
            padding:20px 24px; border-radius:8px; margin-bottom:8px;">
    <h2 style="margin:0; color:{risk_color};">
        {risk_icon} {ticker_input} — {risk_label} RISK &nbsp;
        <span style="font-size:1rem; color:#555;">(Score: {ensemble:.1%})</span>
    </h2>
    <p style="margin:10px 0 4px 0; font-size:1.05rem;">
        <strong>{RISK_MESSAGES[risk_label][0]}</strong>
    </p>
    <p style="margin:0; color:#444;">{RISK_MESSAGES[risk_label][1]}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Score breakdown ───────────────────────────────────────────────────────────
st.subheader("📊 How Was This Score Calculated?")
c1, c2, c3 = st.columns(3)

with c1:
    color = "#e74c3c" if rf_prob>0.5 else "#2ecc71"
    st.markdown(f"""
<div style="background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;
            border-top:4px solid {color};">
    <div style="font-size:2rem;font-weight:bold;color:{color};">{rf_prob:.1%}</div>
    <div style="font-size:0.95rem;margin-top:4px;"><strong>Random Forest Model</strong></div>
    <div style="font-size:0.8rem;color:#666;margin-top:4px;">
        Trained on 80,000+ firm-years of WRDS data.<br>
        Analyzes patterns across all 15 financial ratios.
    </div>
</div>""", unsafe_allow_html=True)

with c2:
    b_color = "#e74c3c" if beneish_flag else "#2ecc71"
    b_text  = "Manipulation Risk" if beneish_flag else "No Manipulation Signal"
    st.markdown(f"""
<div style="background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;
            border-top:4px solid {b_color};">
    <div style="font-size:1.3rem;font-weight:bold;color:{b_color};margin-top:8px;">
        {m_score:.2f}
    </div>
    <div style="font-size:0.85rem;color:#888;">threshold: −1.78</div>
    <div style="font-size:0.95rem;margin-top:4px;"><strong>Beneish M-Score</strong></div>
    <div style="font-size:0.8rem;color:#666;margin-top:4px;">
        Classic accounting model using 8 financial indices.<br>
        Result: <strong>{b_text}</strong>
    </div>
</div>""", unsafe_allow_html=True)

with c3:
    if lr_prob is not None:
        lr_color = "#e74c3c" if lr_prob>0.5 else "#2ecc71"
        st.markdown(f"""
<div style="background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;
            border-top:4px solid {lr_color};">
    <div style="font-size:2rem;font-weight:bold;color:{lr_color};">{lr_prob:.1%}</div>
    <div style="font-size:0.95rem;margin-top:4px;"><strong>Logistic Regression</strong></div>
    <div style="font-size:0.8rem;color:#666;margin-top:4px;">
        Each ratio contributes a weighted score.<br>
        Interpretable statistical baseline.
    </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;
            border-top:4px solid #bdc3c7;">
    <div style="font-size:2rem;font-weight:bold;color:#bdc3c7;">N/A</div>
    <div style="font-size:0.95rem;margin-top:4px;"><strong>Logistic Regression</strong></div>
    <div style="font-size:0.8rem;color:#aaa;margin-top:4px;">logistic_model.pkl not loaded.</div>
</div>""", unsafe_allow_html=True)

st.divider()

# ── Gauge + Benchmark ─────────────────────────────────────────────────────────
col_g, col_b = st.columns(2)
with col_g:
    st.subheader("🎯 Risk Score Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=ensemble*100,
        title={"text":"Ensemble Risk Score (%)"},
        gauge={"axis":{"range":[0,100]},
               "bar":{"color":risk_color},
               "steps":[{"range":[0,30],"color":"#d5f5e3"},
                        {"range":[30,50],"color":"#fef9e7"},
                        {"range":[50,70],"color":"#fdebd0"},
                        {"range":[70,100],"color":"#fadbd8"}],
               "threshold":{"line":{"color":"red","width":3},"value":50}}))
    fig_gauge.update_layout(height=300, margin=dict(t=30,b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_b:
    st.subheader("📌 How Does This Compare?")
    st.markdown("Your company vs. historical fraud cases and healthy companies:")
    bench_data = dict(FRAUD_BENCHMARKS)
    bench_data[f"{ticker_input} (You)"] = ensemble
    bench_df = pd.DataFrame({
        "Company":    list(bench_data.keys()),
        "Risk Score": list(bench_data.values()),
    }).sort_values("Risk Score", ascending=True)

    bcolors = [risk_color if "You" in n
               else "#2ecc71" if "Healthy" in n
               else "#c0392b" for n in bench_df["Company"]]
    fig_bench = go.Figure(go.Bar(
        x=bench_df["Risk Score"]*100, y=bench_df["Company"],
        orientation='h', marker_color=bcolors,
        text=[f"{v:.1%}" for v in bench_df["Risk Score"]],
        textposition="outside",
    ))
    fig_bench.update_layout(height=300,
                             xaxis=dict(title="Risk Score (%)", range=[0,110]),
                             margin=dict(t=10,b=10,l=10,r=60))
    st.plotly_chart(fig_bench, use_container_width=True)

st.divider()

# ── Red flag explanations ─────────────────────────────────────────────────────
triggered = [f for f in FEATURES
             if f in RED_FLAGS
             and not pd.isna(feat_latest[f])
             and RED_FLAGS[f](float(feat_latest[f]))]

st.subheader(f"🚩 Why This Score? — {len(triggered)} Red Flag(s) Detected")

if not triggered:
    st.success("✅ No red flags triggered. All financial ratios are within normal ranges.")
else:
    for feat in triggered:
        val = float(feat_latest[feat])
        title, explanation = FLAG_EXPLANATIONS.get(
            feat, (f"📌 Abnormal {FEATURE_LABELS[feat]}", f"Value {val:.4f} is outside normal range."))
        with st.expander(f"{title}  —  value: **{val:.4f}**", expanded=True):
            st.markdown(explanation)

    clean = [f for f in FEATURES if f not in triggered]
    if clean:
        with st.expander(f"✅ {len(clean)} indicators within normal range"):
            st.dataframe(pd.DataFrame({
                "Indicator": [FEATURE_LABELS[f] for f in clean],
                "Value":     [round(float(feat_latest[f]),4) for f in clean],
                "Status":    ["✅ Normal"] * len(clean),
            }), use_container_width=True, hide_index=True)

st.divider()

# ── What to do next ───────────────────────────────────────────────────────────
st.subheader("💡 What Should I Do Next?")
next_steps = {
    "LOW": [
        "✅ No immediate action required based on financial indicators.",
        "📄 Review the company's annual report (10-K) for qualitative risk factors.",
        "🔁 Re-run this analysis after each quarterly earnings release.",
    ],
    "MODERATE": [
        "🔎 Review the flagged indicators in detail on the **Analysis** page.",
        "📈 Check the **Trends** page to see if metrics have been worsening over time.",
        "📄 Read the MD&A section in the company's annual report for management explanations.",
        "🔁 Monitor closely over the next 1–2 quarters.",
    ],
    "HIGH": [
        "⚠️ Do not rely solely on reported earnings — focus on **cash flow statements**.",
        "📈 Use the **Trends** page to check if red flags are getting worse.",
        "🔢 Run **Benford's Law** analysis to check for digit-level anomalies.",
        "📊 Compare against industry peers on the **Compare** page.",
        "📰 Search for recent news: auditor changes, executive departures, SEC inquiries.",
    ],
    "VERY HIGH": [
        "🔴 Treat all reported financial figures with significant skepticism.",
        "💵 Focus exclusively on actual cash flows, not reported earnings.",
        "📰 Immediately search for SEC filings, audit opinions, and recent news.",
        "🔢 Run **Benford's Law** — digit anomalies often appear in severe cases.",
        "📊 Use the **Compare** page with known fraud cases as benchmarks.",
        "⚖️ Consider whether regulatory or legal proceedings are already underway.",
    ],
}
for step in next_steps[risk_label]:
    st.markdown(f"- {step}")

st.divider()
st.warning("""
⚠️ **Important Disclaimer**

This tool is for **educational purposes only** and does not constitute financial, legal, or investment advice.
A high risk score does not prove fraud — it indicates financial patterns resembling historical fraud cases.
Always consult a qualified financial professional before making investment decisions.
""")

st.caption("Data: Yahoo Finance · Model: Random Forest + Beneish M-Score · BA870/AC820 Boston University")
