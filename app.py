import streamlit as st

st.set_page_config(
    page_title="Financial Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Financial Fraud Detection Dashboard")
st.markdown("##### AI-powered earnings manipulation risk analysis for publicly listed companies")
st.divider()

# ── What this app does ────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("### 🔎 Single Stock\nGet a full fraud risk breakdown for any ticker — ratios, heatmap, peer comparison.")
with col2:
    st.info("### 📈 Financial Trends\nVisualize multi-year trends to spot unusual jumps in key financial metrics.")
with col3:
    st.info("### 🔢 Benford's Law\nDetect abnormal digit patterns in financial statements using Benford's Law analysis.")
with col4:
    st.info("### 📊 Compare Companies\nCompare fraud risk scores across multiple companies side by side.")

st.divider()

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("### 🗺️ How It Works")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown("**① Fetch Data**\n\nReal-time financial statements via `yfinance` (Yahoo Finance)")
with c2:
    st.markdown("**② Compute Ratios**\n\n15 financial ratios automatically calculated from income, balance sheet & cash flow")
with c3:
    st.markdown("**③ Fraud Indicators**\n\nBeneish M-Score + Benford's Law + ML models (Logistic Regression & Random Forest)")
with c4:
    st.markdown("**④ Risk Score**\n\nAggregate fraud probability score with red-flag heatmap")
with c5:
    st.markdown("**⑤ Visualize**\n\nInteractive charts: trends, peer comparison, feature importance, digit distribution")

st.divider()

# ── Models used ───────────────────────────────────────────────────────────────
st.markdown("### 🤖 Models & Methods")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
**📐 Beneish M-Score**
- Classical accounting-based fraud indicator
- Uses 8 financial ratios
- Threshold: M > −1.78 → Manipulation risk
- Used as training label for ML models
""")
with col_b:
    st.markdown("""
**🔢 Benford's Law**
- Tests leading-digit distribution of financial figures
- Normal data follows a predictable log pattern
- Deviation (MAD > 0.02) signals potential manipulation
- Applied per-company across 8 financial statement items
""")
with col_c:
    st.markdown("""
**🌲 Machine Learning**
- **Logistic Regression**: interpretable baseline
- **Random Forest**: main prediction model
- Trained on WRDS Compustat (2000–2025)
- 15 features: ratios, growth rates, accruals
""")

st.divider()

# ── Key features ──────────────────────────────────────────────────────────────
st.markdown("### 📌 Key Financial Indicators")
features_table = {
    "Indicator": [
        "Accrual Ratio", "Revenue Growth", "Asset Growth",
        "CFO / Net Income", "Receivable Ratio",
        "ROA", "Profit Margin vs Industry",
        "Debt Ratio", "SG&A Ratio"
    ],
    "Why It Matters": [
        "Gap between net income and cash flow — high accruals = earnings quality risk",
        "Sudden spikes often precede restatements",
        "Rapid unexplained asset growth may signal balance sheet inflation",
        "Low cash conversion relative to reported income = red flag",
        "Rising receivables vs revenue suggests premature revenue recognition",
        "Sudden ROA improvement can indicate manipulation",
        "Unusually high margins vs peers warrant scrutiny",
        "High leverage creates pressure to manipulate earnings",
        "Rising SG&A relative to revenue signals operational stress"
    ]
}
import pandas as pd
st.dataframe(pd.DataFrame(features_table), use_container_width=True, hide_index=True)

st.divider()
st.caption("Data: Yahoo Finance (`yfinance`) · Model trained on WRDS Compustat 2000–2025 · BA870/AC820 Boston University · For educational purposes only.")
