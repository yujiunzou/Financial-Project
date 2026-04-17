import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (FEATURES, FEATURE_LABELS, INDUSTRY_PM, RED_FLAGS,
                   compute_features_all_years, compute_beneish)

st.set_page_config(page_title="Stock Analysis", page_icon="🔎", layout="wide")
st.title("🔎 Single Stock Analysis")
st.markdown("Enter a stock ticker to get a complete fraud risk breakdown.")

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

industry_avg = {
    'roa': 0.08, 'profit_margin': 0.10, 'current_ratio': 1.5,
    'debt_ratio': 0.5, 'asset_turnover': 0.7, 'ocf_ratio': 0.1,
    'sga_ratio': 0.15, 'depr_ratio': 0.03, 'revenue_growth': 0.08,
    'asset_growth': 0.06, 'income_growth': 0.07, 'accrual_ratio': 0.02,
    'cfo_to_income': 0.9, 'receivable_ratio': 0.12,
    'profit_margin_vs_industry': 0.0,
}

FEATURE_EXPLAIN = {
    'roa':                       ("Return on Assets",     "Net Income / Total Assets"),
    'profit_margin':             ("Profit Margin",        "Net Income / Revenue"),
    'current_ratio':             ("Current Ratio",        "Current Assets / Current Liabilities"),
    'debt_ratio':                ("Debt Ratio",           "Total Liabilities / Total Assets"),
    'asset_turnover':            ("Asset Turnover",       "Revenue / Total Assets"),
    'ocf_ratio':                 ("OCF Ratio",            "Operating Cash Flow / Total Assets"),
    'sga_ratio':                 ("SG&A Ratio",           "SG&A Expense / Revenue"),
    'depr_ratio':                ("Depreciation Ratio",   "Depreciation / Total Assets"),
    'revenue_growth':            ("Revenue Growth",       "YoY Revenue Change"),
    'asset_growth':              ("Asset Growth",         "YoY Asset Change"),
    'income_growth':             ("Income Growth",        "YoY Net Income Change"),
    'accrual_ratio':             ("Accrual Ratio",        "(Net Income − CFO) / Assets"),
    'cfo_to_income':             ("CFO / Net Income",     "Operating Cash Flow / Net Income"),
    'receivable_ratio':          ("Receivable Ratio",     "Accounts Receivable / Revenue"),
    'profit_margin_vs_industry': ("Margin vs Industry",   "Company Margin − Industry Average"),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    run_btn = st.button("🚀 Analyze", use_container_width=True)
    st.divider()
    st.caption("Try: ENRN (Enron), WCOM (WorldCom)")

if not run_btn:
    st.info("👈 Enter a ticker and click **Analyze**."); st.stop()

# ── Auto-detect industry ──────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker_input}**…"):
    try:
        info         = yf.Ticker(ticker_input).info
        sector       = info.get("sector", "Other")
        industry_sel = SECTOR_MAP.get(sector, "Other")
        industry_pm  = INDUSTRY_PM[industry_sel]
        feat_all     = compute_features_all_years(ticker_input, industry_pm)
        m_score, beneish_components = compute_beneish(ticker_input)
    except Exception as e:
        st.error(str(e)); st.stop()

st.caption(f"🏭 Auto-detected industry: **{industry_sel}** (sector: {sector})")

feat_latest = feat_all.iloc[-1][FEATURES].fillna(0)
feat_df     = pd.DataFrame([feat_latest])

rf_prob      = rf_model.predict_proba(feat_df)[0][1]
lr_prob      = lr_model.predict_proba(feat_df)[0][1] if lr_model else None
beneish_flag = int(m_score > -1.78)
scores       = [rf_prob, beneish_flag] + ([lr_prob] if lr_prob else [])
ensemble     = float(np.mean(scores))

# ── Top metrics ───────────────────────────────────────────────────────────────
st.subheader(f"Results for: {ticker_input}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Beneish M-Score",      f"{m_score:.2f}",
          delta="⚠️ Risk" if beneish_flag else "✅ Safe", delta_color="inverse")
c2.metric("RF Fraud Probability",  f"{rf_prob:.1%}")
c3.metric("LR Fraud Probability",  f"{lr_prob:.1%}" if lr_prob else "N/A")
c4.metric("Ensemble Score",        f"{ensemble:.1%}",
          delta="⚠️ HIGH RISK" if ensemble>0.5 else "✅ LOW RISK", delta_color="inverse")

if ensemble > 0.5:
    st.error("🚨 **High manipulation risk.** Multiple indicators flag potential earnings manipulation.")
else:
    st.success("✅ **Low manipulation risk.** Financial ratios appear within normal ranges.")

st.divider()

# ── Gauge + Beneish ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Ensemble Fraud Risk Gauge**")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=ensemble*100,
        title={"text":"Risk Score (%)"},
        gauge={"axis":{"range":[0,100]},
               "bar":{"color":"#e74c3c" if ensemble>0.5 else "#2ecc71"},
               "steps":[{"range":[0,33],"color":"#d5f5e3"},
                        {"range":[33,66],"color":"#fef9e7"},
                        {"range":[66,100],"color":"#fadbd8"}],
               "threshold":{"line":{"color":"red","width":4},"value":50}}))
    fig_g.update_layout(height=300, margin=dict(t=30,b=10))
    st.plotly_chart(fig_g, use_container_width=True)

with col2:
    st.markdown("**Beneish M-Score Components**")
    b_df = pd.DataFrame({
        "Component": list(beneish_components.keys()),
        "Value":     [round(v,3) if not np.isnan(v) else 0
                      for v in beneish_components.values()],
    })
    fig_b = go.Figure(go.Bar(
        x=b_df["Component"], y=b_df["Value"],
        marker_color=["#e74c3c" if v>1 else "#2ecc71" for v in b_df["Value"]],
        text=b_df["Value"].round(3), textposition="outside"
    ))
    fig_b.add_hline(y=1, line_dash="dash", line_color="red",
                    annotation_text="Benchmark = 1")
    fig_b.update_layout(height=300, margin=dict(t=30,b=10), yaxis_title="Index Value")
    st.plotly_chart(fig_b, use_container_width=True)

st.divider()

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.subheader("🌡️ Fraud Risk Heatmap")
st.caption("Red = potential red flag · Green = within normal range")

hm_risk = []
for feat in FEATURES:
    val    = float(feat_latest[feat])
    is_flag = feat in RED_FLAGS and not np.isnan(val) and RED_FLAGS[feat](val)
    hm_risk.append(1 if is_flag else 0)

n_flags = sum(hm_risk)
hm_vals = np.array(hm_risk).reshape(3,5)
hm_text = [[FEATURE_LABELS[FEATURES[i*5+j]] for j in range(5)] for i in range(3)]

fig_hm = go.Figure(go.Heatmap(
    z=hm_vals, text=hm_text, texttemplate="%{text}",
    colorscale=[[0,"#d5f5e3"],[1,"#e74c3c"]],
    showscale=False, xgap=3, ygap=3,
))
fig_hm.update_layout(height=220, margin=dict(t=10,b=10,l=10,r=10),
                     xaxis=dict(showticklabels=False),
                     yaxis=dict(showticklabels=False))
st.plotly_chart(fig_hm, use_container_width=True)
st.markdown(f"**{n_flags} out of {len(FEATURES)} indicators flagged.**")

st.divider()

# ── Peer comparison ───────────────────────────────────────────────────────────
st.subheader("🏢 Peer Comparison vs Industry Average")
peer_feats = ['roa','profit_margin','current_ratio',
              'debt_ratio','asset_turnover','accrual_ratio']
comp_df = pd.DataFrame({
    "Feature":      [FEATURE_LABELS[f] for f in peer_feats],
    ticker_input:   [round(float(feat_latest[f]),4) for f in peer_feats],
    "Industry Avg": [round(industry_avg[f],4)       for f in peer_feats],
})
fig_peer = go.Figure()
fig_peer.add_trace(go.Bar(name=ticker_input, x=comp_df["Feature"],
                          y=comp_df[ticker_input], marker_color="#3498db"))
fig_peer.add_trace(go.Bar(name="Industry Avg", x=comp_df["Feature"],
                          y=comp_df["Industry Avg"], marker_color="#95a5a6"))
fig_peer.update_layout(barmode="group", height=360,
                       margin=dict(t=20,b=10), yaxis_title="Ratio Value")
st.plotly_chart(fig_peer, use_container_width=True)

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("📊 Feature Importance (Random Forest)")
importance = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
importance.index = [FEATURE_LABELS[f] for f in importance.index]
fig_imp = px.bar(importance, orientation='h',
                 color=importance, color_continuous_scale=["#2ecc71","#e74c3c"],
                 labels={"value":"Importance","index":"Feature"})
fig_imp.update_layout(height=420, coloraxis_showscale=False, margin=dict(t=10,b=10))
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# ── Ratio table ───────────────────────────────────────────────────────────────
st.subheader("📋 Full Financial Ratio Breakdown")
rows = []
for feat in FEATURES:
    val  = feat_latest[feat]
    flag = feat in RED_FLAGS and not pd.isna(val) and RED_FLAGS[feat](float(val))
    ln, formula = FEATURE_EXPLAIN[feat]
    rows.append({"Indicator": ln, "Formula": formula,
                 "Value": round(float(val),4) if not pd.isna(val) else "N/A",
                 "Status": "🚩 Flag" if flag else "✅ Normal"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption("Data: Yahoo Finance · For educational purposes only.")
