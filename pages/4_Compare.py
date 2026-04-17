import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import FEATURES, FEATURE_LABELS, INDUSTRY_PM, RED_FLAGS, compute_features_all_years

st.set_page_config(page_title="Compare Companies", page_icon="📊", layout="wide")
st.title("📊 Compare Companies")
st.markdown("Compare fraud risk scores and key financial ratios across multiple companies.")

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input")
    tickers_raw = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, TSLA, GE")
    run_btn = st.button("🚀 Compare", use_container_width=True)
    st.divider()
    st.markdown("**💡 Tip:** Each company's industry is auto-detected from Yahoo Finance.")

if not run_btn:
    st.info("👈 Enter tickers and click **Compare**."); st.stop()

tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

results  = []
progress = st.progress(0, text="Fetching data…")
for i, tk in enumerate(tickers):
    progress.progress((i+1)/len(tickers), text=f"Fetching {tk}…")
    try:
        # Auto-detect industry per ticker
        info         = yf.Ticker(tk).info
        sector       = info.get("sector", "Other")
        industry_sel = SECTOR_MAP.get(sector, "Other")
        industry_pm  = INDUSTRY_PM[industry_sel]

        feat_all    = compute_features_all_years(tk, industry_pm)
        feat_latest = feat_all.iloc[-1][FEATURES].fillna(0)
        feat_df     = pd.DataFrame([feat_latest])

        rf_prob = rf_model.predict_proba(feat_df)[0][1]
        lr_prob = lr_model.predict_proba(feat_df)[0][1] if lr_model else None
        n_flags = sum(1 for f in FEATURES
                      if f in RED_FLAGS and not np.isnan(feat_latest[f])
                      and RED_FLAGS[f](float(feat_latest[f])))

        row = {"Ticker": tk, "Industry": industry_sel,
               "RF Score": round(rf_prob,4), "Red Flags": n_flags,
               "Verdict": "⚠️ HIGH" if rf_prob>0.5 else "✅ LOW"}
        if lr_prob is not None:
            row["LR Score"] = round(lr_prob,4)
        for feat in FEATURES:
            row[FEATURE_LABELS[feat]] = round(float(feat_latest[feat]),4)
        results.append(row)
    except Exception as ex:
        st.warning(f"⚠️ Could not fetch {tk}: {ex}")
progress.empty()

if not results:
    st.error("No valid data returned."); st.stop()

df = pd.DataFrame(results)

# ── Risk score bar chart ──────────────────────────────────────────────────────
st.subheader("🎯 Fraud Risk Score Comparison (Random Forest)")
bar_colors = ["#e74c3c" if r>0.5 else "#2ecc71" for r in df["RF Score"]]
fig_bar = go.Figure(go.Bar(
    x=df["Ticker"], y=df["RF Score"]*100,
    marker_color=bar_colors,
    text=[f"{r:.1%}" for r in df["RF Score"]],
    textposition="outside"
))
fig_bar.add_hline(y=50, line_dash="dash", line_color="red",
                  annotation_text="Risk Threshold (50%)")
fig_bar.update_layout(yaxis_title="Risk Score (%)", yaxis_range=[0,115],
                      height=360, margin=dict(t=20,b=10))
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Red flags + summary ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("🚩 Number of Red Flags")
    fig_flags = px.bar(df, x="Ticker", y="Red Flags",
                       color="Red Flags",
                       color_continuous_scale=["#2ecc71","#e74c3c"],
                       text="Red Flags")
    fig_flags.update_layout(height=300, margin=dict(t=20,b=10),
                             coloraxis_showscale=False)
    st.plotly_chart(fig_flags, use_container_width=True)

with col2:
    st.subheader("📋 Summary Table")
    summary_cols = ["Ticker","Industry","RF Score","Red Flags","Verdict"]
    if "LR Score" in df.columns:
        summary_cols.insert(3,"LR Score")
    st.dataframe(df[summary_cols], use_container_width=True, hide_index=True)

st.divider()

# ── Key ratio comparison ──────────────────────────────────────────────────────
st.subheader("📈 Key Ratio Comparison")
selected_ratios = st.multiselect(
    "Select ratios to compare",
    list(FEATURE_LABELS.values()),
    default=["Accrual Ratio","Profit Margin","Debt Ratio","Revenue Growth"]
)
inv_labels    = {v: k for k, v in FEATURE_LABELS.items()}
selected_keys = [inv_labels[r] for r in selected_ratios if r in inv_labels]

if selected_keys:
    for feat in selected_keys:
        label  = FEATURE_LABELS[feat]
        vals   = [df.loc[df["Ticker"]==tk, label].values[0]
                  if label in df.columns else 0 for tk in df["Ticker"]]
        fcolors = ["#e74c3c"
                   if feat in RED_FLAGS and not np.isnan(v) and RED_FLAGS[feat](v)
                   else "#3498db" for v in vals]
        fig_r = go.Figure(go.Bar(
            x=df["Ticker"], y=vals, marker_color=fcolors,
            text=[f"{v:.3f}" for v in vals], textposition="outside"
        ))
        fig_r.update_layout(title=label, height=280,
                             margin=dict(t=40,b=10), yaxis_title="Value")
        st.plotly_chart(fig_r, use_container_width=True)

st.divider()

# ── Radar chart ───────────────────────────────────────────────────────────────
st.subheader("🕸️ Radar Chart — Risk Profile")
radar_keys   = ['roa','profit_margin','current_ratio','accrual_ratio',
                'receivable_ratio','revenue_growth','debt_ratio']
radar_labels = [FEATURE_LABELS[k] for k in radar_keys]
palette      = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c"]

fig_radar = go.Figure()
for idx, row in df.head(6).iterrows():
    vals_norm = [(max(min(float(row.get(FEATURE_LABELS[k],0) or 0),1),-1)+1)/2
                 for k in radar_keys]
    vals_norm += [vals_norm[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_norm, theta=radar_labels+[radar_labels[0]],
        fill='toself', name=row["Ticker"],
        line_color=palette[idx % len(palette)]
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                        height=450, margin=dict(t=40,b=20))
st.plotly_chart(fig_radar, use_container_width=True)

st.caption("Industry auto-detected per ticker · Data: Yahoo Finance · For educational purposes only.")
