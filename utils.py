import yfinance as yf
import pandas as pd
import numpy as np
import time

FEATURES = [
    'roa', 'profit_margin', 'current_ratio', 'debt_ratio',
    'asset_turnover', 'ocf_ratio', 'sga_ratio', 'depr_ratio',
    'revenue_growth', 'asset_growth', 'income_growth',
    'accrual_ratio', 'cfo_to_income', 'receivable_ratio',
    'profit_margin_vs_industry'
]

FEATURE_LABELS = {
    'roa':                       "Return on Assets",
    'profit_margin':             "Profit Margin",
    'current_ratio':             "Current Ratio",
    'debt_ratio':                "Debt Ratio",
    'asset_turnover':            "Asset Turnover",
    'ocf_ratio':                 "OCF Ratio",
    'sga_ratio':                 "SG&A Ratio",
    'depr_ratio':                "Depreciation Ratio",
    'revenue_growth':            "Revenue Growth",
    'asset_growth':              "Asset Growth",
    'income_growth':             "Income Growth",
    'accrual_ratio':             "Accrual Ratio",
    'cfo_to_income':             "CFO / Net Income",
    'receivable_ratio':          "Receivable Ratio",
    'profit_margin_vs_industry': "Margin vs Industry",
}

INDUSTRY_PM = {
    "Technology": 0.20, "Healthcare": 0.12, "Finance": 0.18,
    "Consumer Goods": 0.08, "Energy": 0.06, "Industrials": 0.07,
    "Utilities": 0.10, "Real Estate": 0.15, "Other": 0.09,
}

# Red-flag thresholds
RED_FLAGS = {
    'accrual_ratio':             lambda x: x > 0.05,
    'cfo_to_income':             lambda x: x < 0.5,
    'receivable_ratio':          lambda x: x > 0.20,
    'revenue_growth':            lambda x: x > 0.50,
    'asset_growth':              lambda x: x > 0.50,
    'debt_ratio':                lambda x: x > 0.80,
    'profit_margin_vs_industry': lambda x: x > 0.15,
    'sga_ratio':                 lambda x: x > 0.30,
    'ocf_ratio':                 lambda x: x < 0.02,
}

def safe_col(df, *keys):
    for k in keys:
        if k in df.columns:
            return df[k]
    return pd.Series([np.nan] * len(df), index=df.index)


def fetch_all_data(ticker_sym: str):
    """Return (inc, bal, cf) DataFrames. Retries up to 3x for cloud reliability."""
    for attempt in range(3):
        try:
            tk  = yf.Ticker(ticker_sym)
            inc = tk.financials.T
            bal = tk.balance_sheet.T
            cf  = tk.cashflow.T
            if inc.empty or bal.empty or cf.empty:
                time.sleep(1.5)
                tk  = yf.Ticker(ticker_sym)
                inc = tk.financials.T
                bal = tk.balance_sheet.T
                cf  = tk.cashflow.T
            if inc.empty or bal.empty or cf.empty:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise ValueError(
                    f"No financial data found for '{ticker_sym}'. "
                    "Please check the ticker symbol and try again.")
            dates = inc.index.intersection(bal.index).intersection(cf.index)
            if len(dates) < 2:
                raise ValueError(f"'{ticker_sym}' has fewer than 2 years of data.")
            return (inc.loc[dates].sort_index(),
                    bal.loc[dates].sort_index(),
                    cf.loc[dates].sort_index())
        except ValueError:
            raise
        except Exception as e:
            if attempt == 2:
                raise ValueError(f"Could not fetch '{ticker_sym}': {e}")
            time.sleep(2)


def compute_features_all_years(ticker_sym: str, industry_pm: float) -> pd.DataFrame:
    """Return a DataFrame with one row per fiscal year, all 15 features computed."""
    inc, bal, cf = fetch_all_data(ticker_sym)

    revt  = safe_col(inc, "Total Revenue")
    ni    = safe_col(inc, "Net Income")
    xsga  = safe_col(inc, "Selling General And Administration")
    at    = safe_col(bal, "Total Assets")
    act   = safe_col(bal, "Current Assets")
    lct   = safe_col(bal, "Current Liabilities")
    lt    = safe_col(bal, "Total Liabilities Net Minority Interest", "Total Liabilities")
    rect  = safe_col(bal, "Accounts Receivable", "Net Receivables")
    dpc   = safe_col(cf,  "Depreciation And Amortization", "Depreciation")
    oancf = safe_col(cf,  "Operating Cash Flow",
                     "Cash Flow From Continuing Operating Activities")

    df = pd.DataFrame(index=revt.index)
    df['roa']            = ni / at
    df['profit_margin']  = ni / revt
    df['current_ratio']  = act / lct
    df['debt_ratio']     = lt  / at
    df['asset_turnover'] = revt / at
    df['ocf_ratio']      = oancf / at
    df['sga_ratio']      = xsga / revt
    df['depr_ratio']     = dpc  / at
    df['revenue_growth'] = revt.pct_change()
    df['asset_growth']   = at.pct_change()
    df['income_growth']  = ni.pct_change()
    df['accrual_ratio']  = (ni - oancf) / at
    df['cfo_to_income']  = oancf / ni.replace(0, np.nan)
    df['receivable_ratio'] = rect / revt
    df['profit_margin_vs_industry'] = df['profit_margin'] - industry_pm

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def compute_beneish(ticker_sym: str) -> tuple[float, dict]:
    """Compute Beneish M-Score for the most recent year. Returns (score, components)."""
    inc, bal, cf = fetch_all_data(ticker_sym)

    revt  = safe_col(inc, "Total Revenue")
    ni    = safe_col(inc, "Net Income")
    cogs  = safe_col(inc, "Cost Of Revenue")
    xsga  = safe_col(inc, "Selling General And Administration")
    at    = safe_col(bal, "Total Assets")
    ca    = safe_col(bal, "Current Assets")
    rect  = safe_col(bal, "Accounts Receivable", "Net Receivables")
    ppe   = safe_col(bal, "Net PPE")
    lt    = safe_col(bal, "Total Liabilities Net Minority Interest", "Total Liabilities")
    dpc   = safe_col(cf,  "Depreciation And Amortization", "Depreciation")
    oancf = safe_col(cf,  "Operating Cash Flow",
                     "Cash Flow From Continuing Operating Activities")

    def v(s, i=-1):
        try: return float(s.iloc[i])
        except: return np.nan
    def safe_div(a, b):
        return a / b if b and b != 0 and not np.isnan(b) else np.nan

    # DSRI
    dsri = safe_div(v(rect)/v(revt), v(rect,-2)/v(revt,-2))
    # GMI
    gm_t  = safe_div(v(revt)   - v(cogs),    v(revt))
    gm_t1 = safe_div(v(revt,-2)- v(cogs,-2), v(revt,-2))
    gmi = safe_div(gm_t1, gm_t)
    # AQI
    nca_t  = v(at)    - v(ca)    - v(ppe)
    nca_t1 = v(at,-2) - v(ca,-2) - v(ppe,-2)
    aqi = safe_div(nca_t/v(at), nca_t1/v(at,-2))
    # SGI
    sgi = safe_div(v(revt), v(revt,-2))
    # DEPI
    dep_t  = safe_div(v(dpc),    v(dpc)    + v(ppe))
    dep_t1 = safe_div(v(dpc,-2), v(dpc,-2) + v(ppe,-2))
    depi = safe_div(dep_t1, dep_t)
    # SGAI
    sgai = safe_div(v(xsga)/v(revt), v(xsga,-2)/v(revt,-2))
    # TATA
    tata = safe_div(v(ni) - v(oancf), v(at))
    # LVGI
    lev_t  = safe_div(v(lt),    v(at))
    lev_t1 = safe_div(v(lt,-2), v(at,-2))
    lvgi = safe_div(lev_t, lev_t1)

    components = {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
                  "DEPI": depi, "SGAI": sgai, "TATA": tata, "LVGI": lvgi}

    vals = [v if v is not None and not np.isnan(v) else 1.0
            for v in components.values()]
    m_score = (-4.84 + 0.920*vals[0] + 0.528*vals[1] + 0.404*vals[2]
               + 0.892*vals[3] + 0.115*vals[4] - 0.172*vals[5]
               + 4.679*vals[6] - 0.327*vals[7])
    return m_score, components


def benford_analysis(ticker_sym: str) -> tuple[pd.DataFrame, float]:
    """Run Benford's Law analysis. Returns (digit_df, MAD)."""
    inc, bal, cf = fetch_all_data(ticker_sym)
    cols_to_check = ["Total Revenue", "Net Income", "Total Assets",
                     "Total Liabilities Net Minority Interest",
                     "Selling General And Administration",
                     "Cost Of Revenue", "Net PPE",
                     "Accounts Receivable"]

    all_digits = []
    for df in [inc, bal, cf]:
        for col in cols_to_check:
            if col in df.columns:
                vals = df[col].dropna().abs()
                for val in vals:
                    if val > 0:
                        s = str(int(val))
                        if s[0].isdigit() and s[0] != '0':
                            all_digits.append(int(s[0]))

    digits = list(range(1, 10))
    benford_expected = [np.log10(1 + 1/d) for d in digits]

    counts = pd.Series(all_digits).value_counts().reindex(digits, fill_value=0)
    observed = counts / counts.sum()

    mad = float(np.mean(np.abs(observed.values - benford_expected)))

    result = pd.DataFrame({
        "Digit": digits,
        "Observed (%)": (observed.values * 100).round(2),
        "Expected (%)": [round(b*100, 2) for b in benford_expected],
    })
    return result, mad
