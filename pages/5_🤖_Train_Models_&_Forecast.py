import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

from ui_components import load_full_ui, render_footer

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Model Training and Forcasting", page_icon="☁️", layout="wide")
st.write("")
load_full_ui()

px.defaults.template = "plotly_white"
CONFIG = {"displayModeBar": False, "scrollZoom": False}

# ======================================================
# SPEED SETTINGS FOR STREAMLIT CLOUD / LOCAL DEPLOYMENT
# ======================================================
# These values keep RF + XGBoost fast while preserving the modelling workflow.
FAST_RF_ESTIMATORS = 150
FAST_RF_MAX_DEPTH = 12
FAST_XGB_ESTIMATORS = 180
FAST_XGB_MAX_DEPTH = 4
FAST_XGB_LEARNING_RATE = 0.07
FAST_PCA_VARIANCE = 0.85
FAST_PCA_MAX_COMPONENTS = 6
FAST_GRU_EPOCHS = 10
FAST_GRU_LOOKBACKS = (24, 48, 72)

# ======================================================
# DATA LOAD
# ======================================================
# Final modelling dataset must come from the Feature Engineering & PCA page.
# In the PCA page, df_model_ready should contain engineered features plus
# selected PCA components/metadata saved in session state.
if "df_model_ready" in st.session_state:
    df = st.session_state["df_model_ready"].copy()
else:
    st.warning(
        """
    ⬅ Please follow the application workflow before using this page:

    1️⃣ Go to the Home page  
    2️⃣ Upload the required datasets from the sidebar  
    3️⃣ Confirm the uploaded dataset on the Data Processing page  
    4️⃣ Complete Data Insights & Cleaning  
    5️⃣ Run Exploratory Analysis & PCA  
    6️⃣ Return here for Model Training & Forecasting
    """
    )
    render_footer()
    st.stop()

# ======================================================
# CSS - IMAGE-LIKE PROFESSIONAL LIGHT LAYOUT
# ======================================================
st.markdown(
    """
<style>
.block-container {
    padding-top: 0.7rem !important;
    padding-left: 1.0rem !important;
    padding-right: 1.0rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 100% !important;
    background: #f7fbff;
}
.main-title {
    font-size: 32px;
    font-weight: 950;
    color: #07184a;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
}
.main-subtitle {
    font-size: 13px;
    color: #475569;
    font-weight: 650;
    margin-bottom: 6px;
}
.top-card, .section-wrap, .mini-card, .status-card {
    background:#ffffff;
    border:1px solid #dbe7ff;
    border-radius:16px;
    box-shadow:0 5px 16px rgba(15,23,42,.045);
}
.top-card {padding: 10px 12px 2px 12px; min-height: 78px;}\n.control-row {margin-top: 10px; margin-bottom: 14px;}\n.header-note {font-size:13px; color:#475569; font-weight:650; margin-bottom:10px;}
.section-wrap {padding: 14px 16px; margin-top: 18px; margin-bottom: 24px;}
.mini-card {padding: 10px 12px; min-height: 80px;}
.status-card {padding: 10px 12px; min-height: 74px; display:flex; align-items:center; gap:10px;}
.section-head {display:flex; align-items:center; gap:20px; margin-bottom: 14px;}
.step-badge {
    min-width:42px; height:42px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    color:white; font-weight:950; font-size:21px;
    box-shadow:0 6px 14px rgba(37,99,235,.25);
}
.badge-blue {background:linear-gradient(135deg,#2f80ff,#0057e7);}
.badge-purple {background:linear-gradient(135deg,#a855f7,#7c3aed);}
.badge-green {background:linear-gradient(135deg,#31c65b,#16a34a);}
.section-title {font-size:20px; font-weight:950; color:#07184a; letter-spacing:-0.02em;}
.section-note {font-size:12px; color:#475569; font-weight:650; margin-left:5px;}
.select-target {font-size:12px; color:#0f172a; font-weight:750; text-align:right;}
.small-title {font-size:13px; color:#07184a; font-weight:950; margin-bottom:6px;}
.green-text {color:#16a34a; font-weight:950;}
.blue-text {color:#0057e7; font-weight:950;}
.purple-text {color:#7c3aed; font-weight:950;}
.info-strip {
    background:linear-gradient(90deg,#eef6ff,#ffffff);
    border:1px solid #bfdbfe;
    border-radius:12px;
    padding:10px 13px;
    color:#0f3f9e;
    font-size:13px;
    font-weight:750;
}
.best-box {
    background:#eff6ff;
    border:1px solid #bfdbfe;
    border-radius:12px;
    padding:14px 14px;
    color:#0057e7;
    font-size:15px;
    font-weight:950;
    margin-top:8px;
}
.best-green {
    background:#f0fdf4;
    border-color:#bbf7d0;
    color:#15803d;
}
.metric-grid {display:grid; grid-template-columns:repeat(2,1fr); gap:10px;}
.metric-box {
    border:1px solid #dbe7ff;
    border-radius:12px;
    padding:12px 8px;
    background:#ffffff;
    text-align:center;
    min-height:82px;
}
.metric-label {font-size:12px; color:#0f172a; font-weight:850;}
.metric-value {font-size:26px; line-height:1.1; font-weight:950; margin-top:7px;}
.table-note {font-size:11px; color:#64748b; font-weight:650; margin-top:4px;}
.footer-strip {
    background:#ffffff;
    border:1px solid #dbe7ff;
    border-radius:10px;
    padding:9px 12px;
    font-size:12px;
    color:#334155;
    font-weight:650;
}
div[data-testid="stVerticalBlock"] {gap: 0.55rem !important;}
div[data-testid="stHorizontalBlock"] {gap: 0.55rem !important;}
div[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden;}
div[data-testid="stDataFrame"] table {font-size:12px;}
div[data-testid="stButton"] button {
    border-radius:10px !important;
    min-height:44px !important;
    font-weight:900 !important;
    border:1px solid #bfdbfe !important;
    color:white !important;
    background:linear-gradient(135deg,#2563eb,#0057e7) !important;
}
div[data-testid="stSelectbox"] label, div[data-testid="stDateInput"] label {
    font-size:12px !important;
    font-weight:950 !important;
    color:#0f172a !important;
    text-transform:uppercase;
}
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================
# BASIC HELPERS
# ======================================================
def clean_name(text):
    return str(text).lower().replace(" ", "").replace("_", "").replace(".", "").replace("-", "").replace("₂", "2")


def find_col(possible_names, data):
    for name in possible_names:
        n = clean_name(name)
        for col in data.columns:
            if n in clean_name(col):
                return col
    return None


def detect_time_col(data):
    for col in data.columns:
        low = str(col).lower()
        if "date" in low or "time" in low or "timestamp" in low:
            return col
    return None


def fmt(v, d=2):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{float(v):,.{d}f}"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrics_dict(y_true, y_pred):
    return {
        "R² Score": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": mape(y_true, y_pred),
    }


def style_fig(fig, height=260, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=24, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=10.5, color="#07184a"),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=10)) if legend else None,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
    return fig


def target_label(t):
    t_clean = clean_name(t)
    if "pm25" in t_clean:
        return "PM2.5"
    if "no2" in t_clean:
        return "NO₂"
    if "aqi" in t_clean:
        return "AQI"
    return str(t)


def pollutant_unit(t):
    t_clean = clean_name(t)
    if "pm25" in t_clean:
        return "µg/m³"
    if "no2" in t_clean:
        return "ppb"
    return "AQI"


def render_status(title, subtitle, color="green"):
    icon_color = "#16a34a" if color == "green" else "#2563eb"
    st.markdown(
        f"""
        <div class="status-card">
            <div style="font-size:19px;color:{icon_color};font-weight:950;">●</div>
            <div>
                <div style="font-weight:950;color:{icon_color};font-size:14px;">{title}</div>
                <div style="font-size:12px;color:#334155;font-weight:650;">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def small_metric(label, value, color_class="green-text"):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================
# COLUMN DETECTION + FILTERING
# ======================================================
time_col = detect_time_col(df)
if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
else:
    st.error("No date/time column was found. Please check the cleaned dataset.")
    render_footer()
    st.stop()

aqi_col = find_col(["AQI", "air quality index"], df)
no2_col = find_col(["NO2", "NO₂", "nitrogen dioxide"], df)
pm25_col = find_col(["PM2.5", "PM25", "PM 2.5"], df)
targets = [c for c in [aqi_col, no2_col, pm25_col] if c]

if not targets:
    st.warning("AQI, NO₂ or PM2.5 columns were not found.")
    render_footer()
    st.stop()

# ======================================================
# TOP CONTROL BAR
# ======================================================
min_date = df[time_col].min().date()
max_date = df[time_col].max().date()

st.markdown('<div class="main-title">☁️ Model Training and Forcasting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-note">Train Random Forest, XGBoost and GRU models using the model-ready dataset from the Feature Engineering & PCA page.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="control-row">', unsafe_allow_html=True)
c_target, c_horizon, c_date = st.columns([1.4, 1.1, 1.8], gap="large")
with c_target:
    selected_target = st.selectbox("Select Target Parameter", targets, format_func=target_label, key="forecast_target")
with c_horizon:
    horizon_label = st.selectbox("Forecast Horizon", ["24 Hours", "48 Hours", "72 Hours"], index=0, key="forecast_horizon")
    future_steps = int(horizon_label.split()[0])
with c_date:
    date_range = st.date_input("Date Range (for training)", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="forecast_date_range")
st.markdown('</div>', unsafe_allow_html=True)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df[time_col].dt.date >= start_date) & (df[time_col].dt.date <= end_date)
df_work = df.loc[mask].copy().reset_index(drop=True)
if len(df_work) < 100:
    st.warning("The selected date range has too few rows. Please choose a wider training period.")
    render_footer()
    st.stop()

# ======================================================
# FEATURE ENGINEERING FOR MODELLING
# ======================================================
def is_time_or_id_column(col):
    c = clean_name(col)
    blocked = ["date", "timestamp", "datetime", "index", "unnamed", "recordid", "sensorid", "stationid", "id"]
    # Keep engineered time columns, but remove raw IDs/dates.
    if c in ["hour", "dayofweek", "dayofweek", "month", "isweekend", "hoursin", "hourcos", "daysin", "daycos", "monthsin", "monthcos"]:
        return False
    return any(k in c for k in blocked)


def is_lag_or_roll(col):
    c = clean_name(col)
    return "lag" in c or "roll" in c


def is_target_leakage_feature(col, target):
    c = clean_name(col)
    t = clean_name(target)
    if is_lag_or_roll(col):
        return False
    if c == t:
        return True
    aliases = {
        clean_name(aqi_col) if aqi_col else "aqi": ["aqi", "airqualityindex"],
        clean_name(no2_col) if no2_col else "no2": ["no2", "nitrogendioxide"],
        clean_name(pm25_col) if pm25_col else "pm25": ["pm25", "pm2.5"],
    }.get(t, [t])
    for a in aliases:
        aa = clean_name(a)
        if aa and aa in c and any(k in c for k in ["ratio", "dispersion", "index"]):
            return True
    return False


def get_numeric_cols(data):
    cols = []
    for col in data.columns:
        if col == time_col or is_time_or_id_column(col):
            continue
        s = pd.to_numeric(data[col], errors="coerce")
        if s.notna().sum() >= 30 and s.nunique(dropna=True) > 1:
            cols.append(col)
    return cols


def add_time_features(data):
    out = data.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out["hour"] = out[time_col].dt.hour
    out["dayofweek"] = out[time_col].dt.dayofweek
    out["month"] = out[time_col].dt.month
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["day_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["day_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_target_lag_features(data, target):
    out = data.copy()
    y = pd.to_numeric(out[target], errors="coerce")
    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"{target}_lag_{lag}"] = y.shift(lag)
    for win in [6, 12, 24]:
        out[f"{target}_rollmean_{win}"] = y.shift(1).rolling(win, min_periods=1).mean()
        out[f"{target}_rollstd_{win}"] = y.shift(1).rolling(win, min_periods=2).std()
    return out


def base_feature_candidates_for_target(data, target):
    numeric_cols = get_numeric_cols(data)
    out = []
    for c in numeric_cols:
        if c in targets or c == target:
            continue
        if is_lag_or_roll(c):
            continue
        if is_target_leakage_feature(c, target):
            continue
        out.append(c)
    return list(dict.fromkeys(out))


def build_target_dataset(data, target):
    model_df = data.copy().sort_values(time_col).reset_index(drop=True)
    model_df = add_time_features(model_df)
    model_df = add_target_lag_features(model_df, target)

    base_cols = base_feature_candidates_for_target(model_df, target)
    time_cols = [c for c in ["hour", "dayofweek", "month", "is_weekend", "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"] if c in model_df.columns]
    lag_cols = [c for c in model_df.columns if c.startswith(f"{target}_lag_") or c.startswith(f"{target}_roll")]
    original_feature_cols = list(dict.fromkeys(base_cols + lag_cols + time_cols))

    keep = [time_col, target] + original_feature_cols
    modelling = model_df[keep].copy()
    for c in [target] + original_feature_cols:
        modelling[c] = pd.to_numeric(modelling[c], errors="coerce")
    modelling = modelling.replace([np.inf, -np.inf], np.nan).dropna(subset=[target] + original_feature_cols).reset_index(drop=True)
    return modelling, base_cols, lag_cols, time_cols, original_feature_cols


def split_time_df(modelling, feature_cols, target, test_size=0.15):
    split_idx = max(30, int(len(modelling) * (1 - test_size)))
    split_idx = min(split_idx, len(modelling) - 10)
    train = modelling.iloc[:split_idx].copy().reset_index(drop=True)
    test = modelling.iloc[split_idx:].copy().reset_index(drop=True)
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y_train = train[target].copy()
    y_test = test[target].copy()
    return train, test, X_train, X_test, y_train, y_test


def apply_pca_to_tree_inputs(train_df, test_df, base_cols, lag_cols, time_cols, target, variance_threshold=FAST_PCA_VARIANCE, max_components=FAST_PCA_MAX_COMPONENTS):
    """Fit StandardScaler + PCA on TRAINING ONLY for RF/XGBoost.

    PCA is applied only to independent environmental/traffic drivers.
    Lag, rolling and cyclic time features are kept outside PCA because they
    are important for short-term forecasting and recursive prediction.
    """
    usable_base = []
    for col in base_cols:
        if col not in train_df.columns:
            continue
        tr = pd.to_numeric(train_df[col], errors="coerce")
        if tr.notna().sum() >= 30 and tr.nunique(dropna=True) > 1:
            usable_base.append(col)

    keep_cols = [c for c in lag_cols + time_cols if c in train_df.columns]

    if len(usable_base) < 2:
        X_train = train_df[keep_cols + usable_base].copy()
        X_test = test_df[keep_cols + usable_base].copy()
        for c in X_train.columns:
            med = X_train[c].median()
            X_train[c] = X_train[c].fillna(med)
            X_test[c] = X_test[c].fillna(med)
        return X_train, X_test, None

    Xb_train = train_df[usable_base].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xb_test = test_df[usable_base].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = Xb_train.median(numeric_only=True)
    Xb_train = Xb_train.fillna(medians)
    Xb_test = Xb_test.fillna(medians)

    scaler = StandardScaler()
    Xb_train_s = scaler.fit_transform(Xb_train)
    Xb_test_s = scaler.transform(Xb_test)

    pca_full = PCA().fit(Xb_train_s)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = max(2, min(n_components, max_components, len(usable_base), Xb_train_s.shape[0]))

    pca = PCA(n_components=n_components)
    train_pc = pca.fit_transform(Xb_train_s)
    test_pc = pca.transform(Xb_test_s)
    pc_cols = [f"PC{i+1}" for i in range(n_components)]

    X_train = pd.DataFrame(train_pc, columns=pc_cols, index=train_df.index)
    X_test = pd.DataFrame(test_pc, columns=pc_cols, index=test_df.index)

    other_train = train_df[keep_cols].copy()
    other_test = test_df[keep_cols].copy()
    for c in keep_cols:
        med = pd.to_numeric(other_train[c], errors="coerce").median()
        other_train[c] = pd.to_numeric(other_train[c], errors="coerce").fillna(med)
        other_test[c] = pd.to_numeric(other_test[c], errors="coerce").fillna(med)

    X_train = pd.concat([X_train.reset_index(drop=True), other_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), other_test.reset_index(drop=True)], axis=1)

    loadings = pd.DataFrame(pca.components_.T, index=usable_base, columns=pc_cols)
    explained = pd.DataFrame({
        "Component": pc_cols,
        "Explained Variance %": pca.explained_variance_ratio_ * 100,
        "Cumulative Variance %": np.cumsum(pca.explained_variance_ratio_) * 100,
    })

    pca_pack = {
        "enabled": True,
        "scaler": scaler,
        "pca": pca,
        "base_cols": usable_base,
        "pc_cols": pc_cols,
        "other_cols": keep_cols,
        "medians": medians,
        "loadings": loadings,
        "explained": explained,
        "variance_explained": float(np.sum(pca.explained_variance_ratio_) * 100),
        "n_components": n_components,
    }
    return X_train, X_test, pca_pack


def transform_latest_with_pca(latest_df, pca_pack):
    base = latest_df[pca_pack["base_cols"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    base = base.fillna(pca_pack["medians"])
    base_s = pca_pack["scaler"].transform(base)
    pc = pca_pack["pca"].transform(base_s)
    pc_df = pd.DataFrame(pc, columns=pca_pack["pc_cols"])
    other = latest_df[pca_pack["other_cols"]].copy()
    for c in pca_pack["other_cols"]:
        other[c] = pd.to_numeric(other[c], errors="coerce").fillna(0)
    return pd.concat([pc_df.reset_index(drop=True), other.reset_index(drop=True)], axis=1)

# ======================================================
# MODEL TRAINING HELPERS
# ======================================================
def train_rf_grid(X_train, y_train):
    """Fast RF training.

    This replaces expensive GridSearchCV during normal dashboard use.
    It keeps the same return format as the previous grid function so the
    rest of the page continues to work.
    """
    model = RandomForestRegressor(
        n_estimators=FAST_RF_ESTIMATORS,
        max_depth=FAST_RF_MAX_DEPTH,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    params = {
        "n_estimators": FAST_RF_ESTIMATORS,
        "max_depth": FAST_RF_MAX_DEPTH,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
    }
    return model, params, np.nan


def train_xgb_grid(X_train, y_train):
    """Fast XGBoost training.

    Uses histogram tree construction and a compact configuration instead of
    GridSearchCV, which was the main reason the page felt slow.
    """
    if not XGBOOST_AVAILABLE:
        return None, {}, np.nan
    model = XGBRegressor(
        n_estimators=FAST_XGB_ESTIMATORS,
        max_depth=FAST_XGB_MAX_DEPTH,
        learning_rate=FAST_XGB_LEARNING_RATE,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    params = {
        "n_estimators": FAST_XGB_ESTIMATORS,
        "max_depth": FAST_XGB_MAX_DEPTH,
        "learning_rate": FAST_XGB_LEARNING_RATE,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
    }
    return model, params, np.nan

def build_gru_windows(X_scaled, y_scaled, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i, :])
        y_seq.append(y_scaled[i])
    return np.asarray(X_seq), np.asarray(y_seq)


def train_gru_sequence(X_train, X_test, y_train, y_test, epochs=30, lookback=24):
    if not TENSORFLOW_AVAILABLE:
        return None, None, None, None, None, "TensorFlow is not installed. Add tensorflow to requirements.txt to train GRU."
    lookback = int(max(6, min(int(lookback), 72)))
    if len(X_train) < lookback + 50 or len(X_test) < 10:
        return None, None, None, None, None, f"Not enough rows for GRU lookback={lookback}."

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()

    X_train_g, y_train_g = build_gru_windows(X_train_s, y_train_s, lookback)
    X_context = np.vstack([X_train_s[-lookback:], X_test_s])
    X_test_g = np.asarray([X_context[i:i + lookback, :] for i in range(len(X_test_s))])

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        GRU(64, input_shape=(lookback, X_train_g.shape[2]), return_sequences=True),
        Dropout(0.20),
        GRU(32, return_sequences=False),
        Dropout(0.15),
        Dense(24, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    history = model.fit(
        X_train_g,
        y_train_g,
        validation_split=0.15,
        epochs=epochs,
        batch_size=64,
        callbacks=[es],
        verbose=0,
        shuffle=False,
    )

    train_pred_s = model.predict(X_train_g, verbose=0)
    test_pred_s = model.predict(X_test_g, verbose=0)
    train_pred = y_scaler.inverse_transform(train_pred_s).ravel()
    test_pred = y_scaler.inverse_transform(test_pred_s).ravel()
    y_train_eval = y_train.iloc[lookback:].reset_index(drop=True)
    y_test_eval = y_test.reset_index(drop=True)

    pack = {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "lookback": lookback,
        "feature_cols": list(X_train.columns),
        "history": history.history,
        "best_val_loss": float(np.min(history.history.get("val_loss", [np.inf]))),
        "epochs_run": len(history.history.get("loss", [])),
    }
    return train_pred, test_pred, y_train_eval, y_test_eval, pack, None


def train_gru_auto(X_train, X_test, y_train, y_test, epochs=30, lookbacks=(24, 48, 72)):
    attempts, errors = [], []
    for lb in lookbacks:
        train_pred, test_pred, y_train_eval, y_test_eval, pack, err = train_gru_sequence(X_train, X_test, y_train, y_test, epochs, lb)
        if err:
            errors.append(f"{lb}h: {err}")
            continue
        attempts.append((pack["best_val_loss"], train_pred, test_pred, y_train_eval, y_test_eval, pack))
    if not attempts:
        return None, None, None, None, None, "; ".join(errors)
    attempts.sort(key=lambda x: x[0])
    _, train_pred, test_pred, y_train_eval, y_test_eval, pack = attempts[0]
    pack["lookback_candidates"] = list(lookbacks)
    return train_pred, test_pred, y_train_eval, y_test_eval, pack, None

# ======================================================
# PLOTS
# ======================================================
def plot_actual_pred(pred_df, target):
    x = pd.to_datetime(pred_df["Time"], errors="coerce") if "Time" in pred_df.columns else np.arange(len(pred_df))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pred_df["Actual"], mode="lines", name="Actual", line=dict(color="#2563eb", width=2.0)))
    fig.add_trace(go.Scatter(x=x, y=pred_df["Predicted"], mode="lines", name="Predicted", line=dict(color="#ef4444", width=2.0)))
    fig.update_layout(xaxis_title="Time", yaxis_title=target_label(target), hovermode="x unified")
    return style_fig(fig, height=245, legend=True)


def plot_feature_importance(model, feature_cols, pca_pack=None):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(10)
    fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", text="Importance")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_color="#2563eb")
    fig.update_layout(xaxis_title="Importance", yaxis_title="")
    return style_fig(fig, height=245, legend=False)


def plot_gru_loss(history):
    fig = go.Figure()
    loss = history.get("loss", [])
    val = history.get("val_loss", [])
    fig.add_trace(go.Scatter(y=loss, x=list(range(1, len(loss)+1)), mode="lines", name="Training Loss", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(y=val, x=list(range(1, len(val)+1)), mode="lines", name="Validation Loss", line=dict(color="#7c3aed", width=2)))
    fig.update_layout(xaxis_title="Epochs", yaxis_title="Loss")
    return style_fig(fig, height=245, legend=True)


def plot_forecast_history(target, forecast_values, data, future_steps, best_model_name, history_days=7):
    working = data.copy().sort_values(time_col)
    valid = pd.DataFrame({"Time": working[time_col], "Value": pd.to_numeric(working[target], errors="coerce")}).dropna()
    if valid.empty:
        return go.Figure()
    hist_df = valid[valid["Time"] >= valid["Time"].max() - pd.Timedelta(days=history_days)].copy()
    if len(hist_df) < 20:
        hist_df = valid.tail(history_days * 24)
    step = valid["Time"].diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(hours=1)
    last_time = hist_df["Time"].iloc[-1]
    future_x = [last_time + step * (i + 1) for i in range(future_steps)]
    forecast_arr = np.asarray(forecast_values[:future_steps], dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["Time"], y=hist_df["Value"], mode="lines", name="Historical", line=dict(color="#2563eb", width=2.2)))
    # last 24h actual visually separated
    last24 = hist_df.tail(min(24, len(hist_df)))
    fig.add_trace(go.Scatter(x=last24["Time"], y=last24["Value"], mode="lines", name="Actual (Last 24h)", line=dict(color="#16a34a", width=2.2)))
    fig.add_trace(go.Scatter(x=future_x[:len(forecast_arr)], y=forecast_arr, mode="lines+markers", name="Forecast (Next 24h)", line=dict(color="#ef4444", width=2.4), marker=dict(size=5)))
    fig.add_vline(x=last_time, line_dash="dash", line_color="#0f172a")
    fig.update_layout(xaxis_title="Last 7 Days (History) → Next Forecast", yaxis_title=target_label(target), hovermode="x unified")
    return style_fig(fig, height=285, legend=True)

# ======================================================
# FORECASTING HELPERS
# ======================================================
def recursive_forecast_ml(model, data, target, dataset, future_steps=24):
    working = data.copy().sort_values(time_col).reset_index(drop=True)
    time_series = pd.to_datetime(working[time_col], errors="coerce").dropna().reset_index(drop=True)
    step = time_series.diff().median() if len(time_series) > 2 else pd.Timedelta(hours=1)
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(hours=1)

    preds = []
    for _ in range(future_steps):
        new_row = {c: np.nan for c in working.columns}
        new_row[time_col] = pd.to_datetime(working[time_col], errors="coerce").dropna().iloc[-1] + step
        # future exogenous variables are approximated using recent 24-hour mean
        for col in working.columns:
            if col not in [target, time_col]:
                s = pd.to_numeric(working[col], errors="coerce").dropna()
                if not s.empty:
                    new_row[col] = s.tail(24).mean()
        working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
        temp = add_time_features(working)
        temp = add_target_lag_features(temp, target)
        latest = temp.iloc[[-1]].copy()
        for c in dataset["original_feature_cols"]:
            if c not in latest.columns:
                latest[c] = 0
            latest[c] = pd.to_numeric(latest[c], errors="coerce")
        if dataset.get("pca_pack") is not None:
            X_latest = transform_latest_with_pca(latest, dataset["pca_pack"])
        else:
            X_latest = latest[dataset["tree_feature_cols"]].replace([np.inf, -np.inf], np.nan).fillna(0)
        pred = float(model.predict(X_latest)[0])
        working.loc[working.index[-1], target] = pred
        preds.append(pred)
    return np.asarray(preds)


def recursive_forecast_gru(pack, dataset, future_steps=24):
    model = pack["model"]
    x_scaler = pack["x_scaler"]
    y_scaler = pack["y_scaler"]
    lookback = int(pack.get("lookback", 24))
    X_context = pd.concat([dataset["X_train_gru"].tail(lookback), dataset["X_test_gru"]], axis=0)
    if len(X_context) < lookback:
        return np.array([])
    current_window = X_context.tail(lookback).copy().reset_index(drop=True)
    preds = []
    for _ in range(future_steps):
        current_s = x_scaler.transform(current_window)
        pred_s = model.predict(current_s.reshape((1, lookback, current_s.shape[1])), verbose=0)
        pred = float(y_scaler.inverse_transform(pred_s).ravel()[0])
        preds.append(pred)
        next_row = current_window.iloc[[-1]].copy()
        lag_cols = sorted([c for c in next_row.columns if "_lag_" in c], key=lambda x: int(x.split("_lag_")[-1]) if x.split("_lag_")[-1].isdigit() else 999)
        for c in lag_cols:
            if c.endswith("_lag_1"):
                next_row[c] = pred
            else:
                prev_lag = int(c.split("_lag_")[-1]) - 1 if c.split("_lag_")[-1].isdigit() else None
                source = c.rsplit("_lag_", 1)[0] + f"_lag_{prev_lag}" if prev_lag else None
                if source in next_row.columns:
                    next_row[c] = next_row[source].values[0]
        for c in [c for c in next_row.columns if "_roll" in c]:
            next_row[c] = np.mean(preds[-min(len(preds), 24):])
        current_window = pd.concat([current_window.iloc[1:], next_row], axis=0).reset_index(drop=True)
    return np.asarray(preds)

# ======================================================
# TRAINING PIPELINE
# ======================================================
def prepare_dataset_for_target(data, target, use_pca=True):
    modelling, base_cols, lag_cols, time_cols, original_feature_cols = build_target_dataset(data, target)
    train_df, test_df, X_train_orig, X_test_orig, y_train, y_test = split_time_df(modelling, original_feature_cols, target)

    X_train_gru = X_train_orig.copy()
    X_test_gru = X_test_orig.copy()

    pca_pack = None
    if use_pca:
        X_train_tree, X_test_tree, pca_pack = apply_pca_to_tree_inputs(train_df, test_df, base_cols, lag_cols, time_cols, target)
    else:
        X_train_tree, X_test_tree = X_train_orig.copy(), X_test_orig.copy()

    return {
        "modelling": modelling,
        "train_df": train_df,
        "test_df": test_df,
        "X_train_tree": X_train_tree,
        "X_test_tree": X_test_tree,
        "X_train_gru": X_train_gru,
        "X_test_gru": X_test_gru,
        "y_train": y_train,
        "y_test": y_test,
        "time_train": train_df[time_col].reset_index(drop=True),
        "time_test": test_df[time_col].reset_index(drop=True),
        "base_cols": base_cols,
        "lag_cols": lag_cols,
        "time_cols": time_cols,
        "original_feature_cols": original_feature_cols,
        "tree_feature_cols": list(X_train_tree.columns),
        "pca_pack": pca_pack,
    }


def train_models_for_selected_target(data, target, run_tree=True, run_gru=True, use_pca=True):
    dataset = prepare_dataset_for_target(data, target, use_pca=use_pca)
    results, fitted, test_preds, train_preds, hp = [], {}, {}, {}, {}

    Xtr_tree, Xte_tree = dataset["X_train_tree"], dataset["X_test_tree"]
    ytr, yte = dataset["y_train"], dataset["y_test"]

    if run_tree:
        rf, rf_params, rf_cv = train_rf_grid(Xtr_tree, ytr)
        for name, model, params, cv_rmse in [("Random Forest", rf, rf_params, rf_cv)]:
            trp, tep = model.predict(Xtr_tree), model.predict(Xte_tree)
            fitted[name] = model
            hp[name] = {"params": params, "cv_rmse": cv_rmse}
            tm = metrics_dict(yte, tep)
            results.append({"Model": name, **tm, "Features": Xtr_tree.shape[1], "Input Type": "PCA + Lag/Time" if dataset["pca_pack"] else "Original Features"})
            train_preds[name] = pd.DataFrame({"Time": dataset["time_train"], "Actual": ytr.reset_index(drop=True), "Predicted": trp})
            test_preds[name] = pd.DataFrame({"Time": dataset["time_test"], "Actual": yte.reset_index(drop=True), "Predicted": tep})

        if XGBOOST_AVAILABLE:
            xgb, xgb_params, xgb_cv = train_xgb_grid(Xtr_tree, ytr)
            trp, tep = xgb.predict(Xtr_tree), xgb.predict(Xte_tree)
            fitted["XGBoost"] = xgb
            hp["XGBoost"] = {"params": xgb_params, "cv_rmse": xgb_cv}
            tm = metrics_dict(yte, tep)
            results.append({"Model": "XGBoost", **tm, "Features": Xtr_tree.shape[1], "Input Type": "PCA + Lag/Time" if dataset["pca_pack"] else "Original Features"})
            train_preds["XGBoost"] = pd.DataFrame({"Time": dataset["time_train"], "Actual": ytr.reset_index(drop=True), "Predicted": trp})
            test_preds["XGBoost"] = pd.DataFrame({"Time": dataset["time_test"], "Actual": yte.reset_index(drop=True), "Predicted": tep})

    if run_gru:
        train_pred, test_pred, y_train_eval, y_test_eval, pack, err = train_gru_auto(
            dataset["X_train_gru"], dataset["X_test_gru"], ytr, yte, epochs=FAST_GRU_EPOCHS, lookbacks=FAST_GRU_LOOKBACKS
        )
        if err:
            st.warning(f"GRU skipped: {err}")
        else:
            fitted["GRU"] = pack
            hp["GRU"] = {"params": {"lookback": pack["lookback"], "hidden_units": "64 + 32", "dropout": "0.20 / 0.15", "learning_rate": 0.001, "epochs": pack["epochs_run"]}, "cv_rmse": np.nan}
            tm = metrics_dict(y_test_eval, test_pred)
            results.append({"Model": "GRU", **tm, "Features": dataset["X_train_gru"].shape[1], "Input Type": "Original Sequential Features"})
            train_time = dataset["time_train"].iloc[-len(y_train_eval):].reset_index(drop=True)
            train_preds["GRU"] = pd.DataFrame({"Time": train_time, "Actual": y_train_eval.reset_index(drop=True), "Predicted": train_pred})
            test_preds["GRU"] = pd.DataFrame({"Time": dataset["time_test"].iloc[:len(y_test_eval)].reset_index(drop=True), "Actual": y_test_eval.reset_index(drop=True), "Predicted": test_pred})

    results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False).reset_index(drop=True) if results else pd.DataFrame()
    return results_df, fitted, test_preds, train_preds, hp, dataset

# ======================================================
# SESSION STATE
# ======================================================
def reset_outputs():
    keys = ["mf_results", "mf_fitted", "mf_test_preds", "mf_train_preds", "mf_hp", "mf_dataset", "mf_forecasts", "mf_target"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

if st.session_state.get("mf_target") != selected_target:
    reset_outputs()
    st.session_state["mf_target"] = selected_target

# ======================================================
# SECTION 1: RF + XGBOOST
# ======================================================
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
h1, h2, h3 = st.columns([5.2, 1.25, 1.85], gap="small")
with h1:
    st.markdown('<div class="section-head"><div class="step-badge badge-blue">1</div><div><span class="section-title">TRAIN RANDOM FOREST & XGBOOST</span><span class="section-note">fast PCA components + lag/time features</span></div></div>', unsafe_allow_html=True)
with h2:
    st.markdown(f'<div class="select-target">Select Target: <span class="blue-text">{target_label(selected_target)}</span></div>', unsafe_allow_html=True)
    use_pca_for_tree = st.toggle("Use PCA", value=True, help="PCA is fitted only on the training split for RF/XGBoost.")
with h3:
    train_tree_clicked = st.button("Train RF + XGBoost", use_container_width=True)

if train_tree_clicked:
    with st.spinner("Fast training RF and XGBoost with PCA + lag/time features..."):
        results_df, fitted, test_preds, train_preds, hp, dataset = train_models_for_selected_target(
            df_work, selected_target, run_tree=True, run_gru=False, use_pca=use_pca_for_tree
        )
        st.session_state["mf_results"] = results_df
        st.session_state["mf_fitted"] = fitted
        st.session_state["mf_test_preds"] = test_preds
        st.session_state["mf_train_preds"] = train_preds
        st.session_state["mf_hp"] = hp
        st.session_state["mf_dataset"] = dataset
        st.session_state["mf_forecasts"] = {}

results_df = st.session_state.get("mf_results", pd.DataFrame())
fitted = st.session_state.get("mf_fitted", {})
test_preds = st.session_state.get("mf_test_preds", {})
train_preds = st.session_state.get("mf_train_preds", {})
hp = st.session_state.get("mf_hp", {})
dataset = st.session_state.get("mf_dataset")

r1c1, r1c2, r1c3, r1c4 = st.columns([1.15, 1.15, 1.55, 1.35], gap="small")
with r1c1:
    st.markdown('<div class="small-title">MODEL PERFORMANCE (TEST SET)</div>', unsafe_allow_html=True)
    if not results_df.empty:
        tree_table = results_df[results_df["Model"].isin(["Random Forest", "XGBoost"])][["Model", "R² Score", "RMSE", "MAE", "MAPE (%)", "Input Type"]].copy()
        for c in ["R² Score", "RMSE", "MAE", "MAPE (%)"]:
            tree_table[c] = tree_table[c].round(3 if c == "R² Score" else 2)
        st.dataframe(tree_table, use_container_width=True, hide_index=True, height=155)
        best_tree = tree_table.sort_values("R² Score", ascending=False).iloc[0] if not tree_table.empty else None
        if best_tree is not None:
            st.markdown(f'<div class="best-box">🏆 Best Tree Model: {best_tree["Model"]}</div>', unsafe_allow_html=True)
    else:
        st.info("Click Train RF + XGBoost.")
with r1c2:
    st.markdown('<div class="small-title">FEATURE / PCA IMPORTANCE</div>', unsafe_allow_html=True)
    if dataset and fitted:
        candidate = "XGBoost" if "XGBoost" in fitted else "Random Forest"
        fig = plot_feature_importance(fitted[candidate], dataset["tree_feature_cols"], dataset.get("pca_pack"))
        if fig:
            st.plotly_chart(fig, use_container_width=True, config=CONFIG)
        if dataset.get("pca_pack"):
            st.markdown(f'<div class="table-note">PCA applied: {dataset["pca_pack"]["n_components"]} components, {dataset["pca_pack"]["variance_explained"]:.1f}% variance.</div>', unsafe_allow_html=True)
    else:
        st.info("Feature importance appears after training.")
with r1c3:
    st.markdown('<div class="small-title">ACTUAL vs PREDICTED (BEST TREE MODEL)</div>', unsafe_allow_html=True)
    if not results_df.empty:
        tree_part = results_df[results_df["Model"].isin(["Random Forest", "XGBoost"])]
        if not tree_part.empty:
            best_tree_name = tree_part.sort_values("R² Score", ascending=False).iloc[0]["Model"]
            st.plotly_chart(plot_actual_pred(test_preds[best_tree_name], selected_target), use_container_width=True, config=CONFIG)
        else:
            st.info("No tree model result yet.")
    else:
        st.info("Actual vs predicted appears after training.")
with r1c4:
    st.markdown('<div class="small-title">HYPERPARAMETER TUNING (BEST)</div>', unsafe_allow_html=True)
    if hp:
        rows = []
        for model_name in ["Random Forest", "XGBoost"]:
            if model_name in hp:
                rows.append({"Model": model_name, "Best Parameters": "\n".join([f"{k}: {v}" for k, v in hp[model_name]["params"].items()]), "Best RMSE (CV)": fmt(hp[model_name]["cv_rmse"], 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=205)
    else:
        st.info("CV tuning summary appears after training.")
st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# SECTION 2: GRU
# ======================================================
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
g1, g2, g3 = st.columns([5.2, 1.25, 1.85], gap="small")
with g1:
    st.markdown('<div class="section-head"><div class="step-badge badge-purple">2</div><div><span class="section-title">TRAIN GRU (DEEP LEARNING)</span><span class="section-note">original sequential features, no PCA compression</span></div></div>', unsafe_allow_html=True)
with g2:
    st.markdown(f'<div class="select-target">Select Target: <span class="blue-text">{target_label(selected_target)}</span></div>', unsafe_allow_html=True)
with g3:
    train_gru_clicked = st.button("Train GRU", use_container_width=True)

if train_gru_clicked:
    with st.spinner("Training GRU with 24/48/72 hour sequence windows..."):
        if dataset is None:
            # Prepare dataset even if tree models were not trained.
            dataset = prepare_dataset_for_target(df_work, selected_target, use_pca=True)
            st.session_state["mf_dataset"] = dataset
        results_new, fitted_new, test_preds_new, train_preds_new, hp_new, dataset_new = train_models_for_selected_target(
            df_work, selected_target, run_tree=False, run_gru=True, use_pca=True
        )
        dataset = dataset_new
        old_results = st.session_state.get("mf_results", pd.DataFrame())
        if not old_results.empty:
            old_results = old_results[old_results["Model"] != "GRU"]
            st.session_state["mf_results"] = pd.concat([old_results, results_new], ignore_index=True).sort_values("R² Score", ascending=False).reset_index(drop=True)
        else:
            st.session_state["mf_results"] = results_new
        st.session_state.setdefault("mf_fitted", {}).update(fitted_new)
        st.session_state.setdefault("mf_test_preds", {}).update(test_preds_new)
        st.session_state.setdefault("mf_train_preds", {}).update(train_preds_new)
        st.session_state.setdefault("mf_hp", {}).update(hp_new)
        st.session_state["mf_dataset"] = dataset
        st.session_state["mf_forecasts"] = {}

results_df = st.session_state.get("mf_results", pd.DataFrame())
fitted = st.session_state.get("mf_fitted", {})
test_preds = st.session_state.get("mf_test_preds", {})
train_preds = st.session_state.get("mf_train_preds", {})
hp = st.session_state.get("mf_hp", {})
dataset = st.session_state.get("mf_dataset")

gc1, gc2, gc3, gc4 = st.columns([1.25, 1.15, 1.55, 1.3], gap="small")
with gc1:
    st.markdown('<div class="small-title">GRU SEQUENCE WINDOW COMPARISON</div>', unsafe_allow_html=True)
    if "GRU" in fitted:
        st.plotly_chart(plot_gru_loss(fitted["GRU"].get("history", {})), use_container_width=True, config=CONFIG)
        st.markdown(f'<div class="table-note">Best Window: {fitted["GRU"].get("lookback", "-")} Hours</div>', unsafe_allow_html=True)
    else:
        st.info("Click Train GRU.")
with gc2:
    st.markdown('<div class="small-title">GRU PERFORMANCE (TEST SET)</div>', unsafe_allow_html=True)
    if not results_df.empty and "GRU" in list(results_df["Model"]):
        row = results_df[results_df["Model"] == "GRU"].iloc[0]
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        small_metric("R² Score", fmt(row["R² Score"], 3), "purple-text")
        small_metric("RMSE", fmt(row["RMSE"], 2), "purple-text")
        small_metric("MAE", fmt(row["MAE"], 2), "purple-text")
        small_metric("MAPE (%)", fmt(row["MAPE (%)"], 2), "purple-text")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("GRU metrics appear after training.")
with gc3:
    st.markdown('<div class="small-title">ACTUAL vs PREDICTED (GRU)</div>', unsafe_allow_html=True)
    if "GRU" in test_preds:
        st.plotly_chart(plot_actual_pred(test_preds["GRU"], selected_target), use_container_width=True, config=CONFIG)
    else:
        st.info("GRU plot appears after training.")
with gc4:
    st.markdown('<div class="small-title">GRU CONFIGURATION (BEST)</div>', unsafe_allow_html=True)
    if "GRU" in hp:
        rows = [{"Item": k, "Value": v} for k, v in hp["GRU"]["params"].items()]
        rows.append({"Item": "Input Type", "Value": "Original sequential features"})
        rows.append({"Item": "PCA Used", "Value": "No - temporal structure preserved"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=230)
    else:
        st.info("GRU configuration appears after training.")
st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# SECTION 3: FORECAST
# ======================================================
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
f1, f2, f3 = st.columns([5.2, 1.25, 1.85], gap="small")
with f1:
    st.markdown(f'<div class="section-head"><div class="step-badge badge-green">3</div><div><span class="section-title">{future_steps}-HOUR FORECAST (BEST MODEL)</span><span class="section-note">forecast next hours using best test-set model</span></div></div>', unsafe_allow_html=True)
with f2:
    st.markdown(f'<div class="select-target">Select Target: <span class="blue-text">{target_label(selected_target)}</span></div>', unsafe_allow_html=True)
with f3:
    forecast_clicked = st.button("Generate Forecast", use_container_width=True)

if forecast_clicked:
    if results_df.empty or not fitted or dataset is None:
        st.warning("Please train at least one model before generating the forecast.")
    else:
        with st.spinner("Generating recursive forecast from the best model..."):
            forecasts = {}
            for model_name, model in fitted.items():
                try:
                    if model_name == "GRU":
                        forecasts[model_name] = recursive_forecast_gru(model, dataset, future_steps)
                    else:
                        forecasts[model_name] = recursive_forecast_ml(model, df_work, selected_target, dataset, future_steps)
                except Exception as exc:
                    st.warning(f"Forecast skipped for {model_name}: {exc}")
            st.session_state["mf_forecasts"] = forecasts

forecasts = st.session_state.get("mf_forecasts", {})
fc1, fc2, fc3, fc4 = st.columns([1.05, 2.7, 1.15, 1.45], gap="small")
with fc1:
    st.markdown('<div class="small-title">BEST MODEL SUMMARY</div>', unsafe_allow_html=True)
    if not results_df.empty:
        best = results_df.sort_values("R² Score", ascending=False).iloc[0]
        st.markdown(f'<div class="best-box best-green">{best["Model"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;font-weight:750;margin-top:10px;">R² Score (Test Set)</div><div class="green-text" style="font-size:20px;">{fmt(best["R² Score"],3)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;font-weight:750;margin-top:6px;">RMSE (Test Set)</div><div class="green-text" style="font-size:20px;">{fmt(best["RMSE"],2)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="table-note">Input: {best.get("Input Type", "-")}</div>', unsafe_allow_html=True)
    else:
        st.info("Train models first.")
with fc2:
    st.markdown(f'<div class="small-title">{future_steps}-HOUR FORECAST</div>', unsafe_allow_html=True)
    if forecasts and not results_df.empty:
        best_name = results_df.sort_values("R² Score", ascending=False).iloc[0]["Model"]
        if best_name not in forecasts:
            best_name = list(forecasts.keys())[0]
        st.plotly_chart(plot_forecast_history(selected_target, forecasts[best_name], df_work, future_steps, best_name), use_container_width=True, config=CONFIG)
    else:
        st.info("Click Generate Forecast after model training.")
with fc3:
    st.markdown('<div class="small-title">FORECAST EVALUATION</div>', unsafe_allow_html=True)
    if not results_df.empty:
        best = results_df.sort_values("R² Score", ascending=False).iloc[0]
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        small_metric("RMSE", fmt(best["RMSE"], 2), "green-text")
        small_metric("MAE", fmt(best["MAE"], 2), "green-text")
        small_metric("MAPE (%)", fmt(best["MAPE (%)"], 2), "green-text")
        small_metric("R² Score", fmt(best["R² Score"], 3), "green-text")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Evaluation appears after training.")
with fc4:
    st.markdown('<div class="small-title">FORECAST SUMMARY</div>', unsafe_allow_html=True)
    if forecasts and not results_df.empty:
        best = results_df.sort_values("R² Score", ascending=False).iloc[0]
        best_name = best["Model"] if best["Model"] in forecasts else list(forecasts.keys())[0]
        vals = pd.Series(forecasts[best_name]).dropna()
        peak = vals.max() if not vals.empty else np.nan
        avg = vals.mean() if not vals.empty else np.nan
        trend = "Slightly Increasing" if len(vals) > 1 and vals.iloc[-1] > vals.iloc[0] else "Slightly Decreasing"
        rows = [
            ("Forecast Horizon", f"Next {future_steps} Hours"),
            ("Best Model", best_name),
            ("PCA for RF/XGB", "Yes" if dataset and dataset.get("pca_pack") else "No"),
            ("Peak Forecast", f"{fmt(peak,2)} {pollutant_unit(selected_target)}"),
            ("Average Forecast", f"{fmt(avg,2)} {pollutant_unit(selected_target)}"),
            ("Trend", trend),
        ]
        for k, v in rows:
            st.markdown(f'<div style="font-size:12.5px;font-weight:750;margin-bottom:7px;">☑️ {k}: <span style="font-weight:850;color:#07184a;">{v}</span></div>', unsafe_allow_html=True)
    else:
        st.info("Forecast summary appears after generation.")
st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# BOTTOM FOOTER STRIP
# ======================================================
pca_components = dataset["pca_pack"]["n_components"] if dataset and dataset.get("pca_pack") else 0
pca_variance = dataset["pca_pack"]["variance_explained"] if dataset and dataset.get("pca_pack") else 0
st.markdown(
    f'<div class="footer-strip">ⓘ Target Parameter: <span class="blue-text">{target_label(selected_target)}</span> &nbsp; | &nbsp; Forecast Horizon: <span class="blue-text">{horizon_label}</span> &nbsp; | &nbsp; Date Range: {start_date} to {end_date} &nbsp; | &nbsp; PCA Components for RF/XGB: <span class="blue-text">{pca_components}</span> ({pca_variance:.1f}% variance)</div>',
    unsafe_allow_html=True,
)

render_footer()
