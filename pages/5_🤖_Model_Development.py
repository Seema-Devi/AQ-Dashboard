import gc
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except Exception:
    tf = None
    TENSORFLOW_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

from ui_components import load_full_ui, render_footer


# ======================================================
# PAGE SETUP
# ======================================================
st.write("")
load_full_ui()

px.defaults.template = "plotly_white"

config = {
    "displayModeBar": False,
    "scrollZoom": False
}

df = st.session_state.get("df_engineered")

if df is None:
    st.warning("⬅ Please complete Feature Engineering first.")
    render_footer()
    st.stop()

df = df.copy()

st.header("🤖 AI Model Development & Forecasting")


# ======================================================
# SAFE ERROR HANDLING
# ======================================================
def show_safe_error(title, err=None):
    st.error(title)
    if err is not None:
        with st.expander("Show technical error details"):
            st.code(traceback.format_exc())
    render_footer()
    st.stop()


# ======================================================
# CSS
# ======================================================
st.markdown("""
<style>
.kpi-card {
    padding: 18px;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.12);
}
.kpi-title {
    font-size: 14px;
    font-weight: 700;
}
.kpi-value {
    font-size: 27px;
    font-weight: 900;
    margin-top: 6px;
}
.blue-card { background: linear-gradient(135deg, #2563eb, #60a5fa); }
.green-card { background: linear-gradient(135deg, #059669, #34d399); }
.orange-card { background: linear-gradient(135deg, #ea580c, #fb923c); }
.purple-card { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
.red-card { background: linear-gradient(135deg, #dc2626, #f87171); }
</style>
""", unsafe_allow_html=True)


def kpi_card(col, title, value, css_class):
    col.markdown(f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ======================================================
# PAGE PURPOSE
# ======================================================
st.markdown("""
<div style="
background:white;
padding:20px;
border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08);
margin-bottom:20px;
">
This page trains, optimises and compares selected forecasting models for AQI, PM2.5 and NO2.
The main results are displayed first, followed by model comparison, residual analysis, and explainability.
</div>
""", unsafe_allow_html=True)


# ======================================================
# BASIC CHECKS
# ======================================================
if "DATETIME_HOUR" not in df.columns:
    show_safe_error("DATETIME_HOUR column is missing. Please rerun Feature Engineering.")

df["DATETIME_HOUR"] = pd.to_datetime(df["DATETIME_HOUR"], errors="coerce")
df = df.dropna(subset=["DATETIME_HOUR"]).sort_values("DATETIME_HOUR").reset_index(drop=True)

available_targets = [c for c in ["AQI", "PM2.5", "NO2"] if c in df.columns]

if not available_targets:
    show_safe_error("AQI, PM2.5 or NO2 target columns were not found.")


# ======================================================
# 1. FORECAST CONFIGURATION
# ======================================================
st.subheader("1️⃣ Forecast Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    target_var = st.selectbox(
        "Forecast Target",
        available_targets,
        index=0
    )

with col2:
    mode_choice = st.radio(
        "Input Mode",
        ["Multivariate", "Univariate"],
        horizontal=True
    )

with col3:
    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        [1, 6, 12, 24],
        index=3,
        format_func=lambda x: f"{x} hours"
    )

selected_models = st.multiselect(
    "Select Model(s)",
    ["Random Forest", "XGBoost", "GRU"],
    default=["XGBoost"]
)

if not selected_models:
    st.warning("Please select at least one model.")
    render_footer()
    st.stop()

if len(selected_models) > 1:
    st.warning(f"You selected {len(selected_models)} models. Training may take longer.")
else:
    st.info(f"Training selected model: {selected_models[0]}")


# ======================================================
# 2. OPTIMISATION SETTINGS
# ======================================================
st.subheader("2️⃣ Model Optimisation Settings")

col1, col2, col3, col4 = st.columns(4)

with col1:
    cv_folds = st.slider(
        "TimeSeriesSplit Folds",
        min_value=2,
        max_value=5,
        value=3,
        step=1
    )

with col2:
    n_iter_search = st.slider(
        "Search Iterations",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )

with col3:
    test_size_pct = st.slider(
        "Test Size (%)",
        min_value=10,
        max_value=30,
        value=20,
        step=5
    )

with col4:
    gru_sequence_length = st.selectbox(
        "GRU Sequence Length",
        [12, 24, 48],
        index=1
    )

enable_shap = st.checkbox(
    "Run SHAP after training",
    value=False,
    help="Recommended only for Random Forest and XGBoost. GRU is not included."
)

run_model = st.button("🚀 Train and Forecast")

if not run_model:
    st.info("Select configuration above, then click **Train and Forecast**.")
    render_footer()
    st.stop()


# ======================================================
# DATA PREPARATION
# ======================================================
def prepare_tree_data(data, target, mode):
    df_m = data.copy()

    if target not in df_m.columns:
        show_safe_error(f"{target} column is missing.")

    df_m["LABEL"] = df_m[target].shift(-1)

    time_features = [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "is_weekend"
    ]

    target_features = [
        target,
        f"{target}_lag_1",
        f"{target}_lag_2",
        f"{target}_lag_3",
        f"{target}_lag_6",
        f"{target}_lag_12",
        f"{target}_lag_24",
        f"{target}_roll_mean_6",
        f"{target}_roll_std_6",
        f"{target}_roll_mean_24",
        f"{target}_roll_std_24"
    ]

    if mode == "Univariate":
        final_features = target_features + time_features
        final_features = [f for f in final_features if f in df_m.columns]
    else:
        exclude_cols = ["DATETIME_HOUR", "LABEL"]
        numeric_cols = df_m.select_dtypes(include=np.number).columns.tolist()
        final_features = [c for c in numeric_cols if c not in exclude_cols]

    if not final_features:
        show_safe_error("No valid model features found. Please rerun Feature Engineering.")

    df_ready = df_m.dropna(subset=["LABEL"] + final_features).reset_index(drop=True)

    X = df_ready[final_features]
    y = df_ready["LABEL"]

    return df_ready, X, y, final_features


def prepare_gru_data(data, target, mode, sequence_length):
    df_m, X, y, features = prepare_tree_data(data, target, mode)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_seq = []
    y_seq = []
    seq_times = []

    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])
        seq_times.append(df_m["DATETIME_HOUR"].iloc[i + sequence_length])

    return (
        df_m,
        X,
        y,
        features,
        np.array(X_seq),
        np.array(y_seq),
        scaler_x,
        scaler_y,
        seq_times
    )


df_model, X, y, feat_cols = prepare_tree_data(df, target_var, mode_choice)

if len(df_model) < 150:
    show_safe_error("Not enough valid rows for reliable modelling. At least 150 rows are recommended.")

split_idx = int(len(df_model) * (1 - test_size_pct / 100))

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

test_times = df_model["DATETIME_HOUR"].iloc[split_idx:].reset_index(drop=True)

n_splits = min(cv_folds, max(2, len(X_train) // 80))
tscv = TimeSeriesSplit(n_splits=n_splits)

st.info(
    f"Chronological split: {100-test_size_pct}% train / {test_size_pct}% test. "
    f"Optimisation uses TimeSeriesSplit with {n_splits} folds and {n_iter_search} search iterations."
)


# ======================================================
# METRICS AND HELPERS
# ======================================================
def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def build_next_feature_row(last_row, target, predicted_value, history_before_prediction, next_time, feat_cols):
    new_row = last_row.copy()

    if target in new_row:
        new_row[target] = predicted_value

    if "hour_sin" in new_row:
        new_row["hour_sin"] = np.sin(2 * np.pi * next_time.hour / 24)

    if "hour_cos" in new_row:
        new_row["hour_cos"] = np.cos(2 * np.pi * next_time.hour / 24)

    if "day_sin" in new_row:
        new_row["day_sin"] = np.sin(2 * np.pi * next_time.dayofweek / 7)

    if "day_cos" in new_row:
        new_row["day_cos"] = np.cos(2 * np.pi * next_time.dayofweek / 7)

    if "month_sin" in new_row:
        new_row["month_sin"] = np.sin(2 * np.pi * next_time.month / 12)

    if "month_cos" in new_row:
        new_row["month_cos"] = np.cos(2 * np.pi * next_time.month / 12)

    if "is_weekend" in new_row:
        new_row["is_weekend"] = int(next_time.dayofweek in [5, 6])

    for lag in [1, 2, 3, 6, 12, 24]:
        lag_col = f"{target}_lag_{lag}"
        if lag_col in new_row and len(history_before_prediction) >= lag:
            new_row[lag_col] = history_before_prediction[-lag]

    if f"{target}_roll_mean_6" in new_row and len(history_before_prediction) >= 6:
        new_row[f"{target}_roll_mean_6"] = np.mean(history_before_prediction[-6:])

    if f"{target}_roll_std_6" in new_row and len(history_before_prediction) >= 6:
        new_row[f"{target}_roll_std_6"] = np.std(history_before_prediction[-6:])

    if f"{target}_roll_mean_24" in new_row and len(history_before_prediction) >= 24:
        new_row[f"{target}_roll_mean_24"] = np.mean(history_before_prediction[-24:])

    if f"{target}_roll_std_24" in new_row and len(history_before_prediction) >= 24:
        new_row[f"{target}_roll_std_24"] = np.std(history_before_prediction[-24:])

    return new_row[feat_cols]


def recursive_forecast_tree(model, df_model, X, target, feat_cols, steps):
    future_preds = []
    future_times = []

    current_row = X.iloc[-1].copy()
    current_time = df_model["DATETIME_HOUR"].iloc[-1]
    history = df_model[target].tolist()

    for _ in range(steps):
        pred = model.predict(pd.DataFrame([current_row], columns=feat_cols))[0]

        next_time = current_time + pd.Timedelta(hours=1)

        future_times.append(next_time)
        future_preds.append(pred)

        history_before_prediction = history.copy()
        history.append(pred)

        current_row = build_next_feature_row(
            current_row,
            target,
            pred,
            history_before_prediction,
            next_time,
            feat_cols
        )

        current_time = next_time

    return pd.Series(future_preds, index=future_times)


def create_gru_model(time_steps, n_features):
    model = Sequential([
        GRU(48, input_shape=(time_steps, n_features)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def recursive_forecast_gru(model, scaler_x, scaler_y, df_model, X, target, feat_cols, steps, sequence_length):
    future_preds = []
    future_times = []

    feature_window = X.tail(sequence_length).copy()
    current_row = X.iloc[-1].copy()
    current_time = df_model["DATETIME_HOUR"].iloc[-1]
    history = df_model[target].tolist()

    for _ in range(steps):
        X_scaled = scaler_x.transform(feature_window)
        X_scaled = X_scaled.reshape(1, sequence_length, len(feat_cols))

        pred_scaled = model.predict(X_scaled, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]

        next_time = current_time + pd.Timedelta(hours=1)

        future_times.append(next_time)
        future_preds.append(pred)

        history_before_prediction = history.copy()
        history.append(pred)

        current_row = build_next_feature_row(
            current_row,
            target,
            pred,
            history_before_prediction,
            next_time,
            feat_cols
        )

        feature_window = pd.concat(
            [feature_window.iloc[1:], pd.DataFrame([current_row], columns=feat_cols)],
            ignore_index=True
        )

        current_time = next_time

    return pd.Series(future_preds, index=future_times)


# ======================================================
# MODEL TRAINING
# ======================================================
results = {}
baseline_results = {}
cv_results = {}
val_preds = {}
actual_series = {}
actual_times = {}
future_preds = {}
importances = {}
best_params_store = {}
model_status = {}
trained_models = {}

models_to_run = selected_models

progress_bar = st.progress(0)
total_models = len(models_to_run)

for idx, model_name in enumerate(models_to_run, start=1):

    with st.spinner(f"Training {model_name}..."):

        try:
            # ==================================================
            # RANDOM FOREST
            # ==================================================
            if model_name == "Random Forest":

                baseline_model = RandomForestRegressor(
                    n_estimators=80,
                    random_state=42,
                    n_jobs=-1
                )

                baseline_model.fit(X_train, y_train)
                base_pred = baseline_model.predict(X_test)
                baseline_results[model_name] = calculate_metrics(y_test, base_pred)

                grid = {
                    "n_estimators": [80, 120],
                    "max_depth": [8, 12, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }

                search = RandomizedSearchCV(
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    grid,
                    n_iter=n_iter_search,
                    cv=tscv,
                    random_state=42,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1
                )

                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                cv_score = cross_val_score(
                    best_model,
                    X_train,
                    y_train,
                    cv=tscv,
                    scoring="neg_root_mean_squared_error"
                )

                pred = best_model.predict(X_test)
                forecast_series = recursive_forecast_tree(
                    best_model,
                    df_model,
                    X,
                    target_var,
                    feat_cols,
                    forecast_horizon
                )

                results[model_name] = calculate_metrics(y_test, pred)
                cv_results[model_name] = -cv_score.mean()
                val_preds[model_name] = pd.Series(pred)
                actual_series[model_name] = pd.Series(y_test.values)
                actual_times[model_name] = test_times
                future_preds[model_name] = forecast_series
                importances[model_name] = best_model.feature_importances_
                best_params_store[model_name] = search.best_params_
                model_status[model_name] = "Success"
                trained_models[model_name] = best_model

            # ==================================================
            # XGBOOST
            # ==================================================
            elif model_name == "XGBoost":

                if XGBRegressor is None:
                    model_status[model_name] = "Skipped: XGBoost not installed"
                    st.warning("XGBoost is not installed.")
                    progress_bar.progress(idx / total_models)
                    continue

                baseline_model = XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror",
                    n_jobs=1
                )

                baseline_model.fit(X_train, y_train)
                base_pred = baseline_model.predict(X_test)
                baseline_results[model_name] = calculate_metrics(y_test, base_pred)

                grid = {
                    "n_estimators": [80, 120],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                }

                search = RandomizedSearchCV(
                    XGBRegressor(
                        random_state=42,
                        objective="reg:squarederror",
                        n_jobs=1
                    ),
                    grid,
                    n_iter=n_iter_search,
                    cv=tscv,
                    random_state=42,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=1
                )

                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                cv_score = cross_val_score(
                    best_model,
                    X_train,
                    y_train,
                    cv=tscv,
                    scoring="neg_root_mean_squared_error"
                )

                pred = best_model.predict(X_test)
                forecast_series = recursive_forecast_tree(
                    best_model,
                    df_model,
                    X,
                    target_var,
                    feat_cols,
                    forecast_horizon
                )

                results[model_name] = calculate_metrics(y_test, pred)
                cv_results[model_name] = -cv_score.mean()
                val_preds[model_name] = pd.Series(pred)
                actual_series[model_name] = pd.Series(y_test.values)
                actual_times[model_name] = test_times
                future_preds[model_name] = forecast_series
                importances[model_name] = best_model.feature_importances_
                best_params_store[model_name] = search.best_params_
                model_status[model_name] = "Success"
                trained_models[model_name] = best_model

            # ==================================================
            # GRU
            # ==================================================
            elif model_name == "GRU":

                if not TENSORFLOW_AVAILABLE:
                    model_status[model_name] = "Skipped: TensorFlow not installed"
                    st.warning("GRU is unavailable because TensorFlow is not installed.")
                    progress_bar.progress(idx / total_models)
                    continue

                tf.keras.backend.clear_session()
                gc.collect()

                (
                    df_gru,
                    X_gru_raw,
                    y_gru_raw,
                    gru_features,
                    X_seq,
                    y_seq,
                    scaler_x,
                    scaler_y,
                    seq_times
                ) = prepare_gru_data(
                    df,
                    target_var,
                    mode_choice,
                    gru_sequence_length
                )

                if len(X_seq) < 100:
                    model_status[model_name] = "Skipped: Not enough sequence data"
                    st.warning("Not enough data for GRU sequence training.")
                    progress_bar.progress(idx / total_models)
                    continue

                split_seq_idx = int(len(X_seq) * (1 - test_size_pct / 100))

                X_train_seq = X_seq[:split_seq_idx]
                X_test_seq = X_seq[split_seq_idx:]

                y_train_seq = y_seq[:split_seq_idx]
                y_test_seq = y_seq[split_seq_idx:]

                seq_test_times = pd.Series(seq_times[split_seq_idx:]).reset_index(drop=True)

                baseline_gru = Sequential([
                    GRU(24, input_shape=(gru_sequence_length, len(gru_features))),
                    Dense(1)
                ])

                baseline_gru.compile(optimizer="adam", loss="mse")

                baseline_gru.fit(
                    X_train_seq,
                    y_train_seq,
                    epochs=5,
                    batch_size=64,
                    verbose=0
                )

                base_scaled = baseline_gru.predict(X_test_seq, verbose=0)
                base_pred = scaler_y.inverse_transform(base_scaled).flatten()
                y_test_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

                baseline_results[model_name] = calculate_metrics(y_test_actual, base_pred)

                tf.keras.backend.clear_session()
                gc.collect()

                gru_model = create_gru_model(
                    gru_sequence_length,
                    len(gru_features)
                )

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True
                )

                history = gru_model.fit(
                    X_train_seq,
                    y_train_seq,
                    validation_split=0.2,
                    epochs=15,
                    batch_size=64,
                    callbacks=[early_stop],
                    verbose=0
                )

                pred_scaled = gru_model.predict(X_test_seq, verbose=0)
                pred = scaler_y.inverse_transform(pred_scaled).flatten()

                forecast_series = recursive_forecast_gru(
                    gru_model,
                    scaler_x,
                    scaler_y,
                    df_gru,
                    X_gru_raw,
                    target_var,
                    gru_features,
                    forecast_horizon,
                    gru_sequence_length
                )

                results[model_name] = calculate_metrics(y_test_actual, pred)
                cv_results[model_name] = np.sqrt(min(history.history["val_loss"]))
                val_preds[model_name] = pd.Series(pred)
                actual_series[model_name] = pd.Series(y_test_actual)
                actual_times[model_name] = seq_test_times
                future_preds[model_name] = forecast_series
                best_params_store[model_name] = {
                    "sequence_length": gru_sequence_length,
                    "epochs_used": len(history.history["loss"]),
                    "batch_size": 64,
                    "early_stopping": True,
                    "patience": 2,
                    "gru_units": [48],
                    "dropout": 0.2
                }
                model_status[model_name] = "Success"
                trained_models[model_name] = gru_model

                gc.collect()

        except Exception:
            model_status[model_name] = "Failed"
            st.warning(f"{model_name} could not be completed.")
            with st.expander(f"Show {model_name} error details"):
                st.code(traceback.format_exc())

    progress_bar.progress(idx / total_models)


if not results:
    show_safe_error("No model was successfully trained.")


# ======================================================
# RESULT OBJECTS
# ======================================================
best_model_name = min(results, key=lambda x: results[x]["RMSE"])
best_metrics = results[best_model_name]

summary_rows = []

for model_name in results:
    base_rmse = baseline_results.get(model_name, {}).get("RMSE", np.nan)
    opt_rmse = results[model_name]["RMSE"]

    improvement = base_rmse - opt_rmse if not pd.isna(base_rmse) else np.nan
    improvement_pct = (improvement / base_rmse * 100) if base_rmse and not pd.isna(base_rmse) else np.nan

    summary_rows.append({
        "Model": model_name,
        "Status": model_status.get(model_name, "Success"),
        "Baseline RMSE": base_rmse,
        "Optimised RMSE": opt_rmse,
        "Improvement": improvement,
        "Improvement %": improvement_pct,
        "MAE": results[model_name]["MAE"],
        "R²": results[model_name]["R2"],
        "CV / Validation RMSE": cv_results.get(model_name, np.nan)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("Optimised RMSE").reset_index(drop=True)
summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))

plot_df = pd.DataFrame({
    "Time": actual_times[best_model_name],
    "Actual": actual_series[best_model_name],
    "Predicted": val_preds[best_model_name]
}).dropna()

plot_df["Residual"] = plot_df["Actual"] - plot_df["Predicted"]

forecast_series = future_preds[best_model_name]

forecast_table = pd.DataFrame({
    "Forecast Time": forecast_series.index,
    f"Forecasted {target_var}": forecast_series.values
})


# ======================================================
# 3. FORECAST RESULTS SUMMARY
# ======================================================
st.markdown("---")
st.subheader("3️⃣ Forecast Results Summary")

c1, c2, c3, c4 = st.columns(4)

kpi_card(c1, "Best Model", best_model_name, "blue-card")
kpi_card(c2, "RMSE", f"{best_metrics['RMSE']:.3f}", "green-card")
kpi_card(c3, "MAE", f"{best_metrics['MAE']:.3f}", "orange-card")
kpi_card(c4, "R²", f"{best_metrics['R2']:.3f}", "purple-card")

st.success(
    f"Best model selected: {best_model_name}. "
    f"It achieved the lowest optimised RMSE of {best_metrics['RMSE']:.3f}."
)


# ======================================================
# 4. ACTUAL VS PREDICTED
# ======================================================
st.markdown("---")
st.subheader("4️⃣ Actual vs Predicted")

plot_recent = plot_df.tail(150)

plot_long = plot_recent.melt(
    id_vars="Time",
    value_vars=["Actual", "Predicted"],
    var_name="Series",
    value_name=target_var
)

fig = px.line(
    plot_long,
    x="Time",
    y=target_var,
    color="Series",
    title=f"Actual vs Predicted {target_var}"
)

fig.update_layout(height=520)

st.plotly_chart(fig, use_container_width=True, config=config)

st.info("""
Insight:
A good model should follow the actual trend closely.
Large gaps between actual and predicted lines show periods where the model struggled.
""")


# ======================================================
# 5. NEXT FORECAST
# ======================================================
st.markdown("---")
st.subheader(f"5️⃣ Next {forecast_horizon}-Hour Forecast")

history_df = df_model[["DATETIME_HOUR", target_var]].dropna().tail(48)

history_plot = pd.DataFrame({
    "Time": history_df["DATETIME_HOUR"],
    "Value": history_df[target_var],
    "Series": "Recent History"
})

forecast_plot = pd.DataFrame({
    "Time": forecast_series.index,
    "Value": forecast_series.values,
    "Series": f"{best_model_name} Forecast"
})

forecast_combined = pd.concat([history_plot, forecast_plot], ignore_index=True)

fig = px.line(
    forecast_combined,
    x="Time",
    y="Value",
    color="Series",
    markers=True,
    title=f"Next {forecast_horizon}-Hour Forecast for {target_var}"
)

fig.update_layout(height=520)

st.plotly_chart(fig, use_container_width=True, config=config)

with st.expander("📋 Open Forecast Values Table"):
    st.dataframe(forecast_table.round(3), use_container_width=True, hide_index=True)


# ======================================================
# 6. MODEL COMPARISON MATRIX
# ======================================================
st.markdown("---")
st.subheader("6️⃣ Model Comparison Table")

with st.expander("📋 Open Model Comparison Table"):
    st.dataframe(summary_df.round(3), use_container_width=True, hide_index=True)

with st.expander("📋 Open Model Parameters"):
    for model_name, params in best_params_store.items():
        st.write(f"**{model_name}**")
        st.json(params)



# ======================================================
# 7. FEATURE IMPORTANCE / SHAP
# ======================================================
st.markdown("---")
st.subheader("7️⃣ Feature Importance and SHAP Explainability")


if enable_shap and SHAP_AVAILABLE and best_model_name in ["Random Forest", "XGBoost"]:

        try:
            shap_sample = X_test.sample(min(150, len(X_test)), random_state=42)

            explainer = shap.TreeExplainer(trained_models[best_model_name])
            shap_values = explainer.shap_values(shap_sample)

            shap_mean = np.abs(shap_values).mean(axis=0)

            shap_df = pd.DataFrame({
                "Feature": shap_sample.columns,
                "Mean |SHAP Value|": shap_mean
            }).sort_values("Mean |SHAP Value|", ascending=False).head(15)

            fig = px.bar(
                shap_df,
                x="Mean |SHAP Value|",
                y="Feature",
                orientation="h",
                color="Mean |SHAP Value|",
                color_continuous_scale="Oranges",
                title=f"SHAP Feature Impact for {best_model_name}"
            )

            fig.update_layout(height=560, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, config=config)

            with st.expander("📋 Open SHAP Summary Table"):
                st.dataframe(shap_df.round(4), use_container_width=True, hide_index=True)

        except Exception:
            st.warning("SHAP explanation could not be generated for this run.")

elif best_model_name == "GRU":
    st.info(
        "SHAP is not generated for GRU in this dashboard because deep-learning SHAP is slower and less stable in Streamlit. "
        "GRU is evaluated using RMSE, MAE, R², validation loss and actual-vs-predicted performance."
    )

else:
    st.info("Feature importance is not available for this model.")


# ======================================================
# 9. FORECAST INTERPRETATION
# ======================================================
st.markdown("---")
st.subheader("8️⃣ Forecast Interpretation")

peak_value = forecast_series.max()
mean_value = forecast_series.mean()

if target_var == "AQI":
    if peak_value <= 50:
        st.success(f"🍃 Good forecast condition. Peak predicted AQI: {peak_value:.1f}")
    elif peak_value <= 100:
        st.warning(f"⚠️ Moderate forecast condition. Peak predicted AQI: {peak_value:.1f}")
    else:
        st.error(f"🚨 High AQI forecast. Peak predicted AQI: {peak_value:.1f}")
else:
    st.info(f"Forecast summary for {target_var}: average {mean_value:.2f}, peak {peak_value:.2f}")

with st.expander("How was the best model selected?"):
    st.write(
        "The best model was selected using the lowest optimised RMSE on the chronological test set. "
        "MAE, R², baseline improvement and validation RMSE were used as supporting evidence."
    )

with st.expander("Why use TimeSeriesSplit?"):
    st.write(
        "TimeSeriesSplit respects chronological order and avoids using future data during training."
    )


render_footer()
