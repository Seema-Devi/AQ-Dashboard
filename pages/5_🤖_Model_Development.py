import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
except:
    XGBRegressor = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from ui_components import load_full_ui, render_footer


# ======================================================
# SETUP PAGE
# ======================================================
st.write("")
load_full_ui()

df = st.session_state.get("df_engineered")

if df is None:
    st.warning("""
    ⬅ Please upload the dataset and do feature engineering.

   App Workflow:

1️⃣ Upload datasets from the sidebar  
2️⃣ Explore the data in Data Visualisation  
3️⃣ Clean and preprocess the dataset  
4️⃣ Analyse trends and patterns in EDA  
5️⃣ Apply Feature Engineering  
6️⃣ Train models and view AQI Forecasting
""")
    render_footer()
    st.stop()

st.header("🤖 AI Model Development")
st.info("👋 **Welcome to the Auckland Council AQI Forecasting AI Models**")
# ======================================================
# SIDEBAR SELECTIONS
# ======================================================
target_var = st.session_state.get("selected_target", "AQI")
model_choice = st.session_state.get("selected_model", "Random Forest")
run_btn = st.session_state.get("run_prediction", False)

mode_choice = st.radio(
    "Select Forecasting Mode",
    ["Multivariate", "Univariate"],
    horizontal=True
)


# ======================================================
# WELCOME PANEL
# ======================================================
if not run_btn:
    
    st.markdown("Please select your **Target Variable** and **Model** from the sidebar and click **Run Forecast**.")
    st.stop()


# ======================================================
# DATA PREPARATION - SAME FEATURE LOGIC AS REFERENCE CODE
# ======================================================
def prepare_data(data, target, mode):
    df_m = data.copy()

    if "DATETIME_HOUR" not in df_m.columns:
        st.error("DATETIME_HOUR column is missing. Please run Feature Engineering again.")
        render_footer()
        st.stop()

    df_m["DATETIME_HOUR"] = pd.to_datetime(df_m["DATETIME_HOUR"], errors="coerce")
    df_m = df_m.dropna(subset=["DATETIME_HOUR"]).sort_values("DATETIME_HOUR").reset_index(drop=True)

    if target not in df_m.columns:
        st.error(f"{target} column is missing from the engineered dataset.")
        render_footer()
        st.stop()

    # Target is next hour value: t+1
    df_m["LABEL"] = df_m[target].shift(-1)

    time_features = [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos"
    ]

    target_time_series_features = [
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
        final_features = target_time_series_features + time_features
        final_features = [f for f in final_features if f in df_m.columns]

    else:
        exclude_cols = ["DATETIME_HOUR", "LABEL", "season"]

        numeric_cols = df_m.select_dtypes(include=np.number).columns.tolist()
        final_features = [c for c in numeric_cols if c not in exclude_cols]

    if len(final_features) == 0:
        st.error("No valid model features found. Please run Feature Engineering again.")
        render_footer()
        st.stop()

    df_ready = df_m.dropna(subset=["LABEL"] + final_features).reset_index(drop=True)

    return df_ready, df_ready[final_features], df_ready["LABEL"], final_features


df_model, X, y, feat_cols = prepare_data(df, target_var, mode_choice)

if len(df_model) < 100:
    st.error("Not enough rows for reliable model training after feature engineering.")
    render_footer()
    st.stop()

split_idx = int(len(df_model) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]


# ======================================================
# FORECAST HELPER FUNCTIONS
# ======================================================
def update_time_features(row, dt):
    if "hour_sin" in row:
        row["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)

    if "hour_cos" in row:
        row["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)

    if "weekday_sin" in row:
        row["weekday_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)

    if "weekday_cos" in row:
        row["weekday_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

    return row


def build_next_feature_row(last_row, target, predicted_value, history_before_prediction, next_time, feat_cols):
    new_row = last_row.copy()

    if target in new_row:
        new_row[target] = predicted_value

    new_row = update_time_features(new_row, next_time)

    lag_steps = [1, 2, 3, 6, 12, 24]

    for lag in lag_steps:
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


def recursive_forecast_tree(model, df_model, X, target, feat_cols, steps=24):
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
            last_row=current_row,
            target=target,
            predicted_value=pred,
            history_before_prediction=history_before_prediction,
            next_time=next_time,
            feat_cols=feat_cols
        )

        current_time = next_time

    return pd.Series(future_preds, index=future_times)


def make_lstm_sequences(X_values, y_values, time_steps=24):
    X_seq, y_seq = [], []

    for i in range(len(X_values) - time_steps):
        X_seq.append(X_values[i:i + time_steps])
        y_seq.append(y_values[i + time_steps])

    return np.array(X_seq), np.array(y_seq)


def recursive_forecast_lstm(model, scaler_x, scaler_y, df_model, X, target, feat_cols, steps=24, time_steps=24):
    future_preds = []
    future_times = []

    feature_window = X.tail(time_steps).copy()
    current_row = X.iloc[-1].copy()
    current_time = df_model["DATETIME_HOUR"].iloc[-1]
    history = df_model[target].tolist()

    for _ in range(steps):
        X_window_scaled = scaler_x.transform(feature_window)
        X_window_scaled = X_window_scaled.reshape(1, time_steps, len(feat_cols))

        pred_scaled = model.predict(X_window_scaled, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]

        next_time = current_time + pd.Timedelta(hours=1)

        future_times.append(next_time)
        future_preds.append(pred)

        history_before_prediction = history.copy()
        history.append(pred)

        current_row = build_next_feature_row(
            last_row=current_row,
            target=target,
            predicted_value=pred,
            history_before_prediction=history_before_prediction,
            next_time=next_time,
            feat_cols=feat_cols
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
val_preds = {}
future_preds = {}
importances = {}

models_to_run = ["Random Forest", "XGBoost", "LSTM"] if model_choice == "Compare All" else [model_choice]

colors = {
    "Random Forest": "#1f77b4",
    "XGBoost": "#2ca02c",
    "LSTM": "#d62728"
}

tscv = TimeSeriesSplit(n_splits=3)

for m in models_to_run:
    with st.spinner(f"Optimising {m} with Automated Tuning..."):

        if m == "Random Forest":
            grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [8, 12, 16, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2, 3]
            }

            search = RandomizedSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                grid,
                n_iter=5,
                cv=tscv,
                random_state=42,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            best_m = search.best_estimator_

            p = best_m.predict(X_test)
            f_series = recursive_forecast_tree(best_m, df_model, X, target_var, feat_cols)

            results[m] = {
                "MAE": mean_absolute_error(y_test, p),
                "RMSE": np.sqrt(mean_squared_error(y_test, p)),
                "R2": r2_score(y_test, p)
            }

            val_preds[m] = pd.Series(p, index=y_test.index)
            future_preds[m] = f_series
            importances[m] = best_m.feature_importances_

            st.caption(f"✅ RF Best Params: {search.best_params_}")

        elif m == "XGBoost":
            if XGBRegressor is None:
                st.warning("XGBoost is not installed. Please install xgboost to use this model.")
                continue

            grid = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }

            search = RandomizedSearchCV(
                XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror"
                ),
                grid,
                n_iter=5,
                cv=tscv,
                random_state=42,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            best_m = search.best_estimator_

            p = best_m.predict(X_test)
            f_series = recursive_forecast_tree(best_m, df_model, X, target_var, feat_cols)

            results[m] = {
                "MAE": mean_absolute_error(y_test, p),
                "RMSE": np.sqrt(mean_squared_error(y_test, p)),
                "R2": r2_score(y_test, p)
            }

            val_preds[m] = pd.Series(p, index=y_test.index)
            future_preds[m] = f_series
            importances[m] = best_m.feature_importances_

            st.caption(f"✅ XGB Best Params: {search.best_params_}")

        elif m == "LSTM":
            if not TENSORFLOW_AVAILABLE:
                st.warning("LSTM model is not available because TensorFlow is not installed on Streamlit Cloud.")
                continue

            time_steps = 24

            if len(X_train) <= time_steps or len(X_test) <= time_steps:
                st.warning("Not enough data for LSTM 24-step sequence training.")
                continue

            sc_x = MinMaxScaler()
            sc_y = MinMaxScaler()

            X_train_scaled = sc_x.fit_transform(X_train)
            X_test_scaled = sc_x.transform(X_test)

            y_train_scaled = sc_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

            X_train_seq, y_train_seq = make_lstm_sequences(
                X_train_scaled,
                y_train_scaled,
                time_steps
            )

            X_test_seq, y_test_seq = make_lstm_sequences(
                X_test_scaled,
                y_test.values,
                time_steps
            )

            lstm_m = Sequential([
                LSTM(64, return_sequences=False, input_shape=(time_steps, len(feat_cols))),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1)
            ])

            lstm_m.compile(optimizer="adam", loss="mse")

            lstm_m.fit(
                X_train_seq,
                y_train_seq,
                epochs=20,
                batch_size=32,
                verbose=0
            )

            p_scaled = lstm_m.predict(X_test_seq, verbose=0)
            p = sc_y.inverse_transform(p_scaled).flatten()

            y_test_lstm = y_test.iloc[time_steps:]

            f_series = recursive_forecast_lstm(
                lstm_m,
                sc_x,
                sc_y,
                df_model,
                X,
                target_var,
                feat_cols,
                steps=24,
                time_steps=time_steps
            )

            results[m] = {
                "MAE": mean_absolute_error(y_test_lstm, p),
                "RMSE": np.sqrt(mean_squared_error(y_test_lstm, p)),
                "R2": r2_score(y_test_lstm, p)
            }

            val_preds[m] = pd.Series(p, index=y_test_lstm.index)
            future_preds[m] = f_series


if not results:
    st.error("No model was successfully trained. Please check model availability and dataset columns.")
    render_footer()
    st.stop()


# ======================================================
# DASHBOARD OUTPUTS
# ======================================================
st.header(f"📊 {target_var} Forecast Insights — {mode_choice}")

# Pro Health Alert
max_v = max([series.max() for series in future_preds.values()])

if max_v <= 50:
    st.success(f"🍃 **Good Air Quality:** Peak predicted at {max_v:.1f}.")
elif max_v <= 100:
    st.warning(f"⚠️ **Moderate Risk:** Peak predicted at {max_v:.1f}.")
else:
    st.error(f"🚨 **Poor Air Quality:** Peak predicted at {max_v:.1f}.")


# ======================================================
# ROW 1: METRICS & FEATURE IMPORTANCE
# ======================================================
c1, c2 = st.columns(2)

with c1:
    st.subheader("🏆 Model Metrics")
    st.table(pd.DataFrame(results).T.style.format("{:.4f}"))

with c2:
    if importances:
        st.subheader("💡 Key Drivers")

        m_to_show = "XGBoost" if "XGBoost" in importances else list(importances.keys())[0]

        imp_values = importances[m_to_show]
        imp_ser = pd.Series(imp_values, index=feat_cols).sort_values().tail(7)

        fig_imp, ax_imp = plt.subplots()
        imp_ser.plot(kind="barh", ax=ax_imp, color="#0059a8")
        ax_imp.set_title(f"Factors for {m_to_show}")
        st.pyplot(fig_imp)
    else:
        st.info("💡 Feature importance is not available for LSTM.")


# ======================================================
# PLOT 1: HISTORICAL VALIDATION
# ======================================================
st.subheader("📉 Plot 1: Historical Validation")

fig1, ax1 = plt.subplots(figsize=(12, 4))

actual_test = y_test.reset_index(drop=True)

ax1.plot(
    actual_test.values[-100:],
    label="Actual",
    color="black",
    alpha=0.5,
    linewidth=2
)

for m_n, p_v in val_preds.items():
    pred_values = pd.Series(p_v).reset_index(drop=True)

    ax1.plot(
        pred_values.values[-100:],
        label=f"{m_n} Pred",
        linestyle="--",
        color=colors.get(m_n)
    )

ax1.set_xlabel("Recent Test Observations")
ax1.set_ylabel(target_var)
ax1.legend()
st.pyplot(fig1)


# ======================================================
# PLOT 2: FUTURE FORECAST
# ======================================================
st.subheader("🔮 Plot 2: 24-Hour Future Trend Forecast")

df_model["DATETIME_HOUR"] = pd.to_datetime(df_model["DATETIME_HOUR"], errors="coerce")
df_model = df_model.dropna(subset=["DATETIME_HOUR"]).sort_values("DATETIME_HOUR")

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    df_model["DATETIME_HOUR"].tail(24),
    df_model[target_var].tail(24),
    label="Recent History",
    color="black",
    linewidth=2
)

for m_n, f_v in future_preds.items():
    ax2.plot(
        f_v.index,
        f_v.values,
        label=f"{m_n} Forecast",
        marker="o",
        linestyle="--",
        color=colors.get(m_n)
    )

plt.xticks(rotation=45)
ax2.set_xlabel("Time")
ax2.set_ylabel(target_var)
ax2.legend()
st.pyplot(fig2)


# ======================================================
# FORECAST TABLE
# ======================================================
with st.expander("📋 View Detailed Forecast Table"):
    forecast_table = pd.DataFrame({m: s for m, s in future_preds.items()})
    st.dataframe(forecast_table.style.format("{:.2f}"))


render_footer()