import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
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
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from ui_components import load_full_ui, render_footer


# ======================================================
# PAGE SETUP
# ======================================================
st.write("")
load_full_ui()

df = st.session_state.get("df_engineered")

if df is None:
    st.warning("""
⬅ Please upload the dataset and complete feature engineering first.

App Workflow:
1️⃣ Upload datasets  
2️⃣ Clean data  
3️⃣ Apply feature engineering  
4️⃣ Select target and model from sidebar  
5️⃣ Run 24-hour forecasting
""")
    render_footer()
    st.stop()

st.header("🤖 AI Model Development & 24-Hour Forecasting")


# ======================================================
# SIDEBAR SELECTIONS
# ======================================================
target_var = st.session_state.get("selected_target", "AQI")
model_choice = st.session_state.get("selected_model", "LSTM")
run_btn = st.session_state.get("run_prediction", False)

mode_choice = st.radio(
    "Forecasting Mode",
    ["Multivariate", "Univariate"],
    horizontal=True
)

if not run_btn:
    st.info("Please select target variable and model from the sidebar, then click Run Forecast.")
    render_footer()
    st.stop()


# ======================================================
# DATA PREPARATION
# ======================================================
def prepare_data(data, target, mode):
    df_m = data.copy()

    if "DATETIME_HOUR" not in df_m.columns:
        st.error("DATETIME_HOUR column is missing. Please run Feature Engineering again.")
        render_footer()
        st.stop()

    df_m["DATETIME_HOUR"] = pd.to_datetime(df_m["DATETIME_HOUR"], errors="coerce")
    df_m = df_m.dropna(subset=["DATETIME_HOUR"])
    df_m = df_m.sort_values("DATETIME_HOUR").reset_index(drop=True)

    if target not in df_m.columns:
        st.error(f"{target} column is missing from the engineered dataset.")
        render_footer()
        st.stop()

    # Predict next-hour target value
    df_m["LABEL"] = df_m[target].shift(-1)

    time_features = [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos"
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

if len(df_model) < 120:
    st.error("Not enough rows for reliable model training.")
    render_footer()
    st.stop()

split_idx = int(len(df_model) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]


# ======================================================
# METRIC FUNCTIONS
# ======================================================
def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def display_metric_cards(metrics):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("RMSE", f"{metrics['RMSE']:.3f}")

    with c2:
        st.metric("MAE", f"{metrics['MAE']:.3f}")

    with c3:
        st.metric("R² Score", f"{metrics['R2']:.3f}")


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
            current_row,
            target,
            pred,
            history_before_prediction,
            next_time,
            feat_cols
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
future_preds = {}
importances = {}
best_params_store = {}

models_to_run = ["Random Forest", "XGBoost", "LSTM"] if model_choice == "Compare All" else [model_choice]

tscv = TimeSeriesSplit(n_splits=5)

colors = {
    "Random Forest": "#1f77b4",
    "XGBoost": "#2ca02c",
    "LSTM": "#d62728"
}

st.subheader(f"🎯 Forecast Target: {target_var}")
st.caption(f"Mode: {mode_choice} | Forecast horizon: Next 24 hours")


for m in models_to_run:

    with st.spinner(f"Training and optimising {m}..."):

        # ==================================================
        # RANDOM FOREST
        # ==================================================
        if m == "Random Forest":

            baseline_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            baseline_model.fit(X_train, y_train)
            base_pred = baseline_model.predict(X_test)
            baseline_results[m] = calculate_metrics(y_test, base_pred)

            grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [8, 12, 16, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 3]
            }

            search = RandomizedSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                grid,
                n_iter=8,
                cv=tscv,
                random_state=42,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            best_m = search.best_estimator_

            cv_score = cross_val_score(
                best_m,
                X_train,
                y_train,
                cv=tscv,
                scoring="neg_root_mean_squared_error"
            )

            p = best_m.predict(X_test)
            f_series = recursive_forecast_tree(best_m, df_model, X, target_var, feat_cols)

            results[m] = calculate_metrics(y_test, p)
            cv_results[m] = -cv_score.mean()
            val_preds[m] = pd.Series(p, index=y_test.index)
            actual_series[m] = pd.Series(y_test.values, index=y_test.index)
            future_preds[m] = f_series
            importances[m] = best_m.feature_importances_
            best_params_store[m] = search.best_params_

        # ==================================================
        # XGBOOST
        # ==================================================
        elif m == "XGBoost":

            if XGBRegressor is None:
                st.warning("XGBoost is not installed. Please add xgboost to requirements.txt.")
                continue

            baseline_model = XGBRegressor(
                random_state=42,
                objective="reg:squarederror"
            )

            baseline_model.fit(X_train, y_train)
            base_pred = baseline_model.predict(X_test)
            baseline_results[m] = calculate_metrics(y_test, base_pred)

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
                n_iter=8,
                cv=tscv,
                random_state=42,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            best_m = search.best_estimator_

            cv_score = cross_val_score(
                best_m,
                X_train,
                y_train,
                cv=tscv,
                scoring="neg_root_mean_squared_error"
            )

            p = best_m.predict(X_test)
            f_series = recursive_forecast_tree(best_m, df_model, X, target_var, feat_cols)

            results[m] = calculate_metrics(y_test, p)
            cv_results[m] = -cv_score.mean()
            val_preds[m] = pd.Series(p, index=y_test.index)
            actual_series[m] = pd.Series(y_test.values, index=y_test.index)
            future_preds[m] = f_series
            importances[m] = best_m.feature_importances_
            best_params_store[m] = search.best_params_

        # ==================================================
        # LSTM
        # ==================================================
        elif m == "LSTM":

            if not TENSORFLOW_AVAILABLE:
                st.warning("LSTM is not available because TensorFlow is not installed.")
                continue

            time_steps = 24

            if len(X_train) <= time_steps or len(X_test) <= time_steps:
                st.warning("Not enough data for LSTM sequence training.")
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

            X_test_seq, y_test_seq_raw = make_lstm_sequences(
                X_test_scaled,
                y_test.values,
                time_steps
            )

            lstm_m = Sequential([
                LSTM(64, return_sequences=True, input_shape=(time_steps, len(feat_cols))),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            lstm_m.compile(optimizer="adam", loss="mse")

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            history = lstm_m.fit(
                X_train_seq,
                y_train_seq,
                validation_split=0.2,
                epochs=40,
                batch_size=32,
                callbacks=[early_stop],
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

            results[m] = calculate_metrics(y_test_lstm, p)
            baseline_results[m] = results[m]
            cv_results[m] = min(history.history["val_loss"]) ** 0.5

            val_preds[m] = pd.Series(p, index=y_test_lstm.index)
            actual_series[m] = pd.Series(y_test_lstm.values, index=y_test_lstm.index)
            future_preds[m] = f_series

            best_params_store[m] = {
                "time_steps": 24,
                "epochs": len(history.history["loss"]),
                "batch_size": 32,
                "early_stopping": True,
                "dropout": 0.2
            }


if not results:
    st.error("No model was successfully trained.")
    render_footer()
    st.stop()


# ======================================================
# BEST MODEL SELECTION
# ======================================================
best_model_name = min(results, key=lambda x: results[x]["RMSE"])
best_metrics = results[best_model_name]


# ======================================================
# MAIN RESULT CARDS
# ======================================================
st.subheader("🏆 Best Prediction Result")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Best Model", best_model_name)

with c2:
    st.metric("RMSE", f"{best_metrics['RMSE']:.3f}")

with c3:
    st.metric("MAE", f"{best_metrics['MAE']:.3f}")

with c4:
    st.metric("R²", f"{best_metrics['R2']:.3f}")


# ======================================================
# HEALTH / FORECAST ALERT
# ======================================================
max_v = future_preds[best_model_name].max()

if max_v <= 50:
    st.success(f"🍃 Good forecast condition. Peak predicted {target_var}: {max_v:.1f}")
elif max_v <= 100:
    st.warning(f"⚠️ Moderate forecast condition. Peak predicted {target_var}: {max_v:.1f}")
else:
    st.error(f"🚨 High forecast condition. Peak predicted {target_var}: {max_v:.1f}")


# ======================================================
# MODEL OPTIMISATION SUMMARY
# ======================================================
st.subheader("⚙️ Optimisation Summary")

summary_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Baseline RMSE": [baseline_results[m]["RMSE"] for m in results.keys()],
    "Optimised RMSE": [results[m]["RMSE"] for m in results.keys()],
    "CV RMSE": [cv_results[m] for m in results.keys()]
})

summary_df["Improvement"] = summary_df["Baseline RMSE"] - summary_df["Optimised RMSE"]

fig_opt, ax_opt = plt.subplots(figsize=(10, 4))
ax_opt.bar(summary_df["Model"], summary_df["Optimised RMSE"])
ax_opt.set_title("Optimised Model RMSE Comparison")
ax_opt.set_ylabel("RMSE")
ax_opt.grid(True, axis="y", alpha=0.3)
st.pyplot(fig_opt)

with st.expander("View optimisation details"):
    st.dataframe(summary_df.round(3), use_container_width=True)

with st.expander("View best model parameters"):
    for m, params in best_params_store.items():
        st.write(f"**{m}**")
        st.json(params)


# ======================================================
# HISTORICAL VALIDATION PLOT
# ======================================================
st.subheader("📈 Actual vs Predicted")

fig1, ax1 = plt.subplots(figsize=(12, 4))

actual_best = actual_series[best_model_name].reset_index(drop=True)
pred_best = val_preds[best_model_name].reset_index(drop=True)

ax1.plot(
    actual_best.values[-100:],
    label="Actual",
    color="black",
    linewidth=2
)

ax1.plot(
    pred_best.values[-100:],
    label=f"{best_model_name} Prediction",
    linestyle="--",
    color=colors.get(best_model_name, "#d62728"),
    linewidth=2
)

ax1.set_xlabel("Recent Test Observations")
ax1.set_ylabel(target_var)
ax1.set_title(f"Historical Validation for {target_var}")
ax1.legend()
ax1.grid(True, alpha=0.3)

st.pyplot(fig1)


# ======================================================
# 24-HOUR FORECAST
# ======================================================
st.subheader("🔮 Next 24-Hour Forecast")

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    df_model["DATETIME_HOUR"].tail(24),
    df_model[target_var].tail(24),
    label="Recent History",
    color="black",
    linewidth=2
)

forecast_series = future_preds[best_model_name]

ax2.plot(
    forecast_series.index,
    forecast_series.values,
    label=f"{best_model_name} 24-Hour Forecast",
    marker="o",
    linestyle="--",
    color=colors.get(best_model_name, "#d62728"),
    linewidth=2
)

plt.xticks(rotation=45)
ax2.set_xlabel("Time")
ax2.set_ylabel(target_var)
ax2.set_title(f"Next 24-Hour Forecast for {target_var}")
ax2.legend()
ax2.grid(True, alpha=0.3)

st.pyplot(fig2)


with st.expander("View 24-hour forecast values"):
    forecast_table = pd.DataFrame({
        "Forecast Time": forecast_series.index,
        f"Forecasted {target_var}": forecast_series.values
    })

    st.dataframe(forecast_table.round(3), use_container_width=True)


# ======================================================
# FEATURE IMPORTANCE
# ======================================================
if best_model_name in importances:
    st.subheader("💡 Top Predictive Features")

    imp_ser = pd.Series(importances[best_model_name], index=feat_cols)
    imp_ser = imp_ser.sort_values(ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
    ax_imp.barh(imp_ser.index[::-1], imp_ser.values[::-1])
    ax_imp.set_title(f"Top Drivers for {best_model_name}")
    ax_imp.set_xlabel("Importance")
    ax_imp.grid(True, axis="x", alpha=0.3)

    st.pyplot(fig_imp)

else:
    st.info("Feature importance is not directly available for LSTM. LSTM performance is evaluated using validation loss, RMSE, MAE, and R².")


# ======================================================
# SHORT SUPERVISOR ANSWERS
# ======================================================
st.subheader("📝 Short Method Answers")

with st.expander("Why did we use many parameters?"):
    st.write(
        "Air quality is affected by pollutant levels, weather conditions, wind behaviour, and time patterns. "
        "The dashboard uses these variables as input features, while the selected sidebar parameter is the forecast target."
    )

with st.expander("Why model optimisation?"):
    st.write(
        "Model optimisation improves prediction accuracy by testing different model settings and selecting the best-performing configuration."
    )

with st.expander("Why cross-validation?"):
    st.write(
        "TimeSeriesSplit cross-validation checks whether the model performs consistently across different time periods without using future data for training."
    )

with st.expander("What are we forecasting?"):
    st.write(
        f"The dashboard forecasts the selected target parameter: {target_var}. "
        "The output shows predicted values for the next 24 hours."
    )

with st.expander("Important forecasting assumption"):
    st.write(
        "For recursive 24-hour forecasting, lag and time features are updated step-by-step. "
        "Future meteorological variables are not known, so their latest available values are carried forward unless future weather inputs are provided."
    )


# ======================================================
# SAVE RESULTS
# ======================================================
st.session_state["forecast_target"] = target_var
st.session_state["best_model"] = best_model_name
st.session_state["forecast_series"] = forecast_series
st.session_state["model_results"] = results


render_footer()