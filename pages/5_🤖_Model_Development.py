import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try: from xgboost import XGBRegressor
except: XGBRegressor = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from ui_components import load_full_ui, render_footer

# Setup Page
st.write("")
load_full_ui()

df = st.session_state.get("df_engineered")
if df is None:
    st.warning("Please upload the dataset and do feature engineering.")
    st.stop()
# Retrieve Sidebar Selections
target_var = st.session_state.get("selected_target", "AQI")
model_choice = st.session_state.get("selected_model", "Random Forest")
run_btn = st.session_state.get("run_prediction", False)

# --- 1. Welcome & Info Panel ---
if not run_btn:
    st.info("👋 **Welcome to the Auckland Council AQI Forecasting Engine**")
    st.markdown(f"Please select your **Target Variable** and **Model** from the sidebar and click **Run Forecast**.")
    st.stop()

# --- 2. Data Preparation ---
def prepare_data(data, target):
    df_m = data.copy().sort_values("DATETIME_HOUR")
    df_m['LABEL'] = df_m[target].shift(-24) # Direct 24h Prediction
    # Scientific Feature List based on your Correlation Findings
    features = ["hour_sin", "hour_cos", "day_sin", "day_cos", "NO2", "PM2.5", "TRAFFICV",
                "TEMP", "RH", "WS", f"{target}_lag_1", f"{target}_lag_24", f"{target}_roll_mean_24"]
    features = [f for f in features if f in df_m.columns]
    df_ready = df_m.dropna(subset=['LABEL'] + features)
    return df_ready[features], df_ready['LABEL'], features

X, y, feat_cols = prepare_data(df, target_var)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

def get_future_features(last_record, feat_list):
    future_rows = []
    last_time = last_record["DATETIME_HOUR"]
    for i in range(1, 25):
        f_time = last_time + pd.Timedelta(hours=i)
        row = {f: last_record.get(f, 0) for f in feat_list}
        row.update({"hour_sin": np.sin(2 * np.pi * f_time.hour / 24), "hour_cos": np.cos(2 * np.pi * f_time.hour / 24)})
        future_rows.append(row)
    return pd.DataFrame(future_rows)[feat_list]

# --- 3. Modeling Loop with Automated Tuning ---
results, val_preds, future_preds, importances = {}, {}, {}, {}
models_to_run = ["Random Forest", "XGBoost", "LSTM"] if model_choice == "Compare All" else [model_choice]
X_fut = get_future_features(df.iloc[-1], feat_cols)
colors = {"Random Forest": "#1f77b4", "XGBoost": "#2ca02c", "LSTM": "#d62728"}

for m in models_to_run:
    with st.spinner(f"Optimising {m} with Automated Tuning..."):
        if m == "Random Forest":
            grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
            search = RandomizedSearchCV(RandomForestRegressor(random_state=42), grid, n_iter=5, cv=3, random_state=42).fit(X_train, y_train)
            best_m = search.best_estimator_
            p, f = best_m.predict(X_test), best_m.predict(X_fut)
            importances[m] = best_m.feature_importances_
            st.caption(f"✅ RF Best Params: {search.best_params_}")

        elif m == "XGBoost":
            grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
            search = RandomizedSearchCV(XGBRegressor(random_state=42), grid, n_iter=5, cv=3, random_state=42).fit(X_train, y_train)
            best_m = search.best_estimator_
            p, f = best_m.predict(X_test), best_m.predict(X_fut)
            importances[m] = best_m.feature_importances_
            st.caption(f"✅ XGB Best Params: {search.best_params_}")

        elif m == "LSTM":
            sc_x, sc_y = MinMaxScaler(), MinMaxScaler()
            Xt_s = sc_x.fit_transform(X_train).reshape((-1, 1, len(feat_cols)))
            yt_s = sc_y.fit_transform(y_train.values.reshape(-1, 1))
            Xv_s = sc_x.transform(X_test).reshape((-1, 1, len(feat_cols)))
            if not TENSORFLOW_AVAILABLE:
                st.warning("LSTM model is not available because TensorFlow is not installed on Streamlit Cloud.")
            else:
                lstm_m = Sequential([
                LSTM(50, activation='relu', input_shape=(1, len(feat_cols))),
                Dense(1)
                ])
            lstm_m.compile(optimizer='adam', loss='mse'); lstm_m.fit(Xt_s, yt_s, epochs=10, verbose=0)
            p = sc_y.inverse_transform(lstm_m.predict(Xv_s)).flatten()
            f = sc_y.inverse_transform(lstm_m.predict(sc_x.transform(X_fut).reshape(24, 1, len(feat_cols)))).flatten()

        results[m] = {"MAE": mean_absolute_error(y_test, p), "RMSE": np.sqrt(mean_squared_error(y_test, p)), "R2": r2_score(y_test, p)}
        val_preds[m], future_preds[m] = p, f

# --- 4. Dashboard Outputs ---
st.header(f"📊 {target_var} Forecast Insights")

# Pro Health Alert
max_v = max([max(v) for v in future_preds.values()])
if max_v <= 50: st.success(f"🍃 **Good Air Quality:** Peak predicted at {max_v:.1f}.")
elif max_v <= 100: st.warning(f"⚠️ **Moderate Risk:** Peak predicted at {max_v:.1f}.")
else: st.error(f"🚨 **Poor Air Quality:** Peak predicted at {max_v:.1f}.")

# --- Row 1: Metrics & Feature Importance ---
c1, c2 = st.columns(2)
with c1: 
    st.subheader("🏆 Model Metrics")
    st.table(pd.DataFrame(results).T.style.format("{:.4f}"))

with c2: 
    # Check if we have importance data and that we aren't just running LSTM
    if importances:
        st.subheader("💡 Key Drivers")
        
        # If 'Compare All', we'll just show XGBoost or Random Forest importance
        # If a single model, we show that specific one
        m_to_show = "XGBoost" if "XGBoost" in importances else list(importances.keys())[0]
        
        # Correctly create the Series using a single key
        imp_values = importances[m_to_show]
        imp_ser = pd.Series(imp_values, index=feat_cols).sort_values().tail(7)
        
        fig_imp, ax_imp = plt.subplots()
        imp_ser.plot(kind='barh', ax=ax_imp, color='#0059a8')
        ax_imp.set_title(f"Factors for {m_to_show}")
        st.pyplot(fig_imp)
    else:
        st.info("💡 Feature importance is not available for LSTM.")


# Plot 1: Validation (Actual vs Predicted)
st.subheader("📉 Plot 1: Historical Validation")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(y_test.values[-100:], label="Actual", color="black", alpha=0.5, linewidth=2)
for m_n, p_v in val_preds.items(): ax1.plot(p_v[-100:], label=f"{m_n} Pred", linestyle="--", color=colors.get(m_n))
ax1.legend(); st.pyplot(fig1)

# Plot 2: Future Forecast (Red-Line Plot)
st.subheader(f"🔮 Plot 2: 24-Hour Future Trend Forecast")
df["DATETIME_HOUR"] = pd.to_datetime(df["DATETIME_HOUR"], errors="coerce")
df = df.dropna(subset=["DATETIME_HOUR"]).sort_values("DATETIME_HOUR")

f_times = pd.date_range(
    start=df["DATETIME_HOUR"].iloc[-1] + pd.Timedelta(hours=1),
    periods=24,
    freq="h"
)
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df["DATETIME_HOUR"].tail(24), df[target_var].tail(24), label="Recent History", color="black", linewidth=2)
for m_n, f_v in future_preds.items(): ax2.plot(f_times, f_v, label=f"{m_n} Forecast", marker="o", linestyle="--", color=colors.get(m_n))
plt.xticks(rotation=45); ax2.legend(); st.pyplot(fig2)

# Row 4: Numerical Forecast Table
with st.expander("📋 View Detailed Forecast Table"):
    st.dataframe(pd.DataFrame(future_preds, index=f_times).style.format("{:.2f}"))

render_footer()
