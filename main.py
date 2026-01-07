import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting – Model Comparison",
    layout="wide"
)

st.title("📊 Sales Forecasting – Dense NN vs XGBoost")

st.markdown(
    """
    This project focuses on **sales forecasting** using machine learning models.

    I compared two approaches:
    - Dense Neural Network
    - XGBoost

    The goal is to predict next-day sales using past sales data.
    """
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    dense_model = load_model("dense_nn_sales_forecast.h5")

    xgb_model = XGBRegressor()
    xgb_model.load_model("xgboost_sales_forecast.json")

    return dense_model, xgb_model


dense_model, xgb_model = load_models()

# --------------------------------------------------
# LOAD & PREPARE DATA
# --------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("train.csv")

    df = df[(df["store_nbr"] == 1) & (df["family"] == "AUTOMOTIVE")]
    df = df[["date", "sales"]]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Feature engineering (same as training)
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)
    df["lag_30"] = df["sales"].shift(30)

    df["rolling_7"] = df["sales"].rolling(7).mean()
    df["rolling_14"] = df["sales"].rolling(14).mean()
    df["rolling_30"] = df["sales"].rolling(30).mean()

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

    df.dropna(inplace=True)

    X = df.drop(["date", "sales"], axis=1)
    y = df["sales"].values

    return df, X, y


df, X_test, y_test = load_and_prepare_data()

# --------------------------------------------------
# RECREATE SCALERS
# --------------------------------------------------
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaler.fit(X_test)
y_scaler.fit(y_test.reshape(-1, 1))

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------
X_test_scaled = X_scaler.transform(X_test)

dense_preds_scaled = dense_model.predict(X_test_scaled)
dense_preds = y_scaler.inverse_transform(dense_preds_scaled).flatten()

X_test_xgb = X_test.drop(["lag_14", "rolling_30"], axis=1)
xgb_preds = xgb_model.predict(X_test_xgb)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
dense_mae = mean_absolute_error(y_test, dense_preds)
dense_rmse = np.sqrt(mean_squared_error(y_test, dense_preds))

xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))

# --------------------------------------------------
# SIDEBAR – VISUALIZATION
# --------------------------------------------------
st.sidebar.header("🔍 Visualization Options")
model_choice = st.sidebar.radio(
    "Select model output:",
    ["Dense Neural Network", "XGBoost", "Compare Both"]
)

# --------------------------------------------------
# METRICS DISPLAY
# --------------------------------------------------
st.subheader("Model Performance on Test Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Daily Sales", f"{y_test.mean():.2f}")

with col2:
    st.metric("Dense NN MAE", f"{dense_mae:.2f}")
    st.metric("Dense NN RMSE", f"{dense_rmse:.2f}")

with col3:
    st.metric("XGBoost MAE", f"{xgb_mae:.2f}")
    st.metric("XGBoost RMSE", f"{xgb_rmse:.2f}")

# --------------------------------------------------
# PLOT
# --------------------------------------------------
st.subheader("Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_test, label="Actual Sales", linewidth=2, alpha=0.7)

if model_choice == "Dense Neural Network":
    ax.plot(dense_preds, label="Dense Neural Network", linestyle="--")
elif model_choice == "XGBoost":
    ax.plot(xgb_preds, label="XGBoost")
else:
    ax.plot(dense_preds, label="Dense Neural Network", linestyle="--")
    ax.plot(xgb_preds, label="XGBoost")

ax.set_xlabel("Time Index")
ax.set_ylabel("Sales Units")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --------------------------------------------------
# MANUAL PREDICTION
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Manual Sales Prediction")

st.sidebar.markdown(
    """
    Enter recent sales values based on past days.
    These inputs are used to estimate the next day's sales.
    """
)


with st.sidebar.form("prediction_form"):
    input_date = st.date_input("Prediction Date")

    st.markdown("### Recent Sales (Units Sold)")

    lag_1 = st.number_input(
        "Sales 1 day ago",
        min_value=0,
        max_value=50,
        value=5,
        help="Number of items sold yesterday"
    )

    lag_7 = st.number_input(
        "Sales 7 days ago",
        min_value=0,
        max_value=50,
        value=4,
        help="Sales exactly one week ago"
    )

    lag_14 = st.number_input(
        "Sales 14 days ago",
        min_value=0,
        max_value=50,
        value=4,
        help="Sales two weeks ago"
    )

    lag_30 = st.number_input(
        "Sales 30 days ago",
        min_value=0,
        max_value=50,
        value=6,
        help="Sales one month ago"
    )

    st.markdown("### Average Sales Trend")

    rolling_7 = st.number_input(
        "7-Day Average Sales",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        help="Average sales of last 7 days"
    )

    rolling_14 = st.number_input(
        "14-Day Average Sales",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        help="Average sales of last 14 days"
    )

    rolling_30 = st.number_input(
        "30-Day Average Sales",
        min_value=0.0,
        max_value=50.0,
        value=6.0,
        help="Average sales of last 30 days"
    )

    submit_btn = st.form_submit_button("Predict Sales")

# --------------------------------------------------
# MANUAL PREDICTION OUTPUT
# --------------------------------------------------
if submit_btn:
    input_data = pd.DataFrame({
        "lag_1": [lag_1],
        "lag_7": [lag_7],
        "lag_14": [lag_14],
        "lag_30": [lag_30],
        "rolling_7": [rolling_7],
        "rolling_14": [rolling_14],
        "rolling_30": [rolling_30],
        "day": [input_date.day],
        "month": [input_date.month],
        "dayofweek": [input_date.weekday()]
    })

    try:
        input_scaled = X_scaler.transform(input_data)
        dense_pred_scaled = dense_model.predict(input_scaled)
        dense_pred = y_scaler.inverse_transform(dense_pred_scaled)[0][0]

        input_xgb = input_data.drop(["lag_14", "rolling_30"], axis=1)
        xgb_pred = xgb_model.predict(input_xgb)[0]

        st.markdown("---")
        st.header("Predicted Sales")

        st.success(
    "The values below show the predicted number of units that may be sold on the selected date."
)


        c1, c2 = st.columns(2)
        with c1:
            st.metric("Dense NN Prediction", f"{dense_pred:.2f} units")
        with c2:
            st.metric("XGBoost Prediction", f"{xgb_pred:.2f} units")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
