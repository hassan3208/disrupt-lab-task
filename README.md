
# Sales Forecasting using Dense Neural Network and XGBoost

<img width="2304" height="950" alt="localhost_8501_" src="https://github.com/user-attachments/assets/633b51ad-76c4-421a-b9b1-1c9394d4f3ee" />

## Overview

This project focuses on **sales forecasting** using machine learning.
The goal is to predict **next-day sales** based on historical sales patterns and compare the performance of two different models:

* **Dense Neural Network (Deep Learning)**
* **XGBoost (Tree-based Machine Learning)**

The project also includes a **Streamlit web application** that allows users to visualize predictions and manually test both models.

---

## Problem Statement

Accurate sales forecasting helps businesses with:

* Inventory planning
* Demand estimation
* Resource allocation

Daily sales data is often noisy, so the objective is to build models that can **learn trends from historical data** and provide reasonable future estimates.

---

## Dataset

* Source: Kaggle – Store Sales Time Series dataset
* Data used:

  * One store (`store_nbr = 1`)
  * One product category (`AUTOMOTIVE`)
* Target variable: `sales`

Only sales-related fields were used to keep the problem focused and realistic.

---

## Feature Engineering

The following features were created from historical sales data:

### Lag Features

* `lag_1` → sales 1 day ago
* `lag_7` → sales 7 days ago
* `lag_14` → sales 14 days ago
* `lag_30` → sales 30 days ago

### Rolling Statistics

* `rolling_7` → 7-day moving average
* `rolling_14` → 14-day moving average
* `rolling_30` → 30-day moving average

### Date Features

* Day of month
* Month
* Day of week

These features help the models capture short-term trends and seasonal patterns.

---

## Models Used

### 1. Dense Neural Network

* Feed-forward neural network
* Input features scaled using `StandardScaler`
* Trained to learn overall demand trends
* Produces smooth predictions

### 2. XGBoost

* Gradient boosting decision tree model
* Uses unscaled features
* More responsive to short-term fluctuations
* Strict about feature consistency during prediction

---

## Model Evaluation

Models were evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

Both models were tested on unseen data and compared fairly using the same test set.

Final model selection was based on **numerical metrics**, not just visual appearance.

---

## Streamlit Application

The Streamlit app provides:

* Model performance comparison (MAE, RMSE)
* Actual vs predicted sales plots
* Option to visualize:

  * Dense Neural Network
  * XGBoost
  * Both models together
* Manual prediction interface where users can:

  * Enter recent sales values
  * See predicted next-day sales from both models

The app is designed so that **non-technical users** can also understand and interact with the predictions.

---

## How to Run the Project

### Install dependencies

```bash
pip install streamlit tensorflow xgboost scikit-learn pandas matplotlib
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Make sure the following files are present in the same directory:

* `app.py`
* `train.csv`
* `dense_nn_sales_forecast.h5`
* `xgboost_sales_forecast.json`

---

## Project Structure

```
├── app.py
├── train.csv
├── dense_nn_sales_forecast.h5
├── xgboost_sales_forecast.json
├── README.md
```

---

## Conclusion

This project demonstrates:

* Proper data preprocessing and feature engineering
* Comparison of deep learning and classical ML models
* Model evaluation using correct regression metrics
* Deployment of models using a simple web interface

The work reflects a **practical, industry-oriented approach** to machine learning rather than focusing only on model accuracy.


## Author

**Hassan Imran**
Bachelor of Computer Science
Aspiring Machine Learning / AI Engineer


