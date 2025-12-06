# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# --- 1. Load dataset ---
df = pd.read_csv(r"C:\Users\sahan\Downloads\50_Startups.csv")  # Update path if needed

st.title("🚀 Startup Profit Prediction App")

st.write("""
This app predicts the **Profit** of a startup based on spending.
""")

# --- 2. Data preprocessing ---
# Only numeric features for simplicity
X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y = df['Profit']

# Train Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# --- 3. User Inputs ---
st.header("Enter Startup Details")
rd_spend = st.number_input("R&D Spend", min_value=0.0, value=100000.0)
admin_spend = st.number_input("Administration Spend", min_value=0.0, value=50000.0)
marketing_spend = st.number_input("Marketing Spend", min_value=0.0, value=50000.0)

# Prepare input features
input_features = [[rd_spend, admin_spend, marketing_spend]]

# --- 4. Prediction ---
predicted_profit = model.predict(input_features)[0]
st.success(f"Predicted Profit for this startup: **₹{predicted_profit:,.2f}**")

# --- 5. Feature contributions ---
st.subheader("Approximate Contribution of Each Feature")
coefficients = model.coef_
feature_names = X.columns

contributions = {}
for i, feature in enumerate(feature_names):
    contributions[feature] = input_features[0][i] * coefficients[i]

# Display contributions in a table with formatted numbers
contrib_df = pd.DataFrame(list(contributions.items()), columns=['Feature', 'Contribution'])
contrib_df['Contribution'] = contrib_df['Contribution'].apply(lambda x: f"₹{x:,.2f}")
st.table(contrib_df)

# --- 6. Comparison with Average Profit ---
st.subheader("Comparison with Average Profit")
avg_profit = y.mean()

fig, ax = plt.subplots()
ax.bar(["Predicted Profit", "Average Profit"], [predicted_profit, avg_profit], color=['blue','orange'])
ax.set_ylabel("Profit")
ax.set_title("Predicted vs Average Profit")
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('₹{x:,.0f}'))
st.pyplot(fig)
