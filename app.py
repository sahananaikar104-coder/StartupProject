# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load dataset ---
df = pd.read_csv(r"C:\Users\sahan\Downloads\50_Startups.csv")  # Update path if needed

st.title("🚀 Startup Profit Prediction App (Indian Cities)")

st.write("""
This app predicts the **Profit** of a startup based on spending and city.
""")

# --- 2. Replace US states with Indian cities ---
df['State'] = df['State'].replace({
    'New York': 'Bangalore',
    'California': 'Mumbai',
    'Florida': 'Delhi'
})

# --- 3. Data preprocessing ---
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
X = df_encoded.drop('Profit', axis=1)
y = df_encoded['Profit']

# Train Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. User Inputs ---
st.header("Enter Startup Details")
rd_spend = st.number_input("R&D Spend", min_value=0.0, value=100000.0)
admin_spend = st.number_input("Administration Spend", min_value=0.0, value=50000.0)
marketing_spend = st.number_input("Marketing Spend", min_value=0.0, value=50000.0)
city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi"])

# Encode city for prediction
city_new = [0, 0]  # Bangalore is baseline
if city == "Mumbai":
    city_new = [1, 0]
elif city == "Delhi":
    city_new = [0, 1]

# Prepare input features
input_features = [[rd_spend, admin_spend, marketing_spend] + city_new]

# --- 5. Prediction ---
predicted_profit = model.predict(input_features)[0]
st.success(f"Predicted Profit for this startup: **₹{predicted_profit:,.2f}**")

# --- 6. Feature contributions ---
st.subheader("Approximate Contribution of Each Feature")
coefficients = model.coef_
feature_names = X.columns

contributions = {}
for i, feature in enumerate(feature_names):
    contributions[feature] = input_features[0][i] * coefficients[i]

# Display contributions in a table
contrib_df = pd.DataFrame(list(contributions.items()), columns=['Feature', 'Contribution'])
st.table(contrib_df)

# --- 7. Optional visualization ---
st.subheader("Comparison with Average Profit")
avg_profit = y.mean()

fig, ax = plt.subplots()
ax.bar(["Predicted Profit", "Average Profit"], [predicted_profit, avg_profit], color=['blue','orange'])
ax.set_ylabel("Profit")
st.pyplot(fig)

# --- 8. Correlation heatmap (optional) ---
if st.checkbox("Show Correlation Heatmap"):
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
