import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# ---- Page Title ----
st.title("🚀 Startup Profit Prediction App")

# ---- Load Dataset ----
df = pd.read_csv("50_Startups.csv")

# Replace US states with Indian cities for model consistency
df["State"] = df["State"].replace({
    "New York": "Bangalore",
    "California": "Mumbai",
    "Florida": "Delhi"
})

# ---- One-Hot Encoding ----
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Profit", axis=1)
y = df_encoded["Profit"]

# ---- Train Model ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ---- User Inputs ----
st.subheader("Enter Startup Details")

rnd = st.number_input("R&D Spend", value=100000.0)
admin = st.number_input("Administration Spend", value=50000.0)
marketing = st.number_input("Marketing Spend", value=50000.0)

city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi"])

# ---- Create Input Row ----
input_data = {
    "R&D Spend": rnd,
    "Administration": admin,
    "Marketing Spend": marketing,
    "State_Delhi": 1 if city == "Delhi" else 0,
    "State_Mumbai": 1 if city == "Mumbai" else 0
}

input_df = pd.DataFrame([input_data])

# ---- Prediction ----
prediction = model.predict(input_df)[0]

st.success(f"Predicted Profit for this startup: ₹{prediction:,.2f}")

# ---- Feature Contribution ----
contributions = model.coef_ * list(input_df.iloc[0])  
feature_names = X.columns

contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contributions
})

st.subheader("Approximate Contribution of Each Feature")
st.write(contrib_df)

# ---- Comparison with Average Profit ----
st.subheader("Comparison with Average Profit")

avg_profit = df["Profit"].mean()

comparison_df = pd.DataFrame({
    "Category": ["Predicted Profit", "Average Profit"],
    "Profit": [prediction, avg_profit]
})

st.bar_chart(comparison_df.set_index("Category"))
