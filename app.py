import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# ---- Page Config ----
st.set_page_config(page_title="Startup Profit Prediction", layout="wide")

# ---- Title ----
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚀 Startup Profit Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---- Load Dataset ----
df = pd.read_csv("50_Startups.csv")

# Replace US states with Indian cities
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

# ---- User Input ----
st.subheader("Enter Startup Details")
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        rnd = st.number_input("R&D Spend", value=100000.0, step=1000.0, format="%.2f")
    with col2:
        admin = st.number_input("Administration Spend", value=50000.0, step=1000.0, format="%.2f")
    with col3:
        marketing = st.number_input("Marketing Spend", value=50000.0, step=1000.0, format="%.2f")

city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi"])

# ---- Prediction ----
input_data = {
    "R&D Spend": rnd,
    "Administration": admin,
    "Marketing Spend": marketing,
    "State_Delhi": 1 if city == "Delhi" else 0,
    "State_Mumbai": 1 if city == "Mumbai" else 0
}
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

# ---- Display Profit ----
st.markdown(f"<h2 style='text-align: center; color: #FF5733;'>💰 Predicted Profit: ₹{prediction:,.2f}</h2>", unsafe_allow_html=True)
st.markdown("---")

# ---- Feature Contributions ----
contributions = model.coef_ * list(input_df.iloc[0])
feature_names = X.columns
contrib_df = pd.DataFrame({"Feature": feature_names, "Contribution": contributions})

st.subheader("Feature Contribution")
fig, ax = plt.subplots(figsize=(6,4))
colors = ['#4CAF50' if val > 0 else '#F44336' for val in contributions]
ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors)
ax.set_xlabel("Contribution Amount")_
