import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# ---- Page Config ----
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---- Custom CSS for Pastel Pink Theme ----
st.markdown("""
    <style>
    .stApp {
        background-color: #ffe4e1;  /* classy pastel pink */
        color: #4d4d4d;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #ffb6b9;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stNumberInput>div>input {
        border-radius: 8px;
        padding: 5px;
    }
    .stSelectbox>div>div>div {
        border-radius: 8px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("🚀 Startup Profit Prediction App")
st.subheader("Enter your startup details below to predict expected profit 💰")

# ---- Load Dataset ----
df = pd.read_csv("50_Startups.csv")
df["State"] = df["State"].replace({
    "New York": "Bangalore",
    "California": "Mumbai",
    "Florida": "Delhi"
})
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Profit", axis=1)
y = df_encoded["Profit"]

# ---- Train Model ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ---- User Input Section ----
st.subheader("Enter Startup Details 🏙️")
rnd = st.number_input("R&D Spend (₹)", value=100000.0)
admin = st.number_input("Administration Spend (₹)", value=50000.0)
marketing = st.number_input("Marketing Spend (₹)", value=50000.0)
city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi"])

# ---- Predict Button ----
if st.button("Predict Profit 💰"):
    input_data = {
        "R&D Spend": rnd,
        "Administration": admin,
        "Marketing Spend": marketing,
        "State_Delhi": 1 if city == "Delhi" else 0,
        "State_Mumbai": 1 if city == "Mumbai" else 0
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Profit: ₹{prediction:,.2f}")

    # ---- Feature Contribution ----
    contributions = model.coef_ * list(input_df.iloc[0])
    feature_names = X.columns
    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    }).sort_values(by="Contribution", ascending=False)
    st.subheader("Feature Contribution 📈")
    st.bar_chart(contrib_df.set_index("Feature"))

    # ---- Profit Comparison ----
    avg_profit = df["Profit"].mean()
    comparison_df = pd.DataFrame({
        "Category": ["Predicted Profit 💰", "Average Profit 📊"],
        "Profit": [prediction, avg_profit]
    })
    st.subheader("Profit Comparison 💹")
    st.bar_chart(comparison_df.set_index("Category"))
