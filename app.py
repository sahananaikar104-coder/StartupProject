import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# ---- Page Config ----
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS for aesthetics ----
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(to bottom right, #ffe4e1, #fff0f5);
        color: #333333;
        font-family: 'Helvetica', sans-serif;
    }
    /* Card style */
    .card {
        background-color: #ffffffaa;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    /* Subheader */
    .st-subheader {
        color: #ff69b4;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Page Title ----
st.markdown("<h1 style='text-align:center;'>🚀 Startup Profit Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Enter your startup details below to predict expected profit 💰</h4>", unsafe_allow_html=True)
st.markdown("---")

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
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter Startup Details 🏙️")

    rnd = st.number_input("💡 R&D Spend (₹)", value=100000.0, step=1000.0, format="%.2f")
    admin = st.number_input("🏢 Administration Spend (₹)", value=50000.0, step=1000.0, format="%.2f")
    marketing = st.number_input("📣 Marketing Spend (₹)", value=50000.0, step=1000.0, format="%.2f")
    city = st.selectbox("🌆 City", ["Bangalore", "Mumbai", "Delhi"])

    st.markdown('</div>', unsafe_allow_html=True)

# ---- Prepare Input Data ----
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

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric(label="💰 Predicted Profit", value=f"₹{prediction:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Feature Contribution ----
contributions = model.coef_ * list(input_df.iloc[0])
feature_names = X.columns
contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contributions
}).sort_values(by="Contribution", ascending=False)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature Contribution 📈")
    st.bar_chart(contrib_df.set_index("Feature"))
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Profit Comparison ----
avg_profit = df["Profit"].mean()
comparison_df = pd.DataFrame({
    "Category": ["Predicted Profit 💰", "Average Profit 📊"],
    "Profit": [prediction, avg_profit]
})

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Profit Comparison 💹")
    st.bar_chart(comparison_df.set_index("Category"))
    st.markdown('</div>', unsafe_allow_html=True)
