import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import altair as alt
import numpy as np

# ---- Page Config ----
st.set_page_config(
    page_title="🚀 Startup Profit Prediction",
    page_icon="💹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---- Background & Custom CSS ----
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f0f4f8, #e8f1f5);
        color: #333333;
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7ea6c1, #f7a6c1);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 0;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown("<h1 style='text-align:center;'>🚀 Startup Profit Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Enter your startup details below to predict expected profit 💰</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---- Load Dataset ----
df = pd.read_csv("50_Startups.csv")

# Map US States to Indian Cities
df["State"] = df["State"].replace({
    "New York": "Bangalore",
    "California": "Mumbai",
    "Florida": "Delhi"
})

# One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Profit", axis=1)
y = df_encoded["Profit"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Input Section ----
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter Startup Details 🏙️")
    rnd = st.number_input("R&D Spend (₹) 🏗️", value=100000.0)
    admin = st.number_input("Administration Spend (₹) 🏢", value=50000.0)
    marketing = st.number_input("Marketing Spend (₹) 📢", value=50000.0)
    city = st.selectbox("City 🌆", ["Bangalore", "Mumbai", "Delhi"])
    st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💰 Predicted Profit")
    st.markdown(f"<h2 style='color:#7ea6c1;'>{prediction:,.2f} ₹</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Feature Contribution ----
contributions = model.coef_ * list(input_df.iloc[0])
feature_names = X.columns

contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contributions
})

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Feature Contribution")
    chart = alt.Chart(contrib_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Contribution', title='Contribution Amount'),
        y=alt.Y('Feature', sort='-x', title=''),
        color=alt.Color('Contribution', scale=alt.Scale(scheme='pastel1'), legend=None),
        tooltip=['Feature', 'Contribution']
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Profit Comparison ----
avg_profit = df["Profit"].mean()
comparison_df = pd.DataFrame({
    "Category": ["Predicted Profit 💰", "Average Profit 📊"],
    "Profit": [prediction, avg_profit]
})

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💹 Profit Comparison")
    chart2 = alt.Chart(comparison_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Category', title=''),
        y=alt.Y('Profit', title='Profit (₹)'),
        color=alt.Color('Category', scale=alt.Scale(range=['#7ea6c1', '#f7a6c1']), legend=None),
        tooltip=['Category', 'Profit']
    ).properties(height=250)
    st.altair_chart(chart2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
