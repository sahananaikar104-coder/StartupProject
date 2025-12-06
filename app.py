import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import altair as alt

# ---- Page Config ----
st.set_page_config(
    page_title="Startup Profit Prediction 🚀",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS for styling ----
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(160deg, #f0f4f8, #e8f1f5);
        color: #1f2937;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Cards for sections */
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    /* Input labels and spacing */
    .stNumberInput, .stSelectbox {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    /* Prediction output */
    .prediction {
        font-size: 24px;
        font-weight: bold;
        color: #2b6cb0;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown("<h1 style='text-align:center;'>🚀 Startup Profit Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter your startup details below to predict expected profit 💰</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---- Load Dataset ----
df = pd.read_csv("50_Startups.csv")
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

# ---- Input Section ----
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter Startup Details 🏙️")
    rnd = st.number_input("R&D Spend (₹) 🏗️", value=100000.0)
    admin = st.number_input("Administration Spend (₹) 🏢", value=50000.0)
    marketing = st.number_input("Marketing Spend (₹) 📢", value=50000.0)
    city = st.selectbox("City 🌆", ["Bangalore", "Mumbai", "Delhi"])
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Predict ----
input_data = {
    "R&D Spend": rnd,
    "Administration": admin,
    "Marketing Spend": marketing,
    "State_Delhi": 1 if city == "Delhi" else 0,
    "State_Mumbai": 1 if city == "Mumbai" else 0
}
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💰 Predicted Profit")
    st.markdown(f"<p class='prediction'>₹{prediction:,.2f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Feature Contribution ----
contributions = model.coef_ * list(input_df.iloc[0])
feature_names = X.columns
contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contributions
})

# Altair chart for contributions
colors = ['#7ea6c1','#f7a6c1','#a8d5ba','#fcd5b5','#c1c8e4']  # pastel elegant colors
chart = alt.Chart(contrib_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
    x=alt.X('Contribution:Q', title='Contribution Amount'),
    y=alt.Y('Feature:N', sort='-x'),
    color=alt.Color('Feature:N', scale=alt.Scale(range=colors)),
    tooltip=['Feature','Contribution']
).properties(height=250)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Feature Contribution")
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Profit Comparison ----
avg_profit = df["Profit"].mean()
comparison_df = pd.DataFrame({
    "Category": ["Predicted Profit", "Average Profit"],
    "Profit": [prediction, avg_profit]
})

comparison_chart = alt.Chart(comparison_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
    x=alt.X('Category:N', title=''),
    y=alt.Y('Profit:Q', title='Profit (₹)'),
    color=alt.Color('Category:N', scale=alt.Scale(range=['#2b6cb0','#f7a6c1'])),
    tooltip=['Category','Profit']
).properties(height=200)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💹 Profit Comparison")
    st.altair_chart(comparison_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
