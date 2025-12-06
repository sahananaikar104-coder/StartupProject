import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import altair as alt

# ---- Page Config ----
st.set_page_config(
    page_title="Startup Profit Prediction",
    layout="wide",
    page_icon="💼"
)

# ---- Custom CSS for background, colors and card style ----
st.markdown("""
<style>
body {
    background: linear-gradient(160deg, #f0f4f8, #e8f1f5);
    font-family: 'Helvetica', sans-serif;
    color: #333;
}

.css-18e3th9 {
    background: linear-gradient(160deg, #f0f4f8, #e8f1f5);
}

.card {
    background-color: rgba(255, 255, 255, 0.92);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    transition: transform 0.2s;
}

.card:hover {
    transform: translateY(-5px);
}

.stButton>button {
    background: linear-gradient(90deg, #7ea6c1, #f7a6c1);
    color: white;
    font-size: 16px;
    border-radius: 12px;
    padding: 10px 25px;
    transition: all 0.2s;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #f7a6c1, #7ea6c1);
    cursor: pointer;
}

h1, h2, h3, h4 {
    color: #4b4b4b;
}

.stNumberInput>div>div>input {
    border-radius: 10px;
    padding: 8px;
    border: 1px solid #ccc;
}

div.stSelectbox>div>div>div>select {
    border-radius: 10px;
    padding: 8px;
    border: 1px solid #ccc;
}
</style>
""", unsafe_allow_html=True)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Title ----
st.markdown("<h1 style='text-align:center'>🚀 Startup Profit Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center'>Enter your startup details below to predict expected profit 💰</h4>", unsafe_allow_html=True)

# ---- Input Card ----
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        rnd = st.number_input("R&D Spend (₹) 🏗️", value=100000.0)
        admin = st.number_input("Administration Spend (₹) 🏢", value=50000.0)
        
    with col2:
        marketing = st.number_input("Marketing Spend (₹) 📢", value=50000.0)
        city = st.selectbox("City 🌆", ["Bangalore", "Mumbai", "Delhi"])
    
    predict_button = st.button("Predict Profit 💰")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Prediction ----
if predict_button:
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

    st.header("📈 Feature Contribution")
    chart = alt.Chart(contrib_df).mark_bar(color="#f7a6c1").encode(
        x='Contribution',
        y=alt.Y('Feature', sort='-x'),
        tooltip=['Feature', 'Contribution']
    ).properties(height=300).interactive()
    st.altair_chart(chart, use_container_width=True)

    # ---- Profit Comparison ----
    avg_profit = df["Profit"].mean()
    comparison_df = pd.DataFrame({
        "Category": ["Predicted Profit 💰", "Average Profit 📊"],
        "Profit": [prediction, avg_profit]
    })
    st.header("💹 Profit Comparison")
    bar = alt.Chart(comparison_df).mark_bar(color="#7ea6c1").encode(
        x='Category',
        y='Profit',
        tooltip=['Category', 'Profit']
    ).properties(height=300).interactive()
    st.altair_chart(bar, use_container_width=True)
