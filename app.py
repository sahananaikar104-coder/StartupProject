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

# ---- Custom CSS for background and colors ----
st.markdown("""
<style>
body {
    background-color: #f9f6f7;  /* Soft pastel beige */
    color: #333333;
    font-family: 'Helvetica', sans-serif;
}

.stButton>button {
    background-color: #7ea6c1;  /* muted teal button */
    color: white;
    font-size: 16px;
    border-radius: 8px;
}

.stNumberInput>div>div>input {
    border-radius: 8px;
}

h1, h2, h3, h4, h5 {
    color: #4b4b4b;
}

.css-18e3th9 {  /* Streamlit main container */
    background-image: url("https://images.unsplash.com/photo-1612832021202-4c5a31c128bc?auto=format&fit=crop&w=1350&q=80");  /* subtle abstract background */
    background-size: cover;
    background-attachment: fixed;
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ---- User Inputs ----
st.header("🏙️ Startup Details")
rnd = st.number_input("R&D Spend (₹)", value=100000.0)
admin = st.number_input("Administration Spend (₹)", value=50000.0)
marketing = st.number_input("Marketing Spend (₹)", value=50000.0)
city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi"])

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
st.success(f"💰 Predicted Profit: ₹{prediction:,.2f}")

# ---- Feature Contribution ----
contributions = model.coef_ * list(input_df.iloc[0])  
feature_names = X.columns
contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contributions
})

st.header("📈 Feature Contribution")
st.write(contrib_df.sort_values(by="Contribution", ascending=False))

# ---- Profit Comparison ----
st.header("💹 Profit Comparison")
avg_profit = df["Profit"].mean()
comparison_df = pd.DataFrame({
    "Category": ["Predicted Profit 💰", "Average Profit 📊"],
    "Profit": [prediction, avg_profit]
})
st.altair_chart(
    alt.Chart(comparison_df)
        .mark_bar(color="#7ea6c1")
        .encode(
            x='Category',
            y='Profit',
            tooltip=['Category', 'Profit']
        )
        .properties(height=300)
)
