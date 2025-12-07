import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import time

# ---- Page Config ----
st.set_page_config(
    page_title="Startup Profit Prediction",
    page_icon="💎",
    layout="wide",
)

# ---- Custom CSS (modern, classy, pastel/metallic) ----
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #f5f7fa, #e0e5ec);  /* subtle metallic pastel */
    color: #2c2c34;
    font-family: 'Segoe UI', sans-serif;
}

/* Headings */
h1, h2, h3, h4, .stMarkdown {
    color: #1f1f28;
    font-weight: 700;
}

/* Buttons - modern metallic pastel gradient + smooth hover */
.stButton>button {
    background: linear-gradient(90deg, #9b59b6, #8e44ad);
    color: white !important;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 12px;
    border: none;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
}

/* Input fields */
input, select, textarea {
    border-radius: 10px !important;
    padding: 8px !important;
    border: 1px solid #b0a4d9 !important;
}

/* Plotly chart style */
.js-plotly-plot .plotly {
    background-color: rgba(255,255,255,0.85) !important;
    border-radius: 12px;
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

# ---- Train Model ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Header ----
st.markdown("<h2 style='text-align:center;'>💎 Startup Profit Prediction App 💎</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter your startup details below to predict expected profit.</p>", unsafe_allow_html=True)

# ---- Inputs ----
rnd = st.number_input("R&D Spend (₹) 🏗️", value=100000.0)
admin = st.number_input("Administration Spend (₹) 🏢", value=50000.0)
marketing = st.number_input("Marketing Spend (₹) 📢", value=50000.0)
city = st.selectbox("City 🌆", ["Bangalore", "Mumbai", "Delhi"])

# ---- Predict Button ----
if st.button("💎 Predict Profit 💎"):
    # Data for prediction
    input_data = {
        "R&D Spend": rnd,
        "Administration": admin,
        "Marketing Spend": marketing,
        "State_Delhi": 1 if city == "Delhi" else 0,
        "State_Mumbai": 1 if city == "Mumbai" else 0
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    # ---- Animated Money Pop ----
    placeholder = st.empty()
    for i in range(15):
        placeholder.markdown(f"<h1 style='text-align:center; font-size:{30+i*2}px;'>💰</h1>", unsafe_allow_html=True)
        time.sleep(0.08)
    placeholder.empty()  # clear the animation

    st.success(f"💎 Predicted Profit: **₹{prediction:,.2f}** 💎")

    # ---- Feature Contribution ----
    contributions = model.coef_ * list(input_df.iloc[0])
    contrib_df = pd.DataFrame({
        "Feature": X.columns,
        "Contribution": contributions
    }).sort_values(by="Contribution", ascending=True)

    fig1 = px.bar(
        contrib_df,
        x="Contribution",
        y="Feature",
        orientation='h',
        color="Contribution",
        color_continuous_scale='Aggrnyl',  # pastel metallic style
        title="📊 Feature Contribution",
        text_auto=True
    )
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Profit Comparison ----
    avg_profit = df["Profit"].mean()
    comparison_df = pd.DataFrame({
        "Category": ["Average Profit 📈", "Predicted Profit 💎"],
        "Profit": [avg_profit, prediction]
    })
    fig2 = px.bar(
        comparison_df,
        x="Category",
        y="Profit",
        color="Category",
        color_discrete_map={"Average Profit 📈":"#d1c4e9", "Predicted Profit 💎":"#9b59b6"},
        text_auto=True,
        title="💹 Profit Comparison"
    )
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)
