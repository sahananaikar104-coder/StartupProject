import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import time

# ---- Page Config ----
st.set_page_config(
    page_title="Startup Profit Prediction",
    page_icon="🚀",
    layout="wide",
)

# ---- Custom CSS ----
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    font-family: 'Helvetica', sans-serif;
}

/* Headings */
h1, h2, h3, h4, .stMarkdown {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

/* Input fields */
.stNumberInput>div>div>input, .stSelectbox>div>div>div>div>input {
    border-radius: 12px !important;
    padding: 10px !important;
    border: 1px solid #3b82f6 !important;
    background-color: white !important;
    color: black !important;
    transition: 0.2s;
}
.stNumberInput>div>div>input:focus, .stSelectbox>div>div>div>div>input:focus {
    border: 2px solid #2563eb !important;
    box-shadow: 0 0 8px rgba(37,99,235,0.4);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white !important;
    font-size: 18px;
    font-weight: 600;
    padding: 12px 30px;
    border-radius: 14px;
    border: none;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(37,99,235,0.4);
}

/* Chart backgrounds */
.js-plotly-plot .plotly {
    background-color: rgba(15,23,42,0.85) !important;
    border-radius: 12px;
}

/* Sparkles animation */
.sparkle {
    position: fixed;
    top: -20px;
    font-size: 18px;
    animation: fall 3s linear infinite;
    z-index: 9999;
    color: #3b82f6;
}
@keyframes fall {
    0% { transform: translateY(0) rotate(0deg); opacity: 0.8; }
    100% { transform: translateY(600px) rotate(360deg); opacity: 0; }
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
st.markdown("<h1 style='text-align:center; color:#f8fafc;'>Startup Profit Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#e2e8f0;'>Enter your startup details below to predict expected profit.</p>", unsafe_allow_html=True)

# ---- Input Section ----
col1, col2 = st.columns(2)
with col1:
    rnd = st.number_input("R&D Spend (₹)", value=100000.0, step=5000.0)
    admin = st.number_input("Administration Spend (₹)", value=50000.0, step=5000.0)
with col2:
    marketing = st.number_input("Marketing Spend (₹)", value=50000.0, step=5000.0)
    city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi"])

# ---- Predict Button ----
if st.button("Predict Profit"):
    # Loading animation
    placeholder = st.empty()
    for i in range(6):
        placeholder.markdown(f"<h3 style='text-align:center; color:#3b82f6;'>Calculating{'.'*i}</h3>", unsafe_allow_html=True)
        time.sleep(0.15)
    placeholder.empty()

    # Prepare input
    input_data = {
        "R&D Spend": rnd,
        "Administration": admin,
        "Marketing Spend": marketing,
        "State_Delhi": 1 if city == "Delhi" else 0,
        "State_Mumbai": 1 if city == "Mumbai" else 0
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    # ---- Sparkles Animation ----
    st.markdown("""
    <div>
        <div class="sparkle" style="left:5%; animation-delay:0s;">✨</div>
        <div class="sparkle" style="left:25%; animation-delay:0.3s;">✨</div>
        <div class="sparkle" style="left:45%; animation-delay:0.6s;">✨</div>
        <div class="sparkle" style="left:65%; animation-delay:0.9s;">✨</div>
        <div class="sparkle" style="left:85%; animation-delay:1.2s;">✨</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Display Prediction ----
    st.success(f"Predicted Profit: ₹{prediction:,.2f}")

    # ---- Feature Contribution Chart ----
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
        color_continuous_scale=['#a5b4fc', '#3b82f6'],
        title="Feature Contribution",
        text_auto=True
    )
    fig1.update_layout(plot_bgcolor='rgba(15,23,42,0)', paper_bgcolor='rgba(15,23,42,0)', font_color='white')
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Profit Comparison Chart ----
    avg_profit = df["Profit"].mean()
    comparison_df = pd.DataFrame({
        "Category": ["Average Profit", "Predicted Profit"],
        "Profit": [avg_profit, prediction]
    })
    fig2 = px.bar(
        comparison_df,
        x="Category",
        y="Profit",
        color="Category",
        color_discrete_map={"Average Profit":"#a5b4fc", "Predicted Profit":"#3b82f6"},
        text_auto=True,
        title="Profit Comparison"
    )
    fig2.update_layout(plot_bgcolor='rgba(15,23,42,0)', paper_bgcolor='rgba(15,23,42,0)', font_color='white')
    st.plotly_chart(fig2, use_container_width=True)
