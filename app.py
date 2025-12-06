import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---- Page Config ----
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="🚀",
    layout="wide"
)

# ---- Custom CSS for Pinterest Aesthetic ----
st.markdown("""
<style>
/* Background & main container */
.stApp {
    background-color: #fff0f5;  /* pastel lavender-pink */
    color: #333333;
    font-family: 'Helvetica', 'Arial', sans-serif;
}

/* Section headers */
h2, h3 {
    color: #d63384;
    font-family: 'Segoe UI', sans-serif;
    margin-bottom: 10px;
}

/* Card styling */
.card {
    background-color: #ffe6f0; 
    padding: 20px;
    border-radius: 20px;
    box-shadow: 3px 3px 15px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

/* Inputs */
.css-1d391kg input, .css-1d391kg select {
    border-radius: 12px;
    border: 1px solid #d63384;
    padding: 8px;
}

/* Metrics styling */
.metric-container {
    background-color: #ffd6e8;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 25px;
}

/* Section divider */
.divider {
    height: 2px;
    background-color: #d63384;
    margin: 30px 0;
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
st.markdown("<h2>🚀 Startup Profit Prediction</h2>", unsafe_allow_html=True)
st.markdown("Enter your startup details below to predict the expected profit 💰")

# ---- Input Section ----
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h3>Enter Startup Details 🏙️</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    rnd = st.number_input("💡 R&D Spend (₹)", value=100000.0, step=1000.0)
with col2:
    admin = st.number_input("🏢 Administration Spend (₹)", value=50000.0, step=1000.0)
with col3:
    marketing = st.number_input("📣 Marketing Spend (₹)", value=50000.0, step=1000.0)

city = st.selectbox("🌆 City", ["Bangalore", "Mumbai", "Delhi"])
st.markdown('</div>', unsafe_allow_html=True)

# ---- Predict Button ----
if st.button("Predict Profit 💹"):
    input_data = {
        "R&D Spend": rnd,
        "Administration": admin,
        "Marketing Spend": marketing,
        "State_Delhi": 1 if city == "Delhi" else 0,
        "State_Mumbai": 1 if city == "Mumbai" else 0
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    # ---- Predicted Profit ----
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown(f"💰 Predicted Profit: ₹{prediction:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Feature Contribution ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Feature Contribution 📊</h3>", unsafe_allow_html=True)
    contributions = model.coef_ * list(input_df.iloc[0])
    feature_names = X.columns
    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    }).sort_values(by="Contribution", ascending=False)
    st.bar_chart(contrib_df.set_index("Feature"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Profit Comparison ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Profit Comparison 💹</h3>", unsafe_allow_html=True)
    avg_profit = df["Profit"].mean()
    comparison_df = pd.DataFrame({
        "Category": ["Predicted Profit 💰", "Average Profit 📊"],
        "Profit": [prediction, avg_profit]
    })
    st.bar_chart(comparison_df.set_index("Category"))
    st.markdown('</div>', unsafe_allow_html=True)
