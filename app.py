import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---- Page Config ----
st.set_page_config(
    page_title="Startup Profit Predictor",
    layout="wide"
)

# ---- Custom CSS for classy pastel pink ----
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #ffeef8, #ffd9e8);
        color: #4d004d;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, .css-18e3th9 {
        color: #993366;
        font-weight: bold;
    }
    h2 {
        color: #b35980;
    }
    input[type=number], div[role="listbox"] > div {
        background-color:#ffeef8;
        color:#4d004d;
        font-weight:bold;
    }
    div.stButton > button:first-child {
        background-color: #ffb3c6;
        color:white;
        height:3em;
        width:100%;
        border-radius:10px;
        border:none;
        font-size:16px;
        font-weight:bold;
    }
    div.stButton > button:hover {
        background-color:#ff99b3;
        color:white;
    }
    .stAlert {
        background-color:#ffd9e8;
        color:#4d004d;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("🚀 Startup Profit Prediction App")
st.markdown("Enter your startup details below to predict expected profit 💰")

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

# ---- Input Form ----
with st.form("startup_form"):
    st.subheader("Enter Startup Details 🏙️")
    rnd = st.number_input("R&D Spend (₹)", value=100000.0, step=1000.0)
    admin = st.number_input("Administration Spend (₹)", value=50000.0, step=1000.0)
    marketing = st.number_input("Marketing Spend (₹)", value=50000.0, step=1000.0)
    city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi"])
    submitted = st.form_submit_button("Predict Profit 📈")

# ---- Prediction & Output ----
if submitted:
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

    # Feature Contribution
    contributions = model.coef_ * list(input_df.iloc[0])
    contrib_df = pd.DataFrame({
        "Feature": X.columns,
        "Contribution": contributions
    })

    # Average profit comparison
    avg_profit = df["Profit"].mean()
    comparison_df = pd.DataFrame({
        "Category": ["Predicted Profit 💰", "Average Profit 📊"],
        "Profit": [prediction, avg_profit]
    })

    # ---- Columns Layout ----
    st.subheader("Insights 🔍")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Feature Contribution 📈")
        st.bar_chart(contrib_df.set_index("Feature"), use_container_width=True)

    with col2:
        st.markdown("### Profit Comparison 💹")
        st.bar_chart(comparison_df.set_index("Category"), use_container_width=True)

    st.markdown("---")
    st.markdown("This prediction uses a linear regression model trained on 50 startup datasets to estimate profit based on your inputs.")
