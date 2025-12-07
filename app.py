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

# ---- Custom CSS (professional, pastel, modern UI) ----
st.markdown("""
<style>
/* Background */
body, .stApp {
    background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
    color: #1f2937;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* Headings */
h1, h2, h3, .stMarkdown {
    color: #111827;
    font-weight: 600;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #7f9cf5, #b4c6fc);
    color: #1f2937 !important;
    font-size: 16px;
    padding: 12px 28px;
    border-radius: 8px;
    border: none;
    transition: all 0.25s ease-in-out;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
}

/* Input fields */
.stNumberInput>div>div>input, .stSelectbox>div>div>select {
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    padding: 10px;
    font-size: 16px;
    transition: 0.2s;
}
.stNumberInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
    border-color: #7f9cf5;
    box-shadow: 0 0 5px rgba(127,156,245,0.4);
}

/* Chart backgrounds */
.js-plotly-plot .plotly {
    background-color: rgba(255,255,255,0.9) !important;
    border-radius: 12px;
}

/* Container spacing */
section[data-testid="stSidebar"] {
    padding: 20px;
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
st.markdown("<h2 style='text-align:center; margin-bottom:10px;'>Startup Profit Prediction</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#374151; margin-bottom:30px;'>Enter your startup's financial details to predict expected profit.</p>", unsafe_allow_html=True)

# ---- Inputs ----
col1, col2, col3, col4 = st.columns([2,2,2,2])
with col1:
    rnd = st.number_input("R&D Spend (₹)", value=100000.0, step=5000.0, format="%.2f")
with col2:
    admin = st.number_input("Administration Spend (₹)", value=50000.0, step=5000.0, format="%.2f")
with col3:
    marketing = st.number_input("Marketing Spend (₹)", value=50000.0, step=5000.0, format="%.2f")
with col4:
    city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi"])

# ---- Predict Button ----
if st.button("Predict Profit"):
    # Prediction data
    input_data = {
        "R&D Spend": rnd,
        "Administration": admin,
        "Marketing Spend": marketing,
        "State_Delhi": 1 if city=="Delhi" else 0,
        "State_Mumbai": 1 if city=="Mumbai" else 0
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    # ---- Animated indicator (modern subtle effect) ----
    placeholder = st.empty()
    for i in range(8):
        placeholder.markdown(f"<h3 style='text-align:center; color:#7f9cf5;'>Predicting{'.'*i}</h3>", unsafe_allow_html=True)
        time.sleep(0.1)
    placeholder.empty()

    st.success(f"Predicted Profit: ₹{prediction:,.2f}")

    # ---- Feature Contribution ----
    contributions = model.coef_ * list(input_df.iloc[0])
    contrib_df = pd.DataFrame({"Feature": X.columns, "Contribution": contributions}).sort_values(by="Contribution")

    fig1 = px.bar(
        contrib_df,
        x="Contribution",
        y="Feature",
        orientation="h",
        color="Contribution",
        color_continuous_scale=["#b4c6fc","#7f9cf5"],
        title="Feature Contribution",
        text_auto=True
    )
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Profit Comparison ----
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
        color_discrete_map={"Average Profit":"#b4c6fc","Predicted Profit":"#7f9cf5"},
        text_auto=True,
        title="Profit Comparison"
    )
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)
