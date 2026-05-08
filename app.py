import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- LOAD MODEL & DATA ----------------
lr_model = pickle.load(open("lr_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

data = pd.read_csv("data.csv")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Real Estate Predictor",
    layout="wide"
)

# ---------------- MODERN DARK UI ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1120, #111827, #0b1120);
    color: white;
    font-family: 'Segoe UI';
}

.glass {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
}

.stButton button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🏡 AI Real Estate Price Predictor")
st.caption("Modern ML-powered housing intelligence system")

st.image(
    "https://images.unsplash.com/photo-1560518883-ce09059eeffa",
    use_container_width=True
)

# ---------------- INPUT ----------------
st.markdown("## Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    area = st.text_input("Area (sq ft)", "1500")

with col2:
    bedrooms = st.text_input("Bedrooms", "3")

with col3:
    bathrooms = st.text_input("Bathrooms", "2")

# ---------------- SAFE CONVERSION ----------------
def safe(x):
    try:
        return float(x)
    except:
        return None

area_val = safe(area)
bed_val = safe(bedrooms)
bath_val = safe(bathrooms)

# ---------------- PREDICTION ----------------
st.markdown("## Prediction")

if st.button("Predict Price 🚀"):
    if None in (area_val, bed_val, bath_val):
        st.error("Please enter valid numeric values.")
    else:
        input_data = np.array([[area_val, bed_val, bath_val]])
        input_scaled = scaler.transform(input_data)

        lr_pred = lr_model.predict(input_scaled)[0]
        rf_pred = rf_model.predict(input_scaled)[0]

        st.markdown("### 💰 Estimated Price Range")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"Linear Model: ₹ {int(lr_pred):,}")

        with col2:
            st.info(f"Random Forest: ₹ {int(rf_pred):,}")

        avg = (lr_pred + rf_pred) / 2
        st.markdown(f"### ⭐ Final Estimated Price: ₹ {int(avg):,}")

# ---------------- VISUALIZATION ----------------
st.markdown("## 📊 Market Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Area vs Price Trend")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(data["area"], data["price"], alpha=0.5)

    z = np.polyfit(data["area"], data["price"], 1)
    p = np.poly1d(z)
    ax.plot(data["area"], p(data["area"]), color="red")

    ax.set_xlabel("Area")
    ax.set_ylabel("Price")

    st.pyplot(fig, use_container_width=True)

with col2:
    st.markdown("### Feature Correlation")

    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        ax=ax2
    )

    st.pyplot(fig2, use_container_width=True)

# ---------------- DATA PREVIEW ----------------
st.markdown("## 📁 Dataset Preview")

st.dataframe(data, use_container_width=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("About")
st.sidebar.info("""
AI Real Estate Predictor

Features:
• Clean ML pipeline  
• Scaled inputs  
• Realistic dataset  
• Dual model comparison  
• Modern UI dashboard  
""")