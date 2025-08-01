import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# Load the trained model
model = joblib.load("house_price_model.pkl")
# Title
st.title("🏡 House Price Prediction App")
st.write("Fill in the details below to estimate the house price.")

# Form for user input
with st.form("house_form"):
    st.subheader("🔢 Enter House Details")

    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=1500)
    sqft_lot = st.number_input("Lot Area (sqft)", min_value=1000, max_value=20000, value=5000)
    floors = st.number_input("Floors", min_value=1.0, max_value=3.5, value=1.0, step=0.5)
    waterfront = st.selectbox("Waterfront View", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    view = st.slider("View Rating (0–4)", 0, 4, 0)
    condition = st.slider("Condition Rating (1–5)", 1, 5, 3)
    sqft_above = st.number_input("Sqft Above", min_value=500, max_value=10000, value=1500)
    sqft_basement = st.number_input("Sqft Basement", min_value=0, max_value=5000, value=0)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
    yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2025, value=0)

    # Submit button inside the form
    submitted = st.form_submit_button("💡 Predict")

# Handle prediction
if submitted:
    input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                            waterfront, view, condition, sqft_above, sqft_basement,
                            yr_built, yr_renovated]])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🏠 Estimated House Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


