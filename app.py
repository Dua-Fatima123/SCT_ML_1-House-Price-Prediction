import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .main {
            background-color: #0F1117;
            color: white;
        }
        .title {
            font-size: 45px;
            font-weight: bold;
            color: #FF4B4B;
        }
        .subtitle {
            font-size: 25px;
            color: #FF914D;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            text-align: center;
            width: 100%;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">🏠 House Price Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the price of a house using linear regression</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📋 Enter House Details")

# Input Fields
area = st.sidebar.slider("Total Area (sq. ft)", 500, 10000, step=100)
bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4])

# Predict Button
if st.sidebar.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)[0]

    st.success(f"💰 **Predicted House Price:** ₹ {round(prediction, 2)}")

# Extra Section: Model Info and Sample Graph
with st.expander("📊 Model Information"):
    st.markdown("This model uses Linear Regression to predict house prices based on:")
    st.markdown("- Square footage (`area`)")
    st.markdown("- Number of bedrooms")
    st.markdown("- Number of bathrooms")

    # Optional Sample Data Display
    sample = {
        'Area': [1500, 2200, 3000],
        'Bedrooms': [3, 4, 3],
        'Bathrooms': [2, 3, 3],
        'Expected Price': [model.predict([[1500, 3, 2]])[0],
                           model.predict([[2200, 4, 3]])[0],
                           model.predict([[3000, 3, 3]])[0]]
    }

    df_sample = pd.DataFrame(sample)
    st.dataframe(df_sample)

# Footer
st.markdown('<div class="footer">Developed by Dua | Powered by Streamlit</div>', unsafe_allow_html=True)
