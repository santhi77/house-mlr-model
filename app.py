import streamlit as st
import pickle
import numpy as np
import pathlib

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image : url("https://img.freepik.com/free-photo/ai-technology-brain-background-digital-transformation-concept_53876-124674.jpg");
background-size: cover;
}
[data-testid="stSidebar"] {
background-image : url("https://www.usrisk.com/siteassets/images/sidebar-background.jpg?v=492e8c");

}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Load the saved model
model = pickle.load(open(r"D:\Users\SANTHI\Desktop\streamlit\MLR-Apps\house_price_model.pkl", "rb"))

st.sidebar.title("Details of this project.")
st.sidebar.markdown("This app is used for predicting the house prices by using Multiple Linear Regression.")
st.sidebar.write("Multiple Linear Regression (MLR) is a statistical technique used to model the relationship between one dependent variable (Y) and two or more independent variables (X1, X2, X3, etc.).")
st.sidebar.write("The goal is to find the linear equation that best predicts the dependent variable based on the independent variables.")
st.sidebar.html("<p>In this model Dapendent Feature is <bold>Price</bold>.so, I used Regression Algorithm.And Independent Variables are bedrooms, bathrooms, sqft_living, floors, sqft_lot.</p>")

# Streamlit app
st.title("Real Estate Price Predictor")
st.write("This app predicts house prices based on the number of bedrooms, bathrooms, square footage, number of floors, and lot size.")

# Input fields for independent variables
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, step=0.25, value=2.0)
sqft_living = st.number_input("Square Footage of Living Space", min_value=0, step=50, value=2000)
floors = st.number_input("Number of Floors", min_value=0.0, step=0.5, value=1.0)
sqft_lot = st.number_input("Square Footage of Lot", min_value=0, step=50, value=5000)

# Predict button
if st.button("Predict Price"):
    # Prepare input for prediction
    input_data = np.array([[bedrooms, bathrooms, sqft_living, floors, sqft_lot]])
    prediction = model.predict(input_data)[0]
    st.write(f"The predicted price of the house is: ${prediction:,.2f}")


