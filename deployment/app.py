import streamlit as st
import pandas as pd 

st.title("Income Inequality Analysis Application")

# Navigation side bar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "EDA", "Prediction"])

if app_mode == "Home":
    st.subheader("Welcome to Income Inequality Analysis App!")
    st.write("""This machine learning project aims to predict individuals' income levels based on demographic and occupational characteristics. By utilizing machine learning, this classification model is expected to assist the government in formulating more targeted and efficient economic policies to reduce existing social disparities.""")

elif app_mode == "EDA":
    st.subheader("Exploratory Data Analysis (EDA)")
    import eda
    eda.run_eda()

elif app_mode == "Prediction":
    st.subheader("Prediction")
    
    # Import the prediction module
    import prediction

    # Get user input from the prediction module
    user_input = prediction.get_user_input()

    # Prediction Button
    if st.button("Prediksi"):
        try:
            # Display user input in table form
            st.subheader("Data Input Pengguna")
            st.table(user_input)  # Display user input as a table

            # Call the run_modelling function from the prediction module
            prediction_result = prediction.run_modelling(user_input)

            st.subheader("Hasil Prediksi")
            if prediction_result[0] == 0:
                st.write("Berdasarkan analisa, individu dengan data tersebut memiliki penghasilan <= 50.000.000 IDR")
            else:
                st.write("Berdasarkan analisa, individu dengan data tersebut memiliki penghasilan > 50.000.000 IDR")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
