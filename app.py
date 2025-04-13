import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.keras')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('onehot_geo_encoder.pkl', 'rb') as f:
    onehot_geo_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)


st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', onehot_geo_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.slider('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0, max_value=100000, value=50000)
num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4])
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0, max_value=150000, value=50000)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=700)

if st.button("Submit"):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
    })

    geography_encoded = onehot_geo_encoder.transform(input_data[['Geography']]).toarray()
    geography_encoded_df = pd.DataFrame(data=geography_encoded, columns=onehot_geo_encoder.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)
    input_data.drop(columns=['Geography'], inplace=True)
    # st.write(input_data.columns)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]
    prediction_class = 'Churn' if prediction_probability > 0.5 else 'Not Churn'

    st.write(f"The customer is likely to {prediction_class}.")
    st.write(f"Prediction Probability: {prediction_probability:.2f}")