import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model and scaler
model = joblib.load('ann_model.joblib')
scaler = joblib.load('scaler.joblib')

# Title of the app
st.title("Breast Cancer Prediction Using ANN")
st.write("This is an interactive web application that predicts whether breast cancer is malignant or benign using an Artificial Neural Network (ANN).")

# Sidebar for feature inputs
st.sidebar.header('Input Features')

# Collect user input for the features with default values set to zero
mean_texture = st.sidebar.slider('Mean Texture', 0.0, 40.0, 0.0, help="Mean of the texture of the cell.")
mean_perimeter = st.sidebar.slider('Mean Perimeter', 40.0, 200.0, 0.0, help="Mean of the perimeter of the cell.")
mean_concavity = st.sidebar.slider('Mean Concavity', 0.0, 1.5, 0.0, help="Mean of the concavity of the cell.")
mean_concave_points = st.sidebar.slider('Mean Concave Points', 0.0, 0.2, 0.0, help="Mean of the concave points of the cell.")
worst_radius = st.sidebar.slider('Worst Radius', 5.0, 30.0, 0.0, help="Worst radius of the cell.")
worst_texture = st.sidebar.slider('Worst Texture', 10.0, 40.0, 0.0, help="Worst texture of the cell.")
worst_perimeter = st.sidebar.slider('Worst Perimeter', 50.0, 250.0, 0.0, help="Worst perimeter of the cell.")
worst_area = st.sidebar.slider('Worst Area', 200.0, 2000.0, 0.0, help="Worst area of the cell.")
worst_concavity = st.sidebar.slider('Worst Concavity', 0.0, 1.5, 0.0, help="Worst concavity of the cell.")
worst_concave_points = st.sidebar.slider('Worst Concave Points', 0.0, 0.2, 0.0, help="Worst concave points of the cell.")

# Store the user input as a list
user_input = np.array([mean_texture, mean_perimeter, mean_concavity, mean_concave_points,
                       worst_radius, worst_texture, worst_perimeter, worst_area, worst_concavity, worst_concave_points])

# Feature scaling using the same scaler that was used during training
user_input_scaled = scaler.transform([user_input])

# Display the input values in a neat format
st.subheader('Input Features\n')
st.write(f"**1. Mean Texture:** {mean_texture}")
st.write(f"**2. Mean Perimeter:** {mean_perimeter}")
st.write(f"**3. Mean Concavity:** {mean_concavity}")
st.write(f"**4. Mean Concave Points:** {mean_concave_points}")
st.write(f"**5. Worst Radius:** {worst_radius}")
st.write(f"**6. Worst Texture:** {worst_texture}")
st.write(f"**7. Worst Perimeter:** {worst_perimeter}")
st.write(f"**8. Worst Area:** {worst_area}")
st.write(f"**9. Worst Concavity:** {worst_concavity}")
st.write(f"**10. Worst Concave Points:** {worst_concave_points}")

# Prediction when user clicks the button
if st.sidebar.button('Predict'):
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)

    # Display result with more user-friendly output
    st.subheader('Prediction Result:')
    if prediction == 0:
        st.success("Malignant (Cancerous)", icon="⚠️")
    else:
        st.success("Benign (Non-cancerous)", icon="✅")

    # Display probability
    st.subheader('Prediction Probabilities:')
    st.write(f"**Probability of Malignant:** {prediction_proba[0][0]:.2f}")
    st.write(f"**Probability of Benign:** {prediction_proba[0][1]:.2f}")
