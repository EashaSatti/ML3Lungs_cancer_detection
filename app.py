import streamlit as st
import pickle
import numpy as np

# Load the models
try:
    with open('lung_cancer_xgb_model.pkl', 'rb') as file_xgb:
        xgb_model = pickle.load(file_xgb)
except (FileNotFoundError, EOFError) as e:
    st.error(f"Error loading XGBoost model: {e}")

try:
    with open('lung_cancer_svm_model.pkl', 'rb') as file_svm:
        svm_model = pickle.load(file_svm)
except (FileNotFoundError, EOFError) as e:
    st.error(f"Error loading SVM model: {e}")

try:
    with open('lung_cancer_knn_model.pkl', 'rb') as file_knn:
        knn_model = pickle.load(file_knn)
except (FileNotFoundError, EOFError) as e:
    st.error(f"Error loading KNN model: {e}")

# Function to predict lung cancer
def predict_lung_cancer(features, model):
    try:
        prediction = model.predict(np.array(features).reshape(1, -1))
        return "YES" if prediction == 1 else "NO"
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Streamlit app interface
st.title('Lung Cancer Prediction App')

st.write('**Please select the features you want to provide. Leave others empty (default values will be used).**')

# Input form
gender = st.selectbox('Gender', ('Select', 'Male', 'Female'))
age = st.number_input('Age', min_value=1, max_value=100, step=1, value=30)
smoking = st.selectbox('Smoking', ('Select', 'No', 'Yes'))
yellow_fingers = st.selectbox('Yellow Fingers', ('Select', 'No', 'Yes'))
anxiety = st.selectbox('Anxiety', ('Select', 'No', 'Yes'))
peer_pressure = st.selectbox('Peer Pressure', ('Select', 'No', 'Yes'))
chronic_disease = st.selectbox('Chronic Disease', ('Select', 'No', 'Yes'))
fatigue = st.selectbox('Fatigue', ('Select', 'No', 'Yes'))
allergy = st.selectbox('Allergy', ('Select', 'No', 'Yes'))
wheezing = st.selectbox('Wheezing', ('Select', 'No', 'Yes'))
alcohol_consuming = st.selectbox('Alcohol Consuming', ('Select', 'No', 'Yes'))
coughing = st.selectbox('Coughing', ('Select', 'No', 'Yes'))
shortness_of_breath = st.selectbox('Shortness of Breath', ('Select', 'No', 'Yes'))
swallowing_difficulty = st.selectbox('Swallowing Difficulty', ('Select', 'No', 'Yes'))
chest_pain = st.selectbox('Chest Pain', ('Select', 'No', 'Yes'))

# Assign numerical values to features based on selections
gender = 1 if gender == 'Male' else (0 if gender == 'Female' else -1)
smoking = 2 if smoking == 'Yes' else (1 if smoking == 'No' else 1)
yellow_fingers = 2 if yellow_fingers == 'Yes' else (1 if yellow_fingers == 'No' else 1)
anxiety = 2 if anxiety == 'Yes' else (1 if anxiety == 'No' else 1)
peer_pressure = 2 if peer_pressure == 'Yes' else (1 if peer_pressure == 'No' else 1)
chronic_disease = 2 if chronic_disease == 'Yes' else (1 if chronic_disease == 'No' else 1)
fatigue = 2 if fatigue == 'Yes' else (1 if fatigue == 'No' else 1)
allergy = 2 if allergy == 'Yes' else (1 if allergy == 'No' else 1)
wheezing = 2 if wheezing == 'Yes' else (1 if wheezing == 'No' else 1)
alcohol_consuming = 2 if alcohol_consuming == 'Yes' else (1 if alcohol_consuming == 'No' else 1)
coughing = 2 if coughing == 'Yes' else (1 if coughing == 'No' else 1)
shortness_of_breath = 2 if shortness_of_breath == 'Yes' else (1 if shortness_of_breath == 'No' else 1)
swallowing_difficulty = 2 if swallowing_difficulty == 'Yes' else (1 if swallowing_difficulty == 'No' else 1)
chest_pain = 2 if chest_pain == 'Yes' else (1 if chest_pain == 'No' else 1)

# Features array
features = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
            fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath,
            swallowing_difficulty, chest_pain]

# Model selection
model_choice = st.selectbox("Choose the model for prediction", ["XGBoost", "SVM", "KNN"])

# Button for prediction
if st.button('Predict'):
    if gender == -1:  # If gender was not selected
        st.warning("Gender is not selected. Default value is being used.")

    if model_choice == "XGBoost":
        if 'xgb_model' in globals():
            result = predict_lung_cancer(features, xgb_model)
            st.success(f'The prediction is: {result}')
        else:
            st.error("XGBoost model is not loaded.")
    elif model_choice == "SVM":
        if 'svm_model' in globals():
            result = predict_lung_cancer(features, svm_model)
            st.success(f'The prediction is: {result}')
        else:
            st.error("SVM model is not loaded.")
    elif model_choice == "KNN":
        if 'knn_model' in globals():
            result = predict_lung_cancer(features, knn_model)
            st.success(f'The prediction is: {result}')
        else:
            st.error("KNN model is not loaded.")
