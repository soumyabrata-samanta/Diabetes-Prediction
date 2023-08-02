import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
data = pd.read_csv("diabetes.csv")

# Preprocessing
X = data.drop(columns="Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# SVM model training
clf = svm.SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# Function to predict diabetes for a given input sample and return confidence score
def predict_diabetes(input_sample):
    input_np_array = np.asarray(input_sample)
    input_np_array_reshaped = input_np_array.reshape(1, -1)
    std_data = scaler.transform(input_np_array_reshaped)
    prediction = clf.predict(std_data)
    confidence_score = np.max(clf.predict_proba(std_data)) * 100
    return prediction[0], confidence_score

# Streamlit app
def main():
    st.title("Diabetes Classification App")

    # Input form
    st.write("Enter the features for prediction:")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=250, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=150, value=30)

    input_sample = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)

    if st.button("Predict"):
        prediction, confidence_score = predict_diabetes(input_sample)
        if prediction == 1:
            st.write("Person is diabetic.")
        else:
            st.write("Person is not diabetic.")
        st.write(f"Confidence Score: {confidence_score:.2f}%")

        # Create DataFrame for chart
        confidence_df = pd.DataFrame({"Confidence Score": [confidence_score]})
        st.bar_chart(confidence_df, height=200)

if __name__ == "__main__":
    main()
