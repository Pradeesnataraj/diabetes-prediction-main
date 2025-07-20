import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Sample input: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
# Example: A 30-year-old person with given values
input_data = np.array([[2, 120, 70, 20, 79, 25.0, 0.5, 30]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)
result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
print("Prediction:", result)
