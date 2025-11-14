from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
data = pd.read_csv("C:/Users/ppooj/Downloads/smart_agriculture_sample_dataset.csv")
data['Crop_Type'] = data['Crop_Type'].astype('category').cat.codes
data = data.dropna()

# Feature Selection
X = data.drop('Yield', axis=1)
y = data['Yield']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    temperature = float(request.form['temperature'])
    rainfall = float(request.form['rainfall'])
    humidity = float(request.form['humidity'])
    soil_ph = float(request.form['soil_ph'])
    crop_type = int(request.form['crop_type'])  # Assuming crop type is already encoded
    fertilizer_used = int(request.form['fertilizer_used'])

    # Prepare the input data in the same format as your model expects
    input_data = np.array([[temperature, rainfall, humidity, soil_ph, crop_type, fertilizer_used]])

    # Predict yield
    prediction = model.predict(input_data)
    predicted_yield = prediction[0]

    return render_template('index.html', prediction=predicted_yield)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
