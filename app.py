from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load trained model and scaler once at startup ---
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Crop and Soil dictionaries ---
crop_dict = {
    1: 'rice', 2: 'wheat', 3: 'pomegranate', 4: 'papaya', 5: 'orange',
    6: 'muskmelon', 7: 'mango', 8: 'apple', 9: 'grapes', 10: 'banana',
    11: 'Chickpea', 12: 'Pigeon Peas', 13: 'Black gram', 14: 'Coconut',
    15: 'Moth Beans', 16: 'Kidney Beans', 17: 'Tobacco', 18: 'Sugarcane',
    19: 'Rubber', 20: 'Peas', 21: 'Ground Nut', 22: 'Cotton', 23: 'Coffee',
    24: 'Jute', 25: 'Lentil', 26: 'maize', 27: 'millet', 28: 'Tea',
    29: 'Mung Bean', 30: 'watermelon'
}

soil_dict = {
    'Loamy Soil': 1,
    'Peaty Soil': 2,
    'Acidic Soil': 3,
    'Neutral Soil': 4,
    'Alkaline Soil': 5
}

# --- Homepage ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1️⃣ Get values from form
        features_list = ['Temperature', 'Humidity', 'Rainfall', 'PH',
                         'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon']
        input_values = [float(request.form[field]) for field in features_list]
        
        soil_name = request.form['Soil']
        soil_numeric = soil_dict.get(soil_name, 1)
        input_values.append(soil_numeric)
        
        # 2️⃣ Convert to DataFrame (preserves feature names for scaler)
        input_df = pd.DataFrame([input_values], columns=features_list + ['Soil_Code'])
        
        # 3️⃣ Scale input
        input_scaled = scaler.transform(input_df)
        
        # 4️⃣ Predict crop
        pred_code = model.predict(input_scaled)[0]
        pred_crop = crop_dict.get(pred_code, "Unknown Crop")
        
        return render_template('index.html', result=pred_crop)
    
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)