# 🌾 Crop Recommendation System Using Machine Learning

## Introduction
The Crop Recommendation System is a machine learning-based web application that recommends the most suitable crop for cultivation based on environmental and soil parameters. The system considers features like Temperature, Humidity, Rainfall, Soil pH, Nitrogen, Phosphorous, Potassium, Carbon content, and Soil Type to predict the ideal crop for a specific region.

This tool can help farmers, agricultural planners, and agritech enthusiasts make informed decisions, maximize yield, and optimize resources.

---

## Features
- Predicts the most suitable crop from **30 different crop options**.
- Takes both **environmental and soil characteristics** as input.
- **Fast and accurate predictions** using Random Forest Classifier.
- **User-friendly web interface** with a form-based input.
- Deployed using **Flask**.

---

## Dataset
- **Source:** Kaggle crop recommendation dataset.  
- **Number of Samples:** ~3,000 records (depending on dataset used).  
- **Features:**
  1. Temperature – in °C  
  2. Humidity – in %  
  3. Rainfall – in mm  
  4. PH – soil pH  
  5. Nitrogen – in kg/ha  
  6. Phosphorous – in kg/ha  
  7. Potassium – in kg/ha  
  8. Carbon – in %  
  9. Soil – Soil Type (Loamy, Peaty, Acidic, Neutral, Alkaline)  
- **Target:** Crop – the crop label (e.g., rice, wheat, mango, etc.)  
- **Data Cleaning:** Removed inconsistent records, e.g., Adzuki Beans with Temperature > 43°C.

---

## Model
- **Algorithm Used:** Random Forest Classifier  
- **Hyperparameters:**
  - `n_estimators = 100`  
  - `max_depth = 15`  
  - `random_state = 42`  
- **Scaling:** Features are scaled using `StandardScaler` for better performance.  
- **Label Encoding:** Crop names and soil types are mapped to numeric codes.

---

## Evaluation Metrics
- **Test Accuracy:** 96%  

**Classification Report (Sample):**

| Crop Code | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| 1         | 1.00      | 0.90   | 0.95     | 21      |
| 2         | 0.90      | 0.68   | 0.78     | 28      |
| 3         | 1.00      | 0.89   | 0.94     | 18      |
| ...       | ...       | ...    | ...      | ...     |
| 30        | 1.00      | 1.00   | 1.00     | 17      |

**Confusion Matrix Insights:**
- Most crops are classified correctly with minimal misclassification.  
- Some crops like Wheat (2) and Pigeon Peas (12) have slightly lower recall due to misclassification.

---

## Deployment

### 1️⃣ Requirements
- Python 3.9+  
- Libraries:
```bash
pip install flask pandas numpy scikit-learn
crop_recommendation_system/
│
├─ app.py
├─ crop_model.pkl
├─ scaler.pkl
├─ templates/
│   └─ index.html
├─ static/
│   └─ (optional CSS/JS files)
└─ README.md
