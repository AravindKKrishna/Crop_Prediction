# 🌾 Crop Recommendation System using Machine Learning

A Machine Learning–based Crop Recommendation System that predicts the most suitable crop to cultivate based on soil nutrients and climatic conditions.  
The system uses a **Random Forest Classifier** and is deployed using an interactive **Streamlit web application**.

---

## 📌 Project Overview

Choosing the right crop according to soil and environmental conditions plays a vital role in improving agricultural productivity.  
This project analyzes soil nutrients and climate parameters to recommend the best crop using Machine Learning techniques.

---

## 🎯 Objectives

- Analyze soil and climatic factors affecting crop growth  
- Build a high-accuracy crop recommendation model  
- Interpret predictions using feature importance  
- Deploy the model through a user-friendly Streamlit interface  

---

## 📊 Dataset Description

- **Dataset Name:** Crop Recommendation Dataset  
- **Total Records:** 2200  
- **Input Features:**
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature (°C)
  - Humidity (%)
  - Soil pH
  - Rainfall (mm)
- **Target Variable:** Crop label (multiclass)

✔ The dataset is clean, balanced, and contains no missing values.

---

## 🔍 Exploratory Data Analysis (EDA)

- No missing or duplicate values  
- Balanced distribution of crop classes  
- No strong multicollinearity between features  
- Rainfall and soil nutrients significantly influence crop prediction  
- Outliers observed are natural and retained  

---

## 🧠 Machine Learning Model

- **Algorithm Used:** Random Forest Classifier  
- **Why Random Forest?**
  - Handles non-linear relationships effectively  
  - Works well with tabular data  
  - Does not require feature scaling  
  - Reduces overfitting using ensemble learning  

---

## 📈 Model Performance

- **Accuracy:** **99.5% (0.995)**  
- **Train–Test Split:** 80% training, 20% testing  
- **Confusion Matrix:** Near-perfect diagonal with minimal misclassification  

This indicates excellent predictive capability and strong generalization.

---

## 📊 Feature Importance

The Random Forest model provides feature importance scores, helping interpret predictions.  
Highly influential features include:
- Rainfall  
- Nitrogen  
- Potassium  

These results align well with real-world agricultural knowledge.

---

## 🌐 Web Application (Streamlit)

The Streamlit frontend allows users to:
- Input soil and climate parameters  
- Receive real-time crop recommendations  
- View feature importance visualization  

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries & Tools:**
  - Pandas, NumPy  
  - Scikit-learn  
  - Matplotlib  
  - Joblib  
  - Streamlit  

---

