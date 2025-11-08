# 🧠 Surface Roughness Prediction in CNC Turning Operations

**Mini Project – Data Science in Manufacturing**  
**By:** Arjun B Raj | National Institute of Technology, Warangal  

---

## 📘 Project Overview

This project focuses on **predicting surface roughness (Ra)** in CNC turning operations using **Machine Learning**.  
Surface roughness directly affects the **quality, durability, and performance** of machined components.  
By building an interpretable predictive model, this work aims to optimize machining parameters,  
reduce inspection time, and enable **data-driven process control** in manufacturing.

---

## 📂 Dataset

**Source:** [Kaggle – CNC Turning Roughness, Forces, and Tool Wear Dataset](https://www.kaggle.com/datasets/adorigueto/cnc-turning-roughness-forces-and-tool-wear)  
**File Used:** `Exp2.csv`  
**Samples:** ~288 machining runs  

### Features
| Type | Variables |
|------|------------|
| Input | Tool condition (TCond), Depth of cut (ap), Cutting speed (vc), Feed rate (f), Cutting forces (Fx, Fy, Fz), Resultant force (F) |
| Target | Average surface roughness (Ra) |

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Cleaned and removed irrelevant columns
   - Handled missing and duplicate values

2. **Feature Scaling**
   - Applied `MinMaxScaler` for normalization

3. **Model Selection**
   - Used `DecisionTreeRegressor` for interpretability and ease of tuning

4. **Training and Evaluation**
   - Data split: 60% Train, 20% Validation, 20% Test
   - Evaluation metrics: R², MAE, MSE, RMSE, MAPE

---

## 🧠 Model Performance

| Metric | Value |
|---------|--------|
| R² Score | 0.97 |
| MAE | 0.027 |
| RMSE | 0.038 |
| MAPE | 0.045 |
| MSE | 0.001 |

✅ The model explains **97% of the variance** in surface roughness with prediction errors around **±0.038 µm**,  
making it highly suitable for **process optimization and quality monitoring**.

---

## 📊 Results & Insights

- **Feed rate (f)** is the most influential factor affecting Ra.  
- **Tool condition** degradation leads to increased roughness — monitoring tool wear is essential.  
- **Cutting forces (Fx, Fy, Fz)** can be used for **real-time prediction** of surface quality.  

### Sample Prediction
```python
Actual Ra: 0.316
Predicted Ra: 0.3265
Error ≈ 3.32%
🚀 Future Scope

Hyperparameter tuning and ensemble methods for improved accuracy.

Integration with CNC controllers for real-time roughness prediction.

Expand dataset to include varied materials and machining conditions.

🛠️ Tech Stack

Language: Python

Libraries: Pandas, NumPy, scikit-learn, Matplotlib

Tools: Jupyter Notebook / Google Colab

Dataset Source: Kaggle

📎 How to Run
# Clone this repository
git clone https://github.com/<your-username>/CNC-SurfaceRoughness-Prediction.git

# Navigate to the project folder
cd CNC-SurfaceRoughness-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
python surface_roughness_model.py
