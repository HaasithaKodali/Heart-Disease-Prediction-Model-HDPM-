# Heart-Disease-Prediction-Model-HDPM-

A machine learning-based **Clinical Decision Support System (CDSS)** designed to assist in the early diagnosis of heart disease using a hybrid pipeline of **DBSCAN**, **SMOTE-ENN**, and **XGBoost**. Built using the **Cleveland Heart Disease dataset**, this project aims to enhance diagnostic accuracy by removing outliers, balancing class distributions, and employing robust classification models.

## 🧠 Project Objective

To develop a prototype HDPM that aids clinicians in diagnosing heart disease at early stages, enabling timely treatment and potentially reducing mortality. This system integrates clustering, resampling, and predictive modeling into a unified ML pipeline.

---

## 🏗️ Architecture Overview

1. **Data Collection** – Utilized the publicly available **Cleveland Heart Disease dataset**.
2. **Preprocessing**:
   - Outlier detection and removal with **DBSCAN**
   - Class balancing with **SMOTE-ENN**
3. **Modeling** – Prediction using **XGBoost** classifier
4. **Evaluation** – Compared against models like **Naïve Bayes**, **Logistic Regression**, **Random Forest**, etc.
5. **Visualization** – Insights into data distribution, model predictions, and performance metrics

---

## 📊 Technologies & Libraries Used

- **Languages**: Python
- **ML Libraries**: `xgboost`, `scikit-learn`, `eli5`, `shap`
- **Data Manipulation**: `pandas`, `numpy`
- **Clustering & Resampling**: `DBSCAN`, `imblearn` (SMOTE-ENN)
- **Visualization**: `matplotlib`, `seaborn`
- **Tools**: Jupyter Notebook, PyCharm

---

## 📁 Dataset

- **Source**: [Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Attributes**: 13 selected features including age, cholesterol, blood pressure, chest pain type, etc.
- **Samples**: 297 (after removing incomplete records)

---

## ⚙️ Implementation Highlights

- **DBSCAN**: Detected and removed noise/outliers from the dataset.
- **SMOTE-ENN**: Balanced the training data by oversampling the minority class and removing ambiguous samples.
- **XGBoost**: Trained on the refined dataset for heart disease prediction.
- **Cross-Validation**: Applied **10-fold cross-validation** to ensure generalization and avoid overfitting.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC Curve.

---

## 📈 Model Comparison

| Model               | Accuracy |
|--------------------|----------|
| XGBoost            | ~91%     |
| Logistic Regression| ~85%     |
| Random Forest      | ~88%     |
| Naïve Bayes        | ~82%     |

---

## 📊 Key Insights

- **Males** had a 30.7% higher incidence rate than females in the dataset.
- Individuals aged **40–65** are at higher risk.
- Optimal **cholesterol** level: < 200 mg/dL  
- Optimal **blood pressure**: < 120 mmHg  

---

## 📌 Visualizations

- Outlier detection using DBSCAN
- Data distribution before and after SMOTE-ENN
- Heart disease prevalence by gender and age
- Histograms and pie charts for patient-level insights

---

## 👥 Authors

- **Haasitha Kodali**  
  California State University, East Bay  
  [Email](mailto:kodalihaasitha@gmail.com)

- **Komali Pyla**  
  California State University, East Bay  
  [Email](mailto:komalipyla0608@gmail.com)

---

## 📜 References

- [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Various published studies on hybrid ML techniques for heart disease prediction (see project report for full reference list)

---

## 🧪 Future Enhancements

- Deploy HDPM as a web-based clinical support tool
- Integrate real-time patient data via EHR APIs
- Add explainability via SHAP/ELI5 for model transparency

---

## 📄 License

This project is for educational and research purposes only.

