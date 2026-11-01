# Machine Learning Classification Model Evaluation

## 1. Problem Statement

The objective of this project is to implement and compare five supervised machine learning classification algorithms on the same dataset.

The models evaluated are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are compared using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

---

## 2. Dataset Description

### Dataset
**Breast Cancer Wisconsin (Diagnostic) Dataset**

### Source
UCI Machine Learning Repository

### Dataset Characteristics

- Total instances: **569**
- Predictive features: **30**
- Target variable: **Diagnosis**
- Classification type: **Binary classification**
- Classes:
  - `B` – Benign
  - `M` – Malignant
- Missing values: **None**

The ID column was removed before model training. The target variable was encoded as:

- `B → 0`
- `M → 1`

The dataset was divided using a stratified **80:20 train-test split** with `random_state=42`.

- Training samples: **455**
- Testing samples: **114**

Feature scaling using `StandardScaler` was applied to Logistic Regression, KNN, and SVM. Decision Tree and Gaussian Naive Bayes were trained using the original feature values.

---

## 3. GitHub Repository

**Repository:**  
https://github.com/abhishekswaraj23/ML_Assignment_2

---

## 4. Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| **SVM** | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

---

## 5. Model-wise Observations

### Logistic Regression

Logistic Regression achieved high overall performance with an accuracy of **96.49%** and the highest AUC of **0.9960**. It provided a good balance between precision and recall.

### Decision Tree

Decision Tree achieved an accuracy of **92.98%**. Its performance was lower than the other models, with an AUC of **0.9246** and MCC of **0.8492**.

### K-Nearest Neighbor

KNN achieved **95.61% accuracy** and an AUC of **0.9823**. It performed well after feature scaling, but its recall was slightly lower than Logistic Regression and SVM.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved **93.86% accuracy** and perfect precision of **1.0000**. However, its recall was **0.8333**, meaning that it missed some malignant cases.

### Support Vector Machine

SVM achieved the highest accuracy of **97.37%**, perfect precision of **1.0000**, recall of **0.9286**, the highest F1 score of **0.9630**, and the highest MCC of **0.9442**.

---

## 6. Overall Winner

**Support Vector Machine (SVM)** is the overall best-performing model for this dataset.

It achieved the highest:

- Accuracy: **97.37%**
- Precision: **1.0000**
- F1 Score: **0.9630**
- MCC: **0.9442**

Although Logistic Regression achieved a slightly higher AUC (**0.9960**) than SVM (**0.9947**), SVM provided the strongest overall performance across the other evaluation metrics.

---

## 7. Streamlit Application

The project includes a Streamlit application for evaluating the trained models.

The application allows the user to:

- Upload test data in CSV format
- Select one of the five trained classification models
- Calculate Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- View the confusion matrix
- View prediction results

The application loads the previously trained models and does **not retrain the models** on uploaded test data.

---

## 8. Project Structure

```text
ML_Assignment_2/
│
├── Assignment_ML.ipynb
├── app.py
├── requirements.txt
├── test_data.csv
├── data/
│   └── original_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── scaler.joblib
```

---

## 9. How to Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload `test_data.csv`, select a model, and view its evaluation metrics and confusion matrix.