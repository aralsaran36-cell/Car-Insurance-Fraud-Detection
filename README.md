# 🚗 Car Insurance Fraud Detection

A machine learning model to detect **fraudulent car insurance claims** on a highly imbalanced dataset. Achieved **96% accuracy** using XGBoost with SMOTE for class balancing.

---

## 🎯 Problem Statement

Insurance fraud costs companies billions every year. This project builds a classification model to automatically identify fraudulent claims, reducing false positives and financial losses.

---

## 📊 Dataset

- **File:** `carclaims.csv`
- **Challenge:** Highly imbalanced dataset (fraud cases are rare)
- **Solution:** SMOTE + oversampling + undersampling techniques

---

## 🧠 How It Works

```
Load imbalanced dataset
        ↓
EDA & Data Preprocessing
        ↓
Handle class imbalance (SMOTE / Oversampling / Undersampling)
        ↓
Train multiple ML models
        ↓
Evaluate & compare performance
        ↓
XGBoost → 96% accuracy, 0.89 precision (fraud class)
```

---

## ✨ Key Results

| Model | Accuracy | Fraud Precision |
|---|---|---|
| Logistic Regression | ~78% | ~0.71 |
| Random Forest | ~91% | ~0.83 |
| **XGBoost** | **96%** | **0.89** |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML Models | Logistic Regression, Random Forest, XGBoost |
| Imbalance Handling | SMOTE, Oversampling, Undersampling |
| Libraries | Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/aralsaran36-cell/Car-Insurance-Fraud-Detection.git
cd Car-Insurance-Fraud-Detection

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

# 3. Open notebook
jupyter notebook Car_Insurance_Fraud_Detection.ipynb
```

---

## 📁 Project Structure

```
Car-Insurance-Fraud-Detection/
│
├── Car_Insurance_Fraud_Detection.ipynb   # Main notebook
├── carclaims.csv                          # Dataset
├── Vehicle Insurance Fraud Detection.pdf # Project report
└── README.md
```

---

## 💡 Key Learnings

- Handling imbalanced datasets using SMOTE
- Comparing multiple ML models on the same dataset
- Optimizing precision for minority class (fraud)
- Feature engineering for tabular data

---

## 👨‍💻 Author

**Saravanan S** — Aspiring AI Engineer
- 📧 aralsaran36@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/feed/)
- 🐙 [GitHub](https://github.com/aralsaran36-cell)
