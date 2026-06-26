# 🏦 Bank Transaction Propensity Modeling
### End-to-End Machine Learning Pipeline | Binary Classification on Imbalanced Financial Data

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red?logo=tensorflow)](https://tensorflow.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🛠️ Repository Quick Links

If GitHub takes too long to load the full Jupyter Notebook preview, use these direct links to view the interactive analysis or jump straight into the production script:

* 📓 **[Data Exploration Notebook](./Bank_Transaction_Propensity_Modeling.ipynb):** Step-by-step EDA, feature engineering, and model training. *(If GitHub fails to load the preview, **[click here to view it instantly on Google Colab](https://colab.research.google.com/github/puravi-predicts/bank-transaction-propensity-modeling/blob/main/Bank_Transaction_Propensity_Modeling.ipynb)**)*.
* 🐍 **[Production Pipeline Script](./pipeline.py):** Clean, code-only Python script executing the modular model tuning and training pipeline instantly on GitHub.

---

## 📌 Business Objective

Financial institutions lose significant revenue when potential transacting customers go unidentified. This project builds a **production-grade propensity model** to predict whether a bank customer will execute a future transaction — enabling proactive engagement, targeted outreach, and revenue capture.

> **Domain:** Banking & Financial Services  
> **Problem Type:** Binary Classification on severely imbalanced data (90:10 ratio)  
> **Dataset:** Santander Customer Transaction Dataset — 200,000 customers × 200 anonymized features  

---

## 🏗️ Pipeline Architecture

```
Raw Data (200K rows × 202 cols)
        │
        ▼
┌─────────────────────────────┐
│  Phase 1: EDA               │  Target distribution, KDE plots,
│                             │  correlation heatmap, outlier analysis
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 2: Preprocessing     │  Train/Test split → IQR capping →
│                             │  StandardScaler → SMOTE (train only)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 3: Model Building    │  9 algorithms trained and evaluated
│                             │  Linear → Tree → Ensemble → ANN
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 4: Tuning            │  RandomizedSearchCV + StratifiedKFold
│                             │  on RF, XGBoost, LightGBM
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 5: Conclusion        │  Model comparison, business insights,
│                             │  challenges & solutions
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Advanced Extensions        │  SHAP Explainability,
│                             │  Threshold Optimization,
│                             │  Cost-Benefit Analysis
└─────────────────────────────┘
```

---

## 📊 Models Trained & Evaluated

| # | Model | Type | Imbalance Strategy |
|---|-------|------|-------------------|
| 1 | Logistic Regression | Linear | `class_weight='balanced'` |
| 2 | Linear SVM | Linear | `class_weight='balanced'` + Platt Scaling |
| 3 | Decision Tree | Tree | `class_weight='balanced'` |
| 4 | Random Forest | Ensemble | `class_weight='balanced'` |
| 5 | AdaBoost | Ensemble | SMOTE |
| 6 | Gradient Boosting | Ensemble | SMOTE |
| 7 | XGBoost | Gradient Boosting | `scale_pos_weight` |
| 8 | LightGBM | Gradient Boosting | `is_unbalance=True` |
| 9 | ANN (MLP) | Deep Learning | `class_weight` in loss |
| 10 | Random Forest (Tuned) | Ensemble | RandomizedSearchCV |
| 11 | XGBoost (Tuned) | Gradient Boosting | RandomizedSearchCV |
| 12 | LightGBM (Tuned) ⭐ | Gradient Boosting | RandomizedSearchCV |

---

## 🏆 Results

> **Primary Metrics: F1-Score & PR-AUC** (Accuracy is misleading on 90:10 imbalanced data)

| Model | F1-Score | PR-AUC | ROC-AUC |
|-------|----------|--------|---------|
| LightGBM (Tuned) ⭐ | **Best** | **Best** | **Best** |
| XGBoost (Tuned) | 2nd | 2nd | 2nd |
| Random Forest (Tuned) | 3rd | 3rd | 3rd |
| Linear SVM | High Accuracy ⚠️ | Low PR-AUC ⚠️ | Misleading ⚠️ |

> ⚠️ **The Accuracy Paradox:** Linear SVM achieves 91% accuracy but only 0.37 F1-Score and 0.50 PR-AUC — barely better than random for identifying transacting customers. A dummy classifier predicting Class 0 always scores ~90% accuracy on this dataset.


<img width="917" height="662" alt="Screenshot 2026-06-02 141838" src="https://github.com/user-attachments/assets/053e5e69-9f92-4f74-a087-02240d2935d1" />
<img width="1446" height="544" alt="Screenshot 2026-06-02 142000" src="https://github.com/user-attachments/assets/e5c04dc1-f2b4-4dc6-9d5e-f1d7781f654a" />

## Business Impact

Using the tuned LightGBM propensity model and threshold optimization framework:

* Improved identification of likely transacting customers compared to baseline approaches.
* Reduced wasted marketing outreach through probability-based targeting.
* Enabled customer prioritization using expected economic value rather than default classification thresholds.
* Demonstrated how machine learning can support revenue growth and campaign efficiency in retail banking environments.

---

## 🔬 Key Technical Highlights

### ✅ Leak-Free Preprocessing Pipeline
```
Split data first → Fit scaler on X_train only → Transform both → Apply SMOTE on X_train only
```
Never fit any transformer on the full dataset — a common mistake that inflates test scores.

### ✅ Correct SVM Probability Calibration
```python
# ❌ Wrong: min-max scaling of decision function scores
# ✅ Correct: Platt scaling via CalibratedClassifierCV
svm_clf = CalibratedClassifierCV(LinearSVC(...), cv=3, method='sigmoid')
```

### ✅ Multiple Imbalance Strategies
Different models use different strategies — not a one-size-fits-all SMOTE approach:
- Tree ensembles → `class_weight='balanced'`
- XGBoost → `scale_pos_weight` (ratio of negatives to positives)
- LightGBM → `is_unbalance=True`
- ANN → `class_weight` passed to `model.fit()`

---

## 🚀 Advanced Extensions

### 1. SHAP Explainability
Individual customer-level prediction explanations using TreeSHAP — satisfies model transparency requirements in regulated banking environments.

```python
explainer   = shap.TreeExplainer(final_production_lgb)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values[1], X_test_sample)   # Global feature importance
shap.plots.waterfall(explanation)                   # Per-customer explanation
```

### 2. Threshold Optimization
Default 0.5 threshold is suboptimal for imbalanced data. Swept thresholds 0.10 → 0.90 to find the F1-maximizing decision boundary.

```python
best_threshold = max(thresholds, key=lambda t: f1_score(y_test, probs >= t))
```

### 3. Business Cost-Benefit Analysis
Assigned financial values to each prediction outcome and found the **economically optimal threshold** — distinct from both the default and F1-optimal thresholds.

```
True Positive  → +₹5,000  (revenue captured)
False Positive →  -₹500   (wasted outreach)
False Negative → -₹2,000  (missed revenue)
True Negative  →    ₹0    (no cost)
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy`, `scipy` |
| Visualization | `matplotlib`, `seaborn` |
| ML Models | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep Learning | `tensorflow`, `keras` |
| Imbalanced Data | `imbalanced-learn` (SMOTE) |
| Explainability | `shap` |
| Model Persistence | `joblib` |

---

## 📁 Repository Structure

```
bank-transaction-propensity-modeling/
│
├── Bank_Transaction_Propensity_Modeling.ipynb       # Main notebook
├── lgb_propensity_model_v1.pkl                      # Saved best model
├── scaler.pkl                                       # Fitted StandardScaler
├── optimal_threshold.pkl                            # Economically optimal threshold
├── requirements.txt                                 # All dependencies
└── README.md                                        # This file
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/puravi-predicts/bank-transaction-propensity-modeling.git
cd bank-transaction-propensity-modeling<img width="917" height="662" alt="Screenshot 2026-06-02 141838" src="https://github.com/user-attachments/assets/d1cbb47e-3a70-4985-9338-03d4b5c387ee" />


# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
# Place train.csv in the root directory

# 4. Launch the notebook
jupyter notebook Bank_Transaction_Propensity_Modeling.ipynb
```

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.26
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=2.0
lightgbm>=4.0
tensorflow>=2.13
shap>=0.44
joblib>=1.3
scipy>=1.11
```

---

## 💡 Business Insights

1. **Accuracy is not the metric** — on 90:10 imbalanced data, always evaluate using F1-Score and PR-AUC
2. **Gradient boosting dominates** — LightGBM and XGBoost consistently outperform linear and deep learning models on anonymized tabular financial data due to additive error correction and native feature subsampling
3. **Threshold matters more than model** — optimizing the decision threshold from 0.5 to the economically optimal value generates more business value than switching model architectures
4. **Explainability is non-negotiable in banking** — SHAP waterfall plots satisfy RBI and Basel III model transparency requirements; a black-box model alone is not production-acceptable in regulated finance

---

## 👤 Author

**Puravi Pradhan**  
Data Science & Machine Learning  
📧 puravipradhan15@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/puravi-pradhan-593465383) | [GitHub](https://github.com/puravi-predicts)

---

## 📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
---

*Dataset source: [Santander Customer Transaction Prediction — Kaggle](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)*
