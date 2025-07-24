# 🧠 Predictive Modeling in R: Housing Prices & Credit Default Classification

![R](https://img.shields.io/badge/R-Programming-blue?logo=r)
![Project Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Modeling](https://img.shields.io/badge/Analysis-Regression%20%7C%20Classification%20%7C%20Cost--Sensitive-orange)
![Tools](https://img.shields.io/badge/Tools-rpart%20%7C%20glm%20%7C%20ROCR-lightgrey)

> 📁 **Case Study 4 — End-to-End Predictive Modeling in R**

This repository presents an end-to-end modeling analysis using two real-world datasets to explore both regression and classification tasks. The project evaluates real-world challenges such as overfitting, model generalizability, and cost-sensitive decision-making in predictive modeling.

---

## 📌 Overview

| Dataset                | Type          | Goal                              | Algorithms Used                     |
|------------------------|---------------|-----------------------------------|-------------------------------------|
| Boston Housing         | Regression     | Predict housing prices            | CART Regression Tree, Linear Model  |
| German Credit Default  | Classification | Predict credit default risk       | Cost-sensitive CART, Logistic Reg.  |


---

## 🏠 Boston Housing Analysis

### 🎯 Objective  
Predict the **median home value (`medv`)** using a variety of socioeconomic and structural predictors.

### 🔍 Approach  
- Data split: 90% training / 10% test
- Two models trained and evaluated:
  - Regression Tree (CART) using `rpart`
  - Linear Regression using `lm` (excluding `indus` and `age`)

### 📊 Performance Summary

| Model            | In-Sample MSE | Out-of-Sample MSE | Notes                          |
|------------------|---------------|-------------------|--------------------------------|
| Regression Tree  | Lower         | Higher            | Slight overfitting observed    |
| Linear Regression| Moderate      | More stable       | Less variance, more robust     |

### 🧠 Insight  
The regression tree fits the training data very well, but its performance on unseen data degrades — highlighting **model variance**. Linear regression, though more constrained, offers more reliable generalization.

---

## 💳 German Credit Score Modeling

### 🎯 Objective  
Classify whether a customer will **default on credit payment** next month.

### ⚙️ Methodology  
- Data split: 80% training / 20% test
- Target: `default.payment.next.month`
- Features include demographics and credit history
- Categorical variables encoded as factors

### 🛠️ Models Used  
1. **CART Classification Tree (Default)**  
2. **CART with Custom Loss Matrix**  
   - False Negative (missed default): **cost = 5**
   - False Positive (false alert): **cost = 1**
3. **Logistic Regression**  
   - Evaluated using the same cost framework  
   - Cutoff = 1/6 based on cost ratio

### 📈 Evaluation Metrics

| Model                       | Test Cost (lower = better) | AUC        | Comments                           |
|----------------------------|----------------------------|------------|------------------------------------|
| Default CART               | Higher                     | ~0.68      | Ignores cost imbalance             |
| Cost-sensitive CART        | ✅ Lowest                  | ~0.70      | Reduces costly misclassifications  |
| Logistic Regression        | Moderate                   | ~0.71      | Performs well but needs calibration|

### 📉 ROC Curve  
ROC and AUC were computed using the `ROCR` package.  
AUC scores indicate acceptable discrimination for both tree and logistic models.

---

## 📊 Visual Tools Used
- `rpart.plot` — Decision tree structure visualization
- `ROCR` — ROC curve plotting and AUC calculation
- `table()` — Confusion matrices for each model
- Custom cost function — Quantifies economic risk of misclassification

---

## ✅ Conclusion

This case study demonstrates that **technical accuracy is not always aligned with business utility**:

- In **housing prediction**, linear models can outperform complex trees when generalization matters most.
- In **credit risk modeling**, incorporating **cost asymmetry** leads to smarter decision systems that reflect real-world stakes — such as loss mitigation from false approvals.

By thoughtfully choosing evaluation metrics and considering economic consequences, this project reflects the practical judgment required of data analysts working in applied domains like real estate and finance.

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/case-study-predictive-modeling.git
