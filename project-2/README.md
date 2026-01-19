# Credit Risk Prediction System

ML Capstone Project by Chandragupt Sharma

---

## What is this?

This project predicts if a loan applicant will default or not. Built it as part of my ML capstone - it's an end-to-end system from data analysis to a working web app.

**Try it:** `python -m streamlit run app.py`

---

## Dataset

Got this from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) - about 32K loan records with info like income, loan amount, credit history etc.

Main features used:
- Age, income, employment years
- Loan amount, interest rate, purpose
- Credit history and previous defaults
- Home ownership status

Target: whether they defaulted (1) or paid back (0)

---

## What I did

**1. Data exploration**
- Found 21.8% default rate
- Data was imbalanced (3.6:1 ratio)
- Some weird ages like 144 years old - had to fix those

**2. Preprocessing**
- Filled missing values with median
- Created some new features like risk_score
- Scaled everything with StandardScaler

**3. Tried multiple models**

| Model | ROC-AUC |
|-------|---------|
| XGBoost | 0.951 |
| Random Forest | 0.930 |
| Gradient Boosting | 0.929 |
| Logistic Regression | 0.867 |

XGBoost won, so used that.

**4. Tuned the model**
- Used RandomizedSearchCV with 5-fold CV
- Best params: max_depth=5, learning_rate=0.2

**5. Built a Streamlit app**
- User enters their details
- Shows if they're high or low risk
- Displays probability and risk factors

---

## Files

```
app.py              - web app
src/eda.py          - data analysis
src/preprocess.py   - cleaning
src/train.py        - model training
src/tune.py         - hyperparameter tuning
models/             - saved models
data/               - dataset and processed files
```

---

## How to run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## What I learned

- Handling imbalanced data is tricky - class weights helped
- XGBoost is really good out of the box
- Feature engineering matters more than I expected
- Streamlit makes deployment super easy

---

Made by Chandragupt Sharma
