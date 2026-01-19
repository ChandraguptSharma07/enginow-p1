# Hyperparameter Tuning

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import pickle

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

scale = sum(y_train == 0) / sum(y_train == 1)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

xgb = XGBClassifier(scale_pos_weight=scale, random_state=42, use_label_encoder=False, eval_metric='logloss')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(xgb, param_grid, n_iter=30, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best CV AUC: {search.best_score_:.4f}")

best_model = search.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, y_prob):.4f}")

with open('models/xgb_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Tuning complete")
