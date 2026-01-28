# Customer Churn Prediction 📉

Predicting telecom customer churn using Machine Learning (XGBoost).

## Key Findings 🔍
*   **Contract Type** is the #1 Driver. Customers on **Month-to-month** contracts are extremely likely to churn.
*   **Tenure**: New customers (0-12 months) are the most volatile.
*   **Fiber Optic**: High usage of Fiber internet correlates with higher churn (likely due to price/service issues).

## How to Run it 🏃‍♂️

1.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**
    ```bash
    streamlit run app/app.py
    ```

## Project Structure
*   `data/`: Contains the Telco dataset.
*   `notebooks/`:
    *   `eda.ipynb`: Initial exploration.
    *   `feature_engineering.ipynb`: Creating new features.
    *   `churn_analysis.ipynb`: Visualizing the data.
    *   `modeling.ipynb`: Training XGBoost/RF models.
    *   `interpretation.ipynb`: SHAP analysis (The "Why").
*   `src/`: Refactored cleaning code (`preprocessing.py`).
*   `models/`: Saved model files.
