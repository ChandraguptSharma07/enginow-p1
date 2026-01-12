# Customer Purchase Prediction

This project analyzes customer data to predict purchase behavior using machine learning classification models.

## Project Structure

- `enginow_p1.ipynb`: The main Jupyter Notebook with code, analysis, and results.
- `customer-data.csv`: The dataset used for training and testing. (download link: https://www.kaggle.com/datasets/gauthamvijayaraj/customer-purchase-behavior-dataset-e-commerce)
- `requirements.txt`: List of Python libraries required to run the project.

## Instructions

1.  **Install Dependencies:**
    Ensure you have Python installed. Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Notebook:**
    Launch Jupyter Notebook and open `enginow_p1.ipynb`:
    ```bash
    jupyter notebook enginow_p1.ipynb
    ```
    Execute the cells sequentially to reproduce the analysis and model training.

## Approach

1.  **Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical features.
2.  **EDA:** Visualizing data distributions and correlations.
3.  **Modeling:** Training Logistic Regression, Decision Tree, Random Forest, and KNN models.
4.  **Evaluation:** Comparing models based on Accuracy, Precision, Recall, and F1-Score.

## Results

The Logistic Regression and Random Forest models performed best, achieving high accuracy in predicting customer purchases. Detailed metrics and confusion matrices are available in the notebook.
