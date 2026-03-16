"""
Advertising Sales Predictor — Multiple Linear Regression
==========================================================
Predicts product sales based on TV, Radio, and Newspaper advertising budgets.
Includes full EDA, correlation analysis, model training, evaluation,
and actual vs predicted visualisation.

Dataset: Advertising.csv
Features: TV, radio, newspaper
Target:   sales

Steps:
    1.  Load dataset
    2.  Remove unwanted columns
    3.  Check missing values
    4.  Statistical summary (EDA)
    5.  Correlation matrix
    6.  Separate features (X) and target (Y)
    7.  Train/test split (80/20)
    8.  Train Linear Regression model
    9.  Predict on test set
    10. Evaluate: MSE, RMSE, R²
    11. Model coefficients and intercept
    12. Compare actual vs predicted values
    13. Scatter plot — Actual vs Predicted
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BORDER = "-" * 50


def build_advertising_model(data_path):
    # --------------------------------------------------
    # Step 1: Load Dataset
    # --------------------------------------------------
    print(BORDER)
    print("Step 1 — Load Dataset")
    print(BORDER)
    df = pd.read_csv(data_path)
    print(df.head())

    # --------------------------------------------------
    # Step 2: Remove Unwanted Columns
    # --------------------------------------------------
    print(BORDER)
    print("Step 2 — Remove unwanted columns")
    print(BORDER)
    print("Shape before:", df.shape)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    print("Shape after :", df.shape)

    # --------------------------------------------------
    # Step 3: Check Missing Values
    # --------------------------------------------------
    print(BORDER)
    print("Step 3 — Check missing values")
    print(BORDER)
    print(df.isnull().sum())

    # --------------------------------------------------
    # Step 4: Statistical Summary (EDA)
    # --------------------------------------------------
    print(BORDER)
    print("Step 4 — Statistical Summary")
    print(BORDER)
    print(df.describe())

    # --------------------------------------------------
    # Step 5: Correlation Matrix
    # --------------------------------------------------
    print(BORDER)
    print("Step 5 — Correlation Matrix")
    print(BORDER)
    print(df.corr())

    # --------------------------------------------------
    # Step 6: Separate Features (X) and Target (Y)
    # --------------------------------------------------
    print(BORDER)
    print("Step 6 — Separate features and target")
    print(BORDER)
    X = df[["TV", "radio", "newspaper"]]
    Y = df["sales"]
    print("X shape:", X.shape, " | Y shape:", Y.shape)

    # --------------------------------------------------
    # Step 7: Train / Test Split (80/20)
    # --------------------------------------------------
    print(BORDER)
    print("Step 7 — Train/test split")
    print(BORDER)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print("X_train:", X_train.shape, " | X_test:", X_test.shape)

    # --------------------------------------------------
    # Step 8: Train the Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 8 — Train Linear Regression model")
    print(BORDER)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    print("Model trained.")

    # --------------------------------------------------
    # Step 9: Predict on Test Set
    # --------------------------------------------------
    Y_pred = model.predict(X_test)

    # --------------------------------------------------
    # Step 10: Evaluate the Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 10 — Model Evaluation")
    print(BORDER)
    mse  = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(Y_test, Y_pred)
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # --------------------------------------------------
    # Step 11: Model Coefficients
    # --------------------------------------------------
    print(BORDER)
    print("Step 11 — Coefficients")
    print(BORDER)
    for col, coef in zip(X.columns, model.coef_):
        print(f"  {col:12s}: {coef:.4f}")
    print(f"  Intercept   : {model.intercept_:.4f}")

    # --------------------------------------------------
    # Step 12: Actual vs Predicted Comparison
    # --------------------------------------------------
    print(BORDER)
    print("Step 12 — Actual vs Predicted (first 10 rows)")
    print(BORDER)
    result = pd.DataFrame({
        "Actual Sales":    Y_test.values,
        "Predicted Sales": Y_pred
    })
    print(result.head(10))

    # --------------------------------------------------
    # Step 13: Scatter Plot — Actual vs Predicted
    # --------------------------------------------------
    print(BORDER)
    print("Step 13 — Scatter plot")
    print(BORDER)
    plt.figure(figsize=(8, 5))
    plt.scatter(Y_test, Y_pred, alpha=0.7)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales — Linear Regression")
    plt.grid(True)
    plt.show()


def main():
    build_advertising_model("Advertising.csv")


if __name__ == "__main__":
    main()
