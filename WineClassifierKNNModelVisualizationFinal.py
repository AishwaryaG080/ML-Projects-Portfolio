"""
Wine Classifier — KNN with Hyperparameter Tuning
==================================================
Classifies wines into categories using K-Nearest Neighbours.
Tests K values from 1 to 20, plots K vs Accuracy, and builds
a final model using the best K found.

Dataset: WinePredictor.csv
Target column: Class

Steps:
    1.  Load the dataset
    2.  Clean — remove empty rows
    3.  Separate features (X) and target (Y)
    4.  Train/test split (stratified)
    5.  Feature scaling
    6.  Tune K (1–20) and record accuracy
    7.  Plot K vs Accuracy
    8.  Select best K
    9.  Build final model with best K
    10. Evaluate: Accuracy, Confusion Matrix, Classification Report
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

BORDER = "-" * 50


def build_wine_classifier(data_path):
    # --------------------------------------------------
    # Step 1: Load Dataset
    # --------------------------------------------------
    print(BORDER)
    print("Step 1 — Load the dataset")
    print(BORDER)
    df = pd.read_csv(data_path)
    print(df.head())

    # --------------------------------------------------
    # Step 2: Clean — remove rows with missing values
    # --------------------------------------------------
    print(BORDER)
    print("Step 2 — Remove empty rows")
    print(BORDER)
    df.dropna(inplace=True)
    print(f"Records: {df.shape[0]}  |  Columns: {df.shape[1]}")

    # --------------------------------------------------
    # Step 3: Separate Features and Target
    # --------------------------------------------------
    print(BORDER)
    print("Step 3 — Separate features (X) and target (Y)")
    print(BORDER)
    X = df.drop(columns=["Class"])
    Y = df["Class"]
    print("Input columns :", X.columns.tolist())
    print("X shape:", X.shape, " | Y shape:", Y.shape)

    # --------------------------------------------------
    # Step 4: Train / Test Split (stratified, 80/20)
    # --------------------------------------------------
    print(BORDER)
    print("Step 4 — Train/test split")
    print(BORDER)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    print("X_train:", X_train.shape, " | X_test:", X_test.shape)

    # --------------------------------------------------
    # Step 5: Feature Scaling
    # --------------------------------------------------
    print(BORDER)
    print("Step 5 — Feature scaling")
    print(BORDER)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)   # transform only (not fit again)
    print("Scaling complete.")

    # --------------------------------------------------
    # Step 6: Tune K from 1 to 20
    # --------------------------------------------------
    print(BORDER)
    print("Step 6 — Hyperparameter tuning: testing K = 1 to 20")
    print(BORDER)
    accuracy_scores = []
    k_values = range(1, 21)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, Y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy_scores.append(accuracy_score(Y_test, y_pred))

    for k, acc in zip(k_values, accuracy_scores):
        print(f"  K={k:2d}  Accuracy={acc:.4f}")

    # --------------------------------------------------
    # Step 7: Plot K vs Accuracy
    # --------------------------------------------------
    print(BORDER)
    print("Step 7 — K vs Accuracy plot")
    print(BORDER)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracy_scores, marker="o")
    plt.title("K Values vs Accuracy — Wine Classifier")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.xticks(list(k_values))
    plt.grid(True)
    plt.show()

    # --------------------------------------------------
    # Step 8: Select Best K
    # --------------------------------------------------
    best_k = list(k_values)[accuracy_scores.index(max(accuracy_scores))]
    print(BORDER)
    print(f"Step 8 — Best K = {best_k}")
    print(BORDER)

    # --------------------------------------------------
    # Step 9: Build Final Model with Best K
    # --------------------------------------------------
    print(BORDER)
    print(f"Step 9 — Final model using K={best_k}")
    print(BORDER)
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train_scaled, Y_train)
    Y_pred = final_model.predict(X_test_scaled)

    # --------------------------------------------------
    # Step 10: Evaluate Final Model
    # --------------------------------------------------
    print(BORDER)
    print("Step 10 — Evaluation")
    print(BORDER)
    print(f"Accuracy         : {accuracy_score(Y_test, Y_pred) * 100:.2f}%")
    print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))


def main():
    print(BORDER)
    print("Wine Classifier using KNN")
    print(BORDER)
    build_wine_classifier("WinePredictor.csv")


if __name__ == "__main__":
    main()
