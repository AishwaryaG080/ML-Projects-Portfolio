"""
Iris Dataset Classification Pipeline
======================================
Builds a Scikit-learn Pipeline combining StandardScaler and KNeighborsClassifier.
Uses GridSearchCV to find the best value of K, then evaluates the model.

Steps:
    1. Load dataset
    2. Split into train/test sets
    3. Build Pipeline (Scaler + KNN)
    4. Define hyperparameter grid
    5. Run GridSearchCV (finds best K using cross-validation)
    6. Evaluate: Accuracy, Confusion Matrix, F1 Score, Cross-Val Scores
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    # --------------------------------------------------
    # Step 1: Load Dataset
    # --------------------------------------------------
    iris = load_iris()
    X = iris.data
    Y = iris.target

    # --------------------------------------------------
    # Step 2: Split into Train / Test Sets (80/20)
    # --------------------------------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------
    # Step 3 & 4: Build Pipeline and Define Param Grid
    # --------------------------------------------------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ])

    param_grid = {
        "model__n_neighbors": [3, 5, 7, 9]
    }

    # --------------------------------------------------
    # Step 5: GridSearchCV — finds best K via 5-fold CV
    # --------------------------------------------------
    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X_train, Y_train)
    print("Best Parameters:", grid.best_params_)

    # --------------------------------------------------
    # Step 6: Evaluate the Best Model
    # --------------------------------------------------
    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)

    print("Accuracy       :", accuracy_score(Y_test, pred))
    print("F1 Score       :", f1_score(Y_test, pred, average="macro"))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, pred))

    # Cross-validation on full dataset
    scores = cross_val_score(pipeline, X, Y, cv=5)
    print("Cross-Val Scores :", scores)
    print("Average Accuracy :", scores.mean())


if __name__ == "__main__":
    main()
