# 🤖 Machine Learning Projects Portfolio

A collection of hands-on Machine Learning projects built using Python and Scikit-learn.  
Each project covers a different ML concept — from supervised classification to unsupervised clustering.

---

## 📁 Projects

| Project | Algorithm | Type | Dataset |
|---|---|---|---|
| [Iris Classifier Pipeline](#1-iris-classifier-pipeline) | KNN + GridSearchCV | Supervised Classification | Iris (built-in) |
| [Mall Customer Segmentation](#2-mall-customer-segmentation) | K-Means + Elbow Method | Unsupervised Clustering | Mall_Customers.csv |
| [Wine Classifier](#3-wine-classifier-knn) | KNN with Hyperparameter Tuning | Supervised Classification | WinePredictor.csv |
| [Advertising Sales Predictor](#4-advertising-sales-predictor) | Linear Regression | Supervised Regression | Advertising.csv |

---

## 1. Iris Classifier Pipeline

**File:** `IrisDataset_Pipeline.py`

Builds a full Scikit-learn Pipeline combining feature scaling and KNN classification, then uses GridSearchCV to find the best hyperparameters automatically.

**Concepts covered:**
- Scikit-learn Pipeline (StandardScaler + KNeighborsClassifier)
- GridSearchCV for hyperparameter tuning
- Cross-validation (5-fold)
- Accuracy, Confusion Matrix, F1 Score

**How to run:**
```bash
python IrisDataset_Pipeline.py
```
> No external dataset needed — uses Scikit-learn's built-in Iris dataset.

---

## 2. Mall Customer Segmentation

**File:** `Mall_Customers_KMean_Elbow.py`

Segments mall customers into groups based on Annual Income and Spending Score using K-Means clustering. Uses the Elbow Method to determine the optimal number of clusters.

**Concepts covered:**
- Unsupervised Learning with K-Means
- Elbow Method (WCSS) for optimal cluster selection
- Feature Scaling with StandardScaler
- Data visualisation with Matplotlib

**How to run:**
```bash
python Mall_Customers_KMean_Elbow.py
```
> Requires `Mall_Customers.csv` in the same directory.

---

## 3. Wine Classifier (KNN)

**File:** `WineClassifierKNNModelVisualizationFinal.py`

Classifies wines into categories using the K-Nearest Neighbours algorithm. Tests K values from 1–20 to find the best performing model and visualises accuracy vs K.

**Concepts covered:**
- KNN classification with manual hyperparameter tuning
- Stratified train-test split
- Accuracy, Confusion Matrix, Classification Report
- K vs Accuracy plot with Matplotlib

**How to run:**
```bash
python WineClassifierKNNModelVisualizationFinal.py
```
> Requires `WinePredictor.csv` in the same directory.

---

## 4. Advertising Sales Predictor

**File:** `AdvertisingCaseStudyModelBuildingVisualization.py`

Predicts product sales based on TV, Radio, and Newspaper advertising budgets using Multiple Linear Regression. Includes full EDA, correlation analysis, and actual vs predicted visualisation.

**Concepts covered:**
- Multiple Linear Regression
- Exploratory Data Analysis (EDA)
- Correlation Matrix
- MSE, RMSE, R² evaluation metrics
- Actual vs Predicted scatter plot

**How to run:**
```bash
python AdvertisingCaseStudyModelBuildingVisualization.py
```
> Requires `Advertising.csv` in the same directory.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib

**Install dependencies:**
```bash
pip install scikit-learn pandas numpy matplotlib
```

---

## 👩‍💻 Author

**Aishwarya Dalvi**  
Python Developer | Machine Learning Enthusiast  
[LinkedIn](https://linkedin.com/in/aishwaryadilipgaikwad080) • [GitHub](https://github.com/AishwaryaG080)
