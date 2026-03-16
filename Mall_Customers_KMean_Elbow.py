"""
Mall Customer Segmentation — K-Means Clustering
=================================================
Segments mall customers into groups based on Annual Income and Spending Score.
Uses the Elbow Method (WCSS) to determine the optimal number of clusters.

Dataset: Mall_Customers.csv
Features used: AnnualIncome, SpendingScore

Steps:
    1. Load and explore the dataset
    2. Select features
    3. Scale the data
    4. Apply Elbow Method to find optimal K
    5. Train final K-Means model and assign clusters
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    # --------------------------------------------------
    # Step 1: Load and Explore the Dataset
    # --------------------------------------------------
    df = pd.read_csv("Mall_Customers.csv")
    print("First 5 records:\n", df.head())
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())

    # --------------------------------------------------
    # Step 2: Select Features
    # --------------------------------------------------
    X = df[["AnnualIncome", "SpendingScore"]]
    print("Selected features shape:", X.shape)

    # --------------------------------------------------
    # Step 3: Scale the Data
    # --------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # Step 4: Elbow Method — find optimal number of clusters
    # --------------------------------------------------
    wcss = []
    for i in range(1, 11):
        model = KMeans(n_clusters=i, random_state=42, n_init=10)
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method — Optimal K")
    plt.grid(True)
    plt.show()

    # --------------------------------------------------
    # Step 5: Train Final K-Means Model (K=4)
    # --------------------------------------------------
    final_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = final_model.fit_predict(X_scaled)

    print("Dataset with cluster labels:\n", df.head(30))


if __name__ == "__main__":
    main()
