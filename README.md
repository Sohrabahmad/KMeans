# K-Means Clustering: Customer/Data Segmentation

## 📌 Project Overview

This project demonstrates **K-Means Clustering**, an unsupervised machine learning algorithm used to group similar data points into clusters. The implementation uses synthetic data generated with Scikit-learn and explores methods for selecting the optimal number of clusters.

The notebook covers data visualization, cluster formation, the Elbow Method, automatic elbow detection, and Silhouette Score analysis.

## 🎯 Objectives

- Generate a synthetic dataset containing multiple clusters.
- Visualize the original data using Seaborn.
- Apply K-Means clustering to group data points.
- Determine a suitable number of clusters using the Elbow Method.
- Automatically detect the elbow point using `KneeLocator`.
- Evaluate clustering quality using the Silhouette Score.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas (optional for additional data analysis)
- Matplotlib
- Seaborn
- Scikit-learn
- `kneed`

## 📂 Project Structure

```text
KMeans-Clustering/
│
├── KMeans.ipynb
└── README.md
```

## 📊 Dataset

The notebook generates a synthetic dataset using Scikit-learn's `make_blobs()` function.

```python
X, y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=4,
    random_state=42
)
```

Dataset characteristics:

- **Number of samples:** 1,000
- **Number of features:** 2
- **Generated centers:** 4
- **Random state:** 42

The generated `X` data is used for clustering. The original `y` labels are available for reference but are not used to train the K-Means model.

## ⚙️ Project Workflow

### 1. Data Visualization

Seaborn is used to visualize the generated two-dimensional dataset with a scatter plot.

### 2. K-Means Clustering

The notebook applies K-Means with four clusters.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

labels = kmeans.fit_predict(X)
```

The algorithm assigns each data point a cluster label based on its position relative to the learned cluster centers.

### 3. Cluster Visualization

The resulting cluster labels are used to color the scatter plot, helping visualize the groups identified by K-Means.

### 4. Elbow Method

The notebook calculates the Within-Cluster Sum of Squares (WCSS), also called inertia, for cluster counts from 1 to 20.

```python
wcss = []

for k in range(1, 21):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
```

A line plot of cluster count against WCSS helps identify the elbow point, where adding more clusters produces diminishing improvements.

### 5. Automatic Elbow Detection

The `kneed` library is used to identify the elbow automatically.

```python
from kneed import KneeLocator

knee = KneeLocator(
    range(1, 21),
    wcss,
    curve="convex",
    direction="decreasing"
)

print("Optimal k =", knee.elbow)
```

The detected value depends on the generated data and the WCSS values. The notebook's code does not document a confirmed output value.

### 6. Silhouette Score

The notebook calculates the Silhouette Score for cluster counts from 2 to 20.

```python
from sklearn.metrics import silhouette_score

ss = []

for k in range(2, 21):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    ss.append(score)
```

The scores are plotted against the number of clusters to compare clustering quality.

A higher Silhouette Score generally indicates better-separated, more cohesive clusters. The best cluster count should be assessed from the plotted results rather than assumed in advance.

## 📈 Key Concepts

| Concept | Description |
|---|---|
| K-Means | Groups data points into a specified number of clusters. |
| Cluster Centroid | The center representing a cluster. |
| WCSS / Inertia | Measures the sum of squared distances from points to their assigned centroids. |
| Elbow Method | Helps estimate a suitable number of clusters by examining WCSS. |
| KneeLocator | Detects an elbow point automatically from a curve. |
| Silhouette Score | Evaluates cluster cohesion and separation. |

## 💻 Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd KMeans-Clustering
```

Replace `<your-repository-url>` with your GitHub repository URL.

### 2. Install Dependencies

```bash
pip install numpy matplotlib seaborn scikit-learn kneed jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook KMeans.ipynb
```

Run the notebook cells in order to generate the data, fit the clustering models, and view the evaluation plots.

## 🔍 Results

The notebook produces:

- A scatter plot of the original synthetic dataset.
- A scatter plot showing the clusters assigned by K-Means.
- An Elbow Method plot using WCSS values for \(k = 1\) through \(20\).
- An automatically detected elbow value, when identified by `KneeLocator`.
- A Silhouette Score plot for \(k = 2\) through \(20\).

The notebook does not include documented final metric values, so specific optimal-cluster or Silhouette Score results should be added after running the analysis.

## 🚀 Applications

K-Means clustering can be applied to:

- Customer segmentation
- Grouping similar data points
- Image segmentation
- Exploratory data analysis
- Pattern discovery
- Document and product grouping

These are potential applications of the algorithm; this notebook specifically demonstrates clustering on synthetic data.

## 🔮