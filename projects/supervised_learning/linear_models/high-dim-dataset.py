'''
🧠 Concept: More Features than Samples
        where: features (columns) > # of samples (rows)
    This is called a high-dimensional dataset.
    This situation is common in:
        Genomics (thousands of genes, few patients)

        Text classification (many word features, short documents)

        Image analysis (each pixel is a feature)
        
⚠️ Why It's a Problem

Having more features than samples can lead to:
Problem	Description
Overfitting	The model may "memorize" the training data — generalization becomes poor.
Singular matrix	The linear algebra breaks down — no unique least-squares solution.
Multicollinearity	Features are correlated, which causes unstable or unreliable coefficients.
    

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


# 2 samples, 4 features
X = np.array([
    [1, 2, 3, 4],
    [4, 3, 2, 1]
])
y = np.array([1, 2])  # target values

#  fit linear regression
model = LinearRegression()
model.fit(X,y)
predictions = model.predict(X)

# Project high deminsional data into 2D for plotting
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)


# Step 4: Plot the projected samples with predicted target as color
plt.figure(figsize=(6, 5))
scatter = plt.scatter(X_2D[:, 0], X_2D[:, 1], c=predictions, cmap='coolwarm', s=100, edgecolors='k')
plt.colorbar(scatter, label='Predicted Value')
plt.title("2 Samples (Projecting 4 Features to 2D via PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
for i, txt in enumerate(y):
    plt.annotate(f"y={txt}", (X_2D[i, 0] + 0.1, X_2D[i, 1]))
plt.grid(True)
plt.tight_layout()
plt.show()


