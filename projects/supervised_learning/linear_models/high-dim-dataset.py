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
from sklearn.linear_model import LinearRegression

# 2 samples, 4 features
X = np.array([
    [1, 2, 3, 4],
    [4, 3, 2, 1]
])
y = np.array([1, 2])  # target values

#  fit linear regression
model = LinearRegression()
model.fit(X,y)

print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)


