"""

    
Least squares is a method to find the best-fitting line by minimizing the total error between the predicted values and the actual values.

    The error for each data point is the difference between the actual yy and the predicted y^y^​ from the line.

    We square these errors to avoid negative and positive errors canceling out.

    Then, we sum all the squared errors for all data points.

Mathematically, for data points (xi,yi)(xi​,yi​) where i=1,2,...,ni=1,2,...,n, we want to minimize:
S=∑i=1n(yi−yi^)2=∑i=1n(yi−(mxi+b))2
S=i=1∑n​(yi​−yi​^​)2=i=1∑n​(yi​−(mxi​+b))2

Why is least squares used?

    It gives a unique, simple way to find the line that best represents the trend in the data.

    Squaring emphasizes larger errors more strongly, making the fit sensitive to big deviations.

    It has nice mathematical properties, making it easy to solve analytically.
    
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create sample data
# Let's make x values from 0 to 10
X = np.arange(0, 10, 1).reshape(-1, 1)  # reshape for sklearn (n_samples, n_features)
# Each value here is one sample (so 10 samples total).
# Because sklearn expects a 2D array (samples × features), we reshape it to have 10 rows (samples) and 1 column (feature).
# So, X.shape will be (10, 1), meaning 10 samples and 1 feature.
# -1: Let NumPy figure this out automatically based on the number of elements.
# 1: Make 1 column (i.e., 1 feature).

print('X.shape', X.shape)
print('X', X)

# y values roughly follow y = 2x + 1 plus some noise
y = 2 * X.flatten() + 1 + np.random.randn(10) * 1.5

# Step 2: Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 3: Print slope and intercept
print(f"Slope (m): {model.coef_[0]:.2f}")
print(f"Intercept (b): {model.intercept_:.2f}")

# Step 4: Predict values using the model
y_pred = model.predict(X)

# Plotting the data points and the fitted line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Fitted line (Least Squares)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression using Least Squares (scikit-learn)')
plt.show()
