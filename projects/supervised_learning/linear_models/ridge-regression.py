# ridge_regression_example.py

from sklearn import linear_model

# Create Ridge regression model with alpha = 0.5
reg = linear_model.Ridge(alpha=0.5)

# Fit the model to the data
reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])

# Print the coefficients and intercept
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
