# ridge_regression_example.py
"""
Ridge Regression is a type of linear regression that includes a regularization term to prevent overfitting 
and handle multicollinearity (when predictor variables are highly correlated).

Ridge regression modifies the standard linear regression by adding a penalty (shrinkage) term to the loss function. The model tries to minimize:
Loss=RSS+ α * ∑β**2:: j=1:p  

Where:
    RSS: Residual Sum of Squares (from linear regression)
    α (alpha): Regularization strength (you set this value)

    β₁, β₂, ..., βₚ: Model coefficients

That added term is the L2 penalty, which discourages large values in the model coefficients.


🧠 Why Use Ridge Regression?

    Handles Multicollinearity:

        In standard linear regression, if predictors are highly correlated, coefficient estimates can become unstable (sensitive to small changes in data).

        Ridge regression shrinks the coefficients, making them more stable and interpretable.

    Reduces Overfitting:

        In high-dimensional spaces (more features than samples), standard regression fits noise in the data.

        Ridge regression controls this by penalizing large coefficients, improving generalization.
        
        The complexity parameter controls the amount of shrinkage: the larger the value of, 
        the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

    Improves Model Performance:

        Especially useful when there are many features or when you want a model that is less sensitive to noise.
        
        ⚖️ Key Difference from Linear Regression
            Feature	                    Linear Regression	Ridge Regression
            Objective	                Minimize RSS	    Minimize RSS + L2 penalty
            Coefficients	            Can be large	    Tend to be smaller
            Handles multicollinearity?	❌ No	           ✅ Yes
            Overfitting risk	        Higher	            Lower

"""

from sklearn import linear_model

# Create Ridge regression model with alpha = 0.5
reg = linear_model.Ridge(alpha=0.5)

# Fit the model to the data
reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])

# Print the coefficients and intercept
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
