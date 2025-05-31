'''
Multivariable linear regression is a type of linear regression where more than one independent variable (feature) is used to 
predict a single dependent variable (target).

🧮 The General Formula
y=β0+β1x1+β2x2+⋯+βnxn+ε


Where:

    y: Dependent variable (what you want to predict)

    x1,x2,…,x1​,x2​,…,xn​: Independent variables (features)

    β0​: Intercept

    β1,…,β1​,…,βn​: Coefficients (weights for each feature)

    ε: Error term (difference between actual and predicted)
    
🎯 Purpose

To model the relationship between multiple input variables and a single output, and use that relationship to:

    Understand how variables influence the target.

    Predict future outcomes.
    
🏡 Example

Suppose you’re predicting house prices (y) based on:

    Size in square feet (x₁)

    Number of bedrooms (x₂)

    Age of the house (x₃)

The model might look like:
Price=β0+β1⋅Size+β2⋅Bedrooms+β3⋅Age+ε

📊 When to Use It?

Use multivariable linear regression when:

    You have more than one predictor.

    You want to quantify relationships.

    You assume a linear relationship between predictors and target.

⚠️ Assumptions

Like all models, multivariable linear regression makes some assumptions:

    Linearity: The relationship between inputs and output is linear.

    Independence: Observations are independent.

    Homoscedasticity: Constant variance of residuals (errors).

    Normality: Residuals should be roughly normally distributed.

    No multicollinearity: Predictors shouldn’t be too highly correlated with each other (or consider using Ridge/Lasso regression).
    
✅ Benefits

    Simple and easy to interpret.

    Useful for understanding the strength and direction of relationships.

🚧 Limitations

    Sensitive to outliers.

    Can perform poorly if assumptions are violated.

    Can overfit if too many variables are used or if variables are correlated.


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create a sample dataset
data = {
    'Size_sqft': [1500, 1600, 1700, 1800, 1900, 2100, 2300, 2500, 2700, 3000],
    'Num_Bedrooms': [3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    'Age': [10, 9, 8, 15, 14, 10, 5, 3, 2, 1],
    'Price': [400000, 420000, 440000, 460000, 480000, 500000, 550000, 600000, 650000, 700000]
}

# 2. Prepare the data
df = pd.DataFrame(data)

x = df[['Size_sqft', 'Num_Bedrooms', 'Age']]  # independent variable
y = df['Price'] # dependent variable

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions 
y_pred = model.predict(X_test)

# 6. Evaluate the model
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)
print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print('R**2 score: ', r2_score(y_test, y_pred))

# 7. Visualize predictions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Acutal vs Predicted House Prices')
plt.plot([min(y_test), max(y_test)],[min(y_test),max(y_test)], color='red' )
plt.grid(True)
plt.show()



