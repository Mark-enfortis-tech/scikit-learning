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


'''