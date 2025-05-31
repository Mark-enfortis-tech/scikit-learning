"""
Linear regression is a statistical method used to model the relationship between one dependent variable yy and one or more independent variables xx.
    The goal is to find the best-fitting straight line (in the simplest case with one variable) through a set of data points.
    This line predicts yy based on xx.

For a single variable, the model looks like:
y=mx+b

where:

    m is the slope of the line,

    b is the intercept (where the line crosses the y-axis).
    
Least squares is a method to find the best-fitting line by minimizing the total error between the predicted values and the actual values.

    The error for each data point is the difference between the actual y and the predicted y^​ from the line.

    We square these errors to avoid negative and positive errors canceling out.

    Then, we sum all the squared errors for all data points.

Mathematically, for data points (xi,yi)(xi​,yi​) where i=1,2,...,ni=1,2,...,n, we want to minimize:
S=∑i=1n(yi−yi^)2=∑i=1n(yi−(mxi+b))2 
S=i=1∑n​(yi​−yi​^​)2=i=1∑n​(yi​−(mxi​+b))2
    
"""


import numpy as np
# NumPy is a library for numerical operations, especially useful for working with arrays and matrices.

from sklearn.model_selection import train_test_split
# This function helps you split your dataset into training and testing parts. 
# Training data is used to train the model, and testing data is used to evaluate its performance.


from sklearn.linear_model import LinearRegression
# This imports the LinearRegression class from scikit-learn. 
# It allows creation of a model that learns the relationship between features and target using a straight line.


from sklearn.metrics import mean_squared_error
# This is a function to evaluate how well your model's predictions match the actual values.
# It calculates the Mean Squared Error (MSE), which is the average of the squared differences between predicted and true values.


def main():
    print("starting main.py")
    
    # data
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    # x is your feature/input array (in this case, a 2D array with one column), and y is your target/output array.
    # This is a simple linear relationship where y = 2 * x.
    
    # split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # Splits the data into training and test sets.
    # 80% of the data goes to training, and 20% goes to testing.
    # This is important to evaluate the model's performance on unseen data.
    #sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
    # test_size:    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
    #               If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size.
    
    # model
    model = LinearRegression()
    # This creates an instance of the LinearRegression model.
    # The model tries to find the best-fitting straight line through the training data
    # by minimizing the error between the predicted and actual values (using the least squares method).
    
    model.fit(x_train, y_train)
    # This trains the model on the training data (x_train and y_train).
    # It calculates the best slope and intercept (in the equation y = mx + b) to fit the data.
    # After calling fit, the model has "learned" the relationship between x and y.

    predictions = model.predict(x_test)
    # This uses the trained model to make predictions on the test data (x_test).
    # It applies the learned equation (y = mx + b) to the test inputs and outputs the predicted values.

    
    # results
    print("predictions:", predictions)
    # Prints the model’s predictions for the test set.
    
    print("MSE:", mean_squared_error(y_test, predictions))
    # Calculates and prints the Mean Squared Error between the predicted and actual test values.
    # Lower MSE means better performance.
    
    print("Actual:", y_test)
    print("Predicted:", predictions)

if __name__ == "__main__":
    main()
