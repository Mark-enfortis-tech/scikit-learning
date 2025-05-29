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
    
    # model
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    
    # results
    print("predictions:", predictions)
    print("MSE:", mean_squared_error(y_test, predictions))
    print("Actual:", y_test)
    print("Predicted:", predictions)

if __name__ == "__main__":
    main()
