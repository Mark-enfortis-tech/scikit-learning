import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    print("starting main.py")
    
    # data
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    # split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # model
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    
    # results
    print("predictions:", predictions)
    print("MSE:", mean_squared_error(y_test, predictions))

if __name__ == "__main__":
    main()
