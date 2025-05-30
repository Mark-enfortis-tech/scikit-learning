import numpy as np
from sklearn.model_selection import train_test_split

# Why split the data in to training and testing. If you trained and tested on the same data:
# Your model might just memorize it (overfitting)
# You’d get an overly optimistic view of performance
# So we hold out part of the data to simulate how the model performs in the real world — on data it's never seen before.

def main():
    # Create a feature matrix X with values from 0 to 9
    # Reshape it into 5 rows and 2 columns -> 5 samples, 2 features each
    
    X = np.arange(10).reshape((5, 2))
    #  np.arrange(n) creates a 1D array of integers from 0::9
    # .reshape(5,2) - takes the 1D array and converts to a 2D array/
    
    # Create a target vector y with 5 values
    y = list(range(5))
    
    z = list(range(5))

    print("X:")
    print(X)

    print("\ny:")
    print(y)
    
    print("\nz:")
    print(z)

    # Split the data into training and test sets
    # test_size=0.33 means roughly 33% of the data will go to the test set
    # random_state=42 ensures reproducible results every time you run it
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X, y, z, test_size=0.33, random_state=42)

    print("\nX_train:")
    print(X_train)
    
    print("\nX_test:")
    print(X_test)

    print("\ny_train:")
    print(y_train)

    print("\ny_test:")
    print(y_test)
    
    print("\nz_train:")
    print(z_train)

    print("\nz_test:")
    print(z_test)

    # Demonstrate splitting without shuffling
    # This is useful when you want to preserve the order of the data (e.g., in time series)
    y_train_seq, y_test_seq = train_test_split(y, shuffle=False)
    print("\nSplit without shuffle:")
    print("y_train_seq:", y_train_seq)
    print("y_test_seq:", y_test_seq)

if __name__ == "__main__":
    main()
