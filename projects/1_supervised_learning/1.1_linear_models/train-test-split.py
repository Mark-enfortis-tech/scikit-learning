import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # Create a feature matrix X with values from 0 to 9
    # Reshape it into 5 rows and 2 columns -> 5 samples, 2 features each
    
    X = np.arange(10).reshape((5, 2))
    
    # Create a target vector y with 5 values
    y = list(range(5))

    print("X:")
    print(X)

    print("\ny:")
    print(y)

    # Split the data into training and test sets
    # test_size=0.33 means roughly 33% of the data will go to the test set
    # random_state=42 ensures reproducible results every time you run it
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    print("\nX_train:")
    print(X_train)

    print("\ny_train:")
    print(y_train)

    print("\nX_test:")
    print(X_test)

    print("\ny_test:")
    print(y_test)

    # Demonstrate splitting without shuffling
    # This is useful when you want to preserve the order of the data (e.g., in time series)
    y_train_seq, y_test_seq = train_test_split(y, shuffle=False)
    print("\nSplit without shuffle:")
    print("y_train_seq:", y_train_seq)
    print("y_test_seq:", y_test_seq)

if __name__ == "__main__":
    main()
