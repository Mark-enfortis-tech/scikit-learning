# logistic regression
"""

Difference between linear and logistic regression
🔑 Key Differences
Feature	            Linear Regression	                            Logistic Regression
Type of problem	    Regression (predict continuous values)	        Classification (predict categories/classes)
Output	            Real-valued number (e.g., price, temperature)	Probability (between 0 and 1), then class label
Target (y)	        Continuous (e.g., 3.5, 100.0)	                Categorical (e.g., 0 or 1)
Model equation	    y = wX + b	                                    p = 1 / (1 + e^(-wX + b)) (sigmoid function)
Final prediction	The number from the regression line	            Thresholded probability (e.g., p > 0.5 → class 1)
Loss function	    Mean Squared Error (MSE)	                    Log Loss (a.k.a. Binary Cross-Entropy)

📘 Example Use Cases:

    Linear Regression:
    Predicting house prices, stock values, temperature, etc.

    Logistic Regression:
    Predicting if an email is spam, if a patient has a disease, or if a transaction is fraudulent (binary/multiclass classification).

📈 Visual Intuition:

    Linear regression fits a line to continuous data.

    Logistic regression fits an S-shaped curve (sigmoid) that maps any input to a value between 0 and 1, then applies a threshold (usually 0.5) to classify.
    
For the iris dataset:
The 4 features (iris.data) are:
Index	Feature	Unit (cm)	Description
0	Sepal length	centimeters (cm)	Length of the outer part of the flower (sepal)
1	Sepal width	centimeters (cm)	Width of the sepal
2	Petal length	centimeters (cm)	Length of the inner petals
3	Petal width	centimeters (cm)	Width of the petals

The target vector (iris.target):
y = [0 0 0 0 0 ... 1 1 1 1 ... 2 2 2 2 ...]
This is the target vector of class labels for each sample in the Iris dataset
What do the values mean?

Each value corresponds to one of the three iris species:
Target Value	Species Name
0	            Iris setosa
1	            Iris versicolor
2	            Iris virginica

How does this relate to your array?

    The first 50 samples have label 0, meaning these samples belong to Iris setosa.
    The next 50 samples have label 1, meaning these are Iris versicolor.
    The last 50 samples have label 2, which means these are Iris virginica


"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

# determine number of classes of data set
print("Target names:", iris.target_names)
print("Number of classes:", len(iris.target_names))

print('X:',X)
num_rows = X.shape[0]
print(f"Number of rows: {num_rows}")
num_cols = X.shape[1]
print(f"Number of cols: {num_cols}")

print('y:', y)

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print('X_train:', X_train)
num_rows = X_train.shape[0]
print(f"Number of rows: {num_rows}")

print('X_test:', X_test)
num_rows = X_test.shape[0]
print(f"Number of rows: {num_rows}")

print('y_train:', y_test)
print('y_test:', y_test)

# define model
model = LogisticRegression(max_iter=200)

#
model.fit(X_train, y_train)

# predict on test data
y_pred = model.predict(X_test)
print('y_pred: ', y_pred)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")



