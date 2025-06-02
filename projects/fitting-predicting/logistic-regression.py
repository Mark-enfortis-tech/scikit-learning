#

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

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



