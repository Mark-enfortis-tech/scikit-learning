# Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# borrow some stuff from support-vector-machines.py
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# load the data
digits = datasets.load_digits()
X, y = digits.data, digits.targe

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train using linear SVC
clf = SVC(kernel='linear')  
clf.fit(X_train, y_train)

# evaluate the model
y_pred = clf.predict(X_test)

# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)


# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



