# Support Vector Machine

"""
💡 What is SVM?
Support Vector Machine is a supervised machine learning algorithm that can be used for classification, regression, and even outlier detection.
It works by finding the best decision boundary (hyperplane) that separates classes with the maximum margin.

📐 Core Idea
In binary classification, SVM tries to find the hyperplane that:
    Separates the classes (e.g., red vs. blue points),
    Maximizes the margin — the distance between the hyperplane and the nearest points from each class (called support vectors).

This margin maximization helps generalize better to unseen data.

✅ When to Use SVM
🔸 Good fit when:
    Feature space is high-dimensional (like text or gene data)
    You need strong generalization with few samples
    You care about robust decision boundaries (clear margins)
    You want a model that can be non-linear, but still interpretable

🔸 Caution when:
    You have very large datasets (training time grows with data)
    You need probabilistic outputs (SVM does not do this natively — must use probability=True, which slows things down)
    
    Feature	    Summary
    Goal	    Maximize margin between classes
    Type	    Supervised, classification and regression
    Key         concept	Decision boundary defined by support vectors
    Strength	High accuracy in high-dimensional space
    Weakness	Can be slow on large datasets, hard to tune for big tasks



🛫 What Is a Hyperplane?

A hyperplane is a generalization of a plane to n-dimensional space. In SVM, it’s the decision boundary that separates different classes.

📊 Examples by Dimension:
Dimensionality of data	        Hyperplane is...	    Example
1D	                            A point	                Divides a line into two parts
2D	                            A line	                Separates the 2D plane
3D	                            A plane	                Splits 3D space (like a sheet)
n-D	                            A hyperplane	        General boundary in n-D space


✏️ Mathematical Form:

A hyperplane in nn-dimensional space is described by a linear equation:
w⋅x+b=0

Where:
    w: vector of weights (normal to the hyperplane)
    x: input feature vector
    b: bias (intercept)
    
📌 Interpretation in SVM:

In classification:
    The hyperplane divides the feature space into regions for different classes.
    SVM chooses the hyperplane that:
        Separates the classes
        Maximizes the margin (distance to the nearest points from each class)

This results in better generalization.

"""

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

digits = datasets.load_digits()
X, y = digits.data, digits.target

# visualize the data 
plt.gray()
plt.matshow(digits.images[0])
plt.title(f"Label: {digits.target[0]}")
plt.show()


# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train an SVC
clf = SVC(kernel='linear') # tryp 'rbf' and 'poly' later for comparison 
clf.fit(X_train, y_train)

# evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

