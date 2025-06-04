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

💡 In SVM (particularly this example using 2D image data X[n]),  for each corresponding X[n], there is a scalar quantity y[n] that maps 1:1. In 
this case y[n] is called a label but represents a known truth about the sample X[n]. The training process is where the model learns the mapping from
X[n] (the image), to y[n] (the digit or value). Given this information (ie once trained), we can take new X inputs and generate y predictions that 
we hope match the true y. This is the essence of supervised learning. 


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
import numpy as np
from collections import Counter


# load the data
digits = datasets.load_digits()
X, y = digits.data, digits.target
# pixel_value = X[0][0]
# print("X[0][0] = ", pixel_value)

# pixel_value = X[0][2]
# print("X[0][2] = ", pixel_value)

# pixel_value = X[0][3]
# print("X[0][3] = ", pixel_value)

# pixel_value = X[0][4]
# print("X[0][4] = ", pixel_value)

# pixel_value = X[0][10]
# print("X[0][10] = ", pixel_value)

# pixel_value = X[0][11]
# print("X[0][11] = ", pixel_value)

# pixel_value = X[0][18]
# print("X[0][18] = ", pixel_value)





# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('X_train:', X_train)
print('X_test:', X_test)

print('Training labels shape: ', np.shape(y_train))
print('Test labels shape: ', np.shape(y_test))
print('y_train: ', y_train)
print('y_test: ', y_test)

# lets determine information about the dataset using collections.Counter
# train_distribution = Counter(y_train)
# test_distribution = Counter(y_test)
# print('Training label distribution: ')
# print(train_distribution)
# print('\nTest distribution:')
# print(test_distribution)

"""
Here's the output of the training dist
Counter({
    np.int64(1): 132,
    np.int64(8): 131,
    np.int64(2): 130,
    np.int64(3): 129,
    np.int64(6): 128,
    np.int64(0): 125,
    np.int64(7): 124,
    np.int64(4): 121,
    np.int64(9): 121,
    np.int64(5): 116
})

and here's the test distribution:
Counter({
    np.int64(5): 66,
    np.int64(4): 60,
    np.int64(9): 59,
    np.int64(7): 55,
    np.int64(3): 54,
    np.int64(6): 53,
    np.int64(0): 53,
    np.int64(1): 50,
    np.int64(2): 47,
    np.int64(8): 43
})

"""

'''
🔍 Key concepts
1. Each sample is a point in high-dimensional space

    X_train[n] is a 64-dimensional vector (from 8×8 pixels).

    So you can think of each image as a point in 64D space.

    SVC is trying to group similar points together (same digit) and separate different digits.

2. Decision boundary (hyperplane)

    For binary classification (e.g., is it a 3 or not?), SVC finds a hyperplane that separates the two classes.

    This hyperplane is chosen to maximize the margin — the distance between the boundary and the closest points from either class.

3. Support vectors

    The points that lie closest to the boundary are called support vectors.

    These are the "hardest" cases — the most informative training examples.

    The hyperplane is defined based on these support vectors.

4. Linear kernel

    The kernel='linear' means we are trying to separate the classes using a straight line (or flat plane in high dimensions).

    It's efficient and works well when the data is linearly separable (or almost).
'''

# # train an SVC
clf = SVC(kernel='linear') # tryp 'rbf' and 'poly' later for comparison 
clf.fit(X_train, y_train)

# evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

"""
Info from the classification report
✅ Precision
    Of all the times the model predicted this class, how often was it right?

    Precision=True Positives / (True Positives + False Positives)
    False positive, claims to be in group but is not

    High precision = low false positives.
    For example, if digit 3 was predicted 50 times, and 48 were actually 3, precision is 0.96.
    
✅ Recall

    Of all the actual instances of this class, how many did the model catch?

    Recall=True Positives / (True Positives + False Negatives)
    False negative, is in the group but was missed.

    High recall = low false negatives.
    If there were 60 true 3s and the model caught 54, recall = 0.90.
    

✅ F1-score

    The harmonic mean of precision and recall.

    F1=2⋅Precision⋅Recall/ (Precision + Recall)


    F1 balances both precision and recall, useful if you want a single metric to optimize.

    Especially useful when classes are imbalanced.
    
✅ Support

    The number of actual instances of this class in the test set.

    This is the ground truth count.

    If support = 59 for digit 0, that means there are 59 real zeroes in y_test.
    
🧠 Why it matters

    This report helps you see:

    Which digits your model is best at predicting

    Which ones it's confusing or missing

    Whether it's biased toward more frequent classes (via support)

"""

"""
🧪 Step 1: Generate the Report
Here's the classification report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        53
           1       0.98      0.98      0.98        50
           2       0.98      1.00      0.99        47
           3       1.00      0.96      0.98        54
           4       0.98      0.98      0.98        60
           5       0.97      0.97      0.97        66
           6       1.00      1.00      1.00        53
           7       0.96      0.98      0.97        55
           8       0.95      0.98      0.97        43
           9       0.97      0.95      0.96        59

    accuracy                           0.98       540
   macro avg       0.98      0.98      0.98       540
weighted avg       0.98      0.98      0.98       540

🔍 Step 2: Interpret the Output
    Consider digit 5: 
    precision   0.97 - when the model predicited a 5 it was right 97% of the time
    recall      0.97 - of the actual 5's the model found 97% of them
    F1 score    0.97 - harmonic mean of precision and recall
    Suport      66   - there were 66 5's in the sample
    
🧠 Step 3: Big Picture Metrics

    Accuracy: 0.97 → The model was right 97% of the time overall.

    Macro avg: Simple average across all classes (treats all classes equally).

    Weighted avg: Weighted by the number of samples per class (support), so classes with more samples influence this more.
    
     The labels 0 through 9 represent the 10 possible classes—each digit is one class the model tries to predict.


✅ Overall concept
    The classifier’s job is to assign each input image to one of these 10 categories. This is a classic multi-class classification problem.
    
    
"""

# visualize the data 
plt.imshow(digits.images[0], cmap='gray', interpolation='none')
plt.title(f"Label: {digits.target[0]}")
plt.colorbar()
plt.show()

