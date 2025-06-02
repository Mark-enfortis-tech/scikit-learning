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