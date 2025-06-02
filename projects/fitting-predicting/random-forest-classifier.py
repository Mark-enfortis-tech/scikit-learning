from sklearn.ensemble import RandomForestClassifier
"""
A Random Forest Classifier is a machine learning model used primarily for classification tasks, though it can also be used for regression. 
It’s part of the ensemble learning family, meaning it combines the predictions of multiple individual models to make more accurate predictions.

🔍 What It Is:

A Random Forest is made up of many decision trees, each trained on slightly different versions of the data. 
The final prediction is typically made by majority vote across all trees (in classification).

🧠 What It’s Used For:
    Classifying images (e.g., identifying whether a photo contains a cat or dog)
    Spam detection (classifying emails as spam or not)
    Medical diagnosis (e.g., predicting if a tumor is benign or malignant)
    Credit scoring (classifying loan applicants as high or low risk)
    Predicting user behavior (e.g., churn prediction in apps)

✅ Why Use It:
    Handles large datasets and high-dimensional features well
    Works for both categorical and numerical data
    Reduces overfitting compared to a single decision tree
    Automatically handles missing values and non-linear relationships

⚠️ Things to Watch Out For:
    Slower to predict and train with very large numbers of trees
    Less interpretable than a single decision tree (though still more interpretable than models like neural networks)
    
 ⚠️Classification models output Categorical Labels:
 
 🎯 What Is a Categorical Label?

In classification, a categorical label is the class or category that an input belongs to. Unlike regression (which predicts a number), 
classification predicts a discrete label from a fixed set of possible values.

🔢 Examples of Categorical Labels
Problem                         Input Features	        Categorical Label (Target)
Email filtering	                Email content	        "spam" or "not spam"
Image classification	        Pixel values	        "cat", "dog", or "car"
Sentiment analysis	            Text of a review	    "positive", "neutral", "negative"
Disease diagnosis	            Symptoms, lab results	"flu", "cold", "COVID"

These labels are names of categories, not numbers. Even if they are represented as numbers (like 0, 1, 2), they stand for categories—not values you can average or compare mathematically.
⚠️ Important Note

A categorical label:

    Has no numeric meaning (e.g., label 1 doesn’t mean it's more than label 0)

    Can be binary (e.g., yes/no, 0/1)

    Or multi-class (e.g., dog/cat/bird)

"""

# Initialize the classifier with a fixed random state for reproducibility
clf = RandomForestClassifier(random_state=0)

# Define training data (2 samples with 3 features each)
X = [[1, 2, 3],
     [11, 12, 13]]

# Define target labels for each sample
y = [0, 1]

# Fit the model to the data
clf.fit(X, y)

"""
The fit method generally accepts 2 inputs:

    The samples matrix (or design matrix) X. The size of X is typically (n_samples, n_features), which means that samples are represented as rows and features are represented as columns.

    The target values y which are real numbers for regression tasks, or integers for classification (or any other discrete set of values). For unsupervised learning tasks, y does not need to be specified. y is usually a 1d array where the i th entry corresponds to the target of the i th sample (row) of X.

Both X and y are usually expected to be numpy arrays or equivalent array-like data types, though some estimators work with other formats such as sparse matrices.

Once the estimator is fitted, it can be used for predicting target values of new data. You don’t need to re-train the estimator:
"""

# Predict on training data
#Once a model (like RandomForestClassifier) has been trained using .fit(X, y), 
# you can use .predict() to predict the class labels for new or existing data.
predicted_train = clf.predict(X)
print("Predicted on training data:", predicted_train)

# Predict on new data
# new_data = [[4, 5, 6], [6,7,8]] # [0,0]
new_data = [[4, 5, 6], [7,8,9]] # [0,1]
predicted_new = clf.predict(new_data)
print("Predicted on new data:", predicted_new)

""" here's how the decision tree arrives at the results:
🤔 How the Model Makes This Decision:

    Similarity: The point [4, 5, 6] is closer in value to [1, 2, 3] than to [11, 12, 13].

    Feature Space:

        Distance from [1, 2, 3]:
        sqrt[(4-1)**2 + (5-2)**2 (6-3)**2] = sqrt(27)

        Distance from [11, 12, 13]:
        sqrt[(11-4)**2 + (12-5)**2 (13-6)**2] = sqrt(147)
        

Decision Trees: Each tree in the Random Forest learns simple split rules (like "if feature 1 < 7, go left"). With only two training points, the tree will likely learn that lower feature values map to class 0.

Voting: Since [4, 5, 6] looks much more like [1, 2, 3], most trees in the forest will vote for class 0.

"""
