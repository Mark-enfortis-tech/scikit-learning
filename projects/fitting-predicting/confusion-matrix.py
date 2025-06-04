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
X, y = digits.data, digits.target

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


'''
✅ Your Setup:

    Horizontal axis (x-axis): Predicted labels (what the model guessed)

    Vertical axis (y-axis): True labels (actual values from y_test)

    Each cell at (row, col) tells us:
    How many times a true digit row was predicted as digit col
    
🧩 So for (0, 0) = 53:

That means:

    The digit 0 was correctly predicted as 0 53 times.

    This is a true positive for class 0.

    Since the rest of column 0 is all zeros:
    The model never falsely predicted a 0 when the actual digit was something else (like a 1–9). 
    Thats a perfect precision for class 0.
    
🧠 How it works:

The confusion matrix is a single 2D array, typically 10x10 for digit classification (0–9):

    Each row corresponds to a true label

    Each column corresponds to a predicted label

From that one matrix, you can extract:
✅ Precision for class i:

Look down column i
→ "Of all things predicted as class i, how many were correct?"
✅ Recall for class i:

Look across row i
→ "Of all true class i items, how many were correctly predicted?"
🔁 Summary:
Metric	Direction	Question Asked	Pull From Confusion Matrix
Precision	Columns	"How many of the predicted class X were right?"	Column X
Recall	Rows	"How many actual class X were caught?"	Row X


🔢 In the Confusion Matrix:
	Predicted: 0	Predicted: 1	...	Predicted: 9
True: 0	✅ True Positive	❌ False Negative		
True: 1	❌ False Positive	✅ True Positive		
...			...	
🔍 Terminology Breakdown:

    True Positive (TP):
    Correct prediction — it’s class n, and the model predicted n.
    → On the diagonal (e.g., cm[3][3])

    False Positive (FP):
    Model predicted n, but the true label was something else.
    → In column n, but off the diagonal

    False Negative (FN):
    Actual label was n, but model predicted something else.
    → In row n, but off the diagonal

🚦 Think of it this way:

    Columns = What the model predicted

        Off-diagonal cells = false positives (model guessed wrong)

    Rows = What the ground truth actually was

        Off-diagonal cells = false negatives (model missed it)
    
    
'''



