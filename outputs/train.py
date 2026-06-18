from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # shape (150,4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: list, y_pred: list, label_list: list, title: str = "Confusion matrix") -> matplotlib.figure.Figure:
    conf_mat = confusion_matrix(y_true, y_pred, labels=label_list)
    print(conf_mat)
    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=label_list, yticklabels=label_list)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.title(title)

    plt.tight_layout()

    return fig
# Source - https://stackoverflow.com/a/56107843
# Posted by sentence, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-18, License - CC BY-SA 4.0

from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC()
clf.fit(X, y)  

##########################
# SAVE-LOAD using joblib #
##########################
import joblib

# save
joblib.dump(clf, "model.pkl") 

# load
clf2 = joblib.load("model.pkl")

clf2.predict(X[0:1])
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true = [0,1,1,1,0]
y_pred = [1,1,1,1,0]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('confusion_matrix.png')