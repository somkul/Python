# this program calculates the accuracy of Decision tree algorithm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# KNN - K Nearest Neighbour

def MarvellousDecision():
    # load the data
    dataset = load_iris()  # dataset is variable name
    data = dataset.data
    target = dataset.target

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

    cobj = tree.DecisionTreeClassifier()

    cobj.fit(data_train, target_train)

    output = cobj.predict(data_test)

    Accuracy = accuracy_score(target_test, output)

    return Accuracy

def MarvellousDecisionKNN():
    # load the data
    dataset = load_iris()  # dataset is variable name
    data = dataset.data
    target = dataset.target

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

    cobj = KNeighborsClassifier()

    cobj.fit(data_train, target_train)

    output = cobj.predict(data_test)

    Accuracy = accuracy_score(target_test, output)

    return Accuracy


def main():
    ret = MarvellousDecision()
    retKNN = MarvellousDecisionKNN()

    print("Accuracy of Decision Tree Algorithm is :", ret * 100, "%")
    print("Accuracy of KNN Algorithm is :", retKNN * 100, "%")


if __name__ == "__main__":
    main()
