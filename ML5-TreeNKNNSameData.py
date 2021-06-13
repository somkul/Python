# this program calculates the accuracy of Decision tree algorithm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def MarvellousDecision(dttrain, dttest,tgtrain,tgtest):

    cobj = tree.DecisionTreeClassifier()

    cobj.fit(dttrain, tgtrain)

    output = cobj.predict(dttest)

    Accuracy = accuracy_score(tgtest, output)

    return Accuracy

def MarvellousDecisionKNN(dttrain, dttest,tgtrain,tgtest):

    cobj = KNeighborsClassifier()

    cobj.fit(dttrain, tgtrain)

    output = cobj.predict(dttest)

    Accuracy = accuracy_score(tgtest, output)

    return Accuracy


def main():
    # load the data
    dataset = load_iris()  # dataset is variable name
    data = dataset.data
    target = dataset.target

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

    ret = MarvellousDecision(data_train, data_test, target_train, target_test)
    retKNN = MarvellousDecisionKNN(data_train, data_test, target_train, target_test)

    print("Accuracy of Decision Tree Algorithm is :", ret * 100, "%")
    print("Accuracy of KNN Algorithm is :", retKNN * 100, "%")


if __name__ == "__main__":
    main()
