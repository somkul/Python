from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

def main():
    dataset = load_iris()
    print("Features of dataset ")
    print(dataset.feature_names)

    print("Target names of dataset")
    print(dataset.target_names)


#    print("IRIS Dataset is :")
#    for icnt in range(len(dataset.target)):
#        print("ID : %d Feature : %s Label : %s  " %(icnt,dataset.data[icnt],dataset.target[icnt]))

# we can use %d, %s like in C.  Note that, we do not use "," in between.
# It replaces %d, %s, etc with values after % respectively
# E.g. %d -> %icnt, %s-> %dataset.data[icnt], etc.

    index=[1,51,101]

    test_target = dataset.target[index]
    test_features = dataset.data[index]

    train_target = np.delete(dataset.target,index)
    train_features = np.delete(dataset.data,index,axis=0)

    obj = tree.DecisionTreeClassifier()

    obj.fit(train_features,train_target)
    result = obj.predict(test_features)

    print("Expected Result is : ", test_target)
    print("Result Prediction by ML is : ", result)


if __name__ == "__main__":
    main()
