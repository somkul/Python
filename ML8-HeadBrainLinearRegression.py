import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def MarvellousHeadBrain(fname):
    dataset = pd.read_csv(fname)
    print("Size of the dataset is : ", dataset.shape)

    X=dataset["Head Size(cm^3)"].values
    Y=dataset["Brain Weight(grams)"].values

    X=X.reshape((-1,1))

    obj=LinearRegression()
    obj.fit(X,Y)

    output = obj.predict(X)
#   datasetTest = pd.read_csv("Test.csv")
#    X_new = datasetTest["Head Size"].values
#    output = obj.predict(X_new)
#    print("Expected Result is ", output)

    rsquare = obj.score(X,Y)

    print ("Value of R Square : ", rsquare)



def main():
    filename = input("Enter the File Name of data : ")
    MarvellousHeadBrain(filename)

if __name__=="__main__":
    main()
