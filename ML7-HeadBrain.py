# This is the case study of Head brain
# The HeadBrain.csv file for data is already copied in the respective folder
# This uses Linear Regression.(Y=mX+C)
# We are going to plot Head Size on X axis and We would like to predict Head Weight on Y axis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def MeanData(arr):
    sum =0
    for i in range(len(arr)):
        sum=sum+arr[i]

    return (sum/len(arr))


def MarvellousHeadBrain(fname):
    dataset = pd.read_csv(fname)
    print("Size of the dataset is : ", dataset.shape)
    X=dataset["Head Size(cm^3)"].values
    Y=dataset["Brain Weight(grams)"].values

    print("Length of X : ", len(X))
    print("Length of Y : ", len(Y))

#    print ("Average of X using my function : ", MeanData(X))
#    print ("Average of X using built-in function of numpy : ", np.mean(X))

    Mean_X = np.mean(X) # Mean of Independent Variable
    Mean_Y = np.mean(Y) # Mean of dependent variable

# m=sum((x-xb)*(Y-Yb)) / sum((x-xb)^2)

    numeratorOFm = 0
    denominatorOFm = 0

    for i in range(len(X)):
        numeratorOFm = numeratorOFm + ((X[i]-Mean_X)*(Y[i]-Mean_Y))
        denominatorOFm = denominatorOFm + ((X[i]-Mean_X)**2)

    m=numeratorOFm/denominatorOFm
    print("Slope of the Regression line(m) : ", m)

# Y=mX+C
# C = Y-mX
# C = Mean_Y - m * Mean_X

    c = Mean_Y - (m*Mean_X)

    print("Value of Y Intercept : ",c)

# We will now plot the datapoints on graph
# First we will calculate Start and End Points of X axis

    X_Start = np.min(X) - 200
    X_End = np.max(X) + 200

    x = np.linspace(X_Start, X_End)
    y= m*x+c

    plt.plot(x,y,color='r',label="Regression line")
    plt.scatter(X,Y,color='b',label="Data Plot" )

    plt.xlabel("Head Size")
    plt.ylabel("Brain Weight")

    plt.legend()
    plt.show()


# Write the logic to calculate R Squared

def main():
    filename = input("Enter the File Name of data : ")
    MarvellousHeadBrain(filename)

if __name__=="__main__":
    main()

