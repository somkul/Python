import numpy as np
import pandas as pd
import seaborn as sb
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def TitanicLogisticRegression():
    print("Inside Titanic Logistic Regression")
# Step 1 - Load data
    Titanic_data=pd.read_csv('MarvellousTitanicDataset.csv')
    print("First 5 records of dataset")
    print(Titanic_data.head(5)) # shows first n records of dataset
    print("Total number of records are : ",len(Titanic_data))
    print("Dataset information : ", Titanic_data.info())

# Step 2 - Analyze the data
    print("Visualization of Survived and Non-Survived Passengers ")
    figure()
    countplot(data=Titanic_data,x="Survived").set_title("Survived v/s Non-Survived")
    show()

    print("Visualization According to Gender")
    figure()
    countplot(data=Titanic_data,x="Survived",hue="Sex").set_title("Visualization according to Sex")
    show()

    print("Visualization According to Passenger Class")
    figure()
    countplot(data=Titanic_data,x="Survived",hue="Pclass").set_title("Visualization according to Passenger Class")
    show()

    print("Survived v/s Non-Survived based on age")
    figure()
    Titanic_data["Age"].plot.hist().set_title("Visualization according to age")
    show()

# Step 3 - Data Cleaning After Analyzing.  We are deleting all such columns/data which are not making impact on deciding the final target
    Titanic_data.drop("zero",axis=1,inplace=True)
# The above statement is equivalent to below statement
#    Titanic_data = Titanic_data.drop("zero",axis=1)
    print("Data After the removal of Column namely zero")
    print(Titanic_data.head(5))

    Sex = pd.get_dummies(Titanic_data["Sex"])
    print(Sex)
    Sex = pd.get_dummies(Titanic_data["Sex"],drop_first=True)
    print("Sex Column after updation")
    print(Sex)

    PClass = pd.get_dummies(Titanic_data["Pclass"]) # Passenger Class
    print(PClass)
    PClass = pd.get_dummies(Titanic_data["Pclass"],drop_first=True)
    print("PClass Column after updation")
    print(PClass)


# Concatenate Sex and PClass fields in our dataset
    Titanic_data = pd.concat([Titanic_data,Sex,PClass],axis=1)
    print("Data After Concatenation ")
    print(Titanic_data.head())
# Removing un-necessary / non-influencing fields from dataset
    Titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True )
    print("Data set after removing non-influencing columns ")
    print(Titanic_data.head())

# Divide the data into X and Y
    X = Titanic_data.drop("Survived",axis=1)
    Y = Titanic_data["Survived"]

# Split the data for training and testing purpose
    Xtrain,XTest,YTrain,YTest=train_test_split(X,Y,test_size=0.5)

    obj = LogisticRegression(max_iter=2000)
# Step 4 - Train the dataset
    obj.fit(Xtrain, YTrain)

# Step 5 - Testing the dataset
    output = obj.predict(XTest)
# Step 6 - Calculating the accuracy
    print("Accuracy of Given data set is")
    print(accuracy_score(YTest, output))

# Step 7 - Calculating and displaying Confusion Matrix for an algorithm

    print("Confusion matrix is ")
    print(confusion_matrix(YTest,output))


def main():
    print("Logistic Regression - Titanic Case Study")
    TitanicLogisticRegression()

if __name__=="__main__":
    main()