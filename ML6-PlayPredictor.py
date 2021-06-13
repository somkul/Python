# this is the case study of play predictor based on weather and temperature
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def MarvellousPredictor(path):
# load the CSV
    data=pd.read_csv(path) # This is called as dataframe.  Multiple data frames together called as data panel
    print("Dataset loaded successfully with size", len(data))
# prepare the data
    Features = ["Weather", "Temperature"]

    print("Feature Names are", Features)

# Below Weather, Temperature and Play are called as data series
    Weather = data.Weather
    Temperature = data.Temperature
    play = data.Play

    lobj = preprocessing.LabelEncoder()

    WeatherX = lobj.fit_transform(Weather)
    TemperatureX = lobj.fit_transform(Temperature)
    Labels = lobj.fit_transform(play)


    print ("Encoded Weather is : ", WeatherX)
    print("Encoded Temperature is : ", TemperatureX)

    # A = [1,2,3,4,5]
    # B = [11,21,51,101,111]
    # Zip=[[1,11],[2,21],[3,51],[4,101],[5,111]]

    features = list(zip(WeatherX, TemperatureX))
# Step # 3

    obj = KNeighborsClassifier(n_neighbors=3)
    obj.fit(features, Labels)

# Step # 4
    output = obj.predict([[0,2]])
    if output :
        print("You can Play")
    else:
        print("You can't Play")



def main():
    print("----------- Marvellous Play Predictor")
    pathfilename=input("Enter the filename which contains dataset")
    MarvellousPredictor(pathfilename)

if __name__=="__main__":
    main()
