import pandas as pd

def main():
    data=pd.read_csv("iris.csv")
    print("Length of data :", len(data))
    print(data.head())
#    print(data)

if __name__ == "__main__":
    main()
