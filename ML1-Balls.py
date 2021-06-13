from sklearn import tree

import sklearn

# Rough 1
# Smooth 0
# Tennis 1
# Cricket 2

def MarvellousML(weight,surface):
    # Step 1 and 2

    #    Features = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],
    #                [48,"Rough"],[90,"Smooth"],[35,"Rough"],
    #                [92,"Smooth"],[35,"Rough"],[35,"Rough"],
    #                [35,"Rough"], [96,"Smooth"],[43,"Rough"],
    #                [110,"Smooth"],[35,"Rough"],[95,"Smooth"]]

    #    Labels = ["Tennis","Tennis","Cricket",
    #              "Tennis","Cricket","Tennis",
    #              "Cricket","Tennis","Tennis",
    #              "Tennis","Cricket","Tennis",
    #              "Cricket","Tennis","Cricket"]

# Volume of the data set is 15

    Features = [[35, 1], [47, 1], [90, 0],
                [48, 1], [90, 0], [35, 1],
                [92, 0], [35, 1], [35, 1],
                [35, 1], [96, 0], [43, 1],
                [110, 0], [35, 1], [95, 0]]

    Labels = [1, 1, 2,
              1, 2, 1,
              2, 1, 1,
              1, 2, 1,
              2, 1, 2]

    # Step 3 In below line, tree is a module and DecisionTreeClassifier is a class.
    # dobj is an object of class DecisionTreeClassifier

    dobj = tree.DecisionTreeClassifier()

    # Step 4 fit method is used to train the datasets
    dobj.fit(Features, Labels)

# Step 5
    result=dobj.predict([[weight,surface]]) # result=dobj.predict([[weight,surface],[w1,s1],[w2,s2],[w3,s3]]
    # result = result[2,2,1,2]  example
    if result==1:
        print("Your Ball looks like to be Tennis Ball")
    else:
        print("Your Ball looks like to be Cricket Ball")


def main():
    print("Jay Ganesh")
    print("This is the example of Supervised Machine Learning")

    weight = int(input("Enter the weight : "))
    surface = input("Enter the type of Surface : ")

    if surface.lower()=="rough":
        surface=1
    elif surface.lower()=="smooth":
        surface=0
    else:
        print("Invalid Input for Surface")
        return


    MarvellousML(weight,surface)



if __name__=="__main__":
    main()
