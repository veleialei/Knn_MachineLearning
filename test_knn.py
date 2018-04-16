from knn import knn
from scipy.io import arff
import pandas as pd
import numpy as np
import os

file = 'PhishingData.arff'

def toCsv(content):
    data = False
    header = ""
    newContent = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute")+1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent

# Main loop for reading and writing files

with open(file , "r") as inFile:
    content = inFile.readlines()
    name,ext = os.path.splitext(inFile.name)
    new = toCsv(content)
    newFile = name + ".csv"
    with open(newFile, "w") as outFile:
        outFile.writelines(new)

df = pd.read_csv(newFile) #legitimate(1), suspicious(0) or phishy(-1) based on whether the site contains pop-up windows
train_X = df.iloc[:,:-1]
train_Labels = df.iloc[:,-1]
train = knn(train_X, train_Labels)
train.fit(5)
print(train.calAcc())
test = 225
newY = train.predictOne(train_X.iloc[test])
newYs = train.predict(train_X)
print(newY, train_Labels.iloc[test])
print("============================")
print(newYs)
