from knn import knn
from scipy.io import arff
import pandas as pd
import numpy as np
import os

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

file = 'PhishingData.arff'
newFile = ''
with open(file , "r") as inFile:
    content = inFile.readlines()
    name,ext = os.path.splitext(inFile.name)
    new = toCsv(content)
    newFile = name + ".csv"
    with open(newFile, "w") as outFile:
        outFile.writelines(new)

df = pd.read_csv(newFile)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
l = round(len(X)*0.8)
train_X = np.array(X[:l])
train_Y = np.array(Y[:l])
test_X = np.array(X[l:])
test_Y = np.array(Y[l:])

oop = knn(train_X, train_Y, test_X, test_Y)
for i in range(2,13):
    a = oop.fit(i)
    b = oop.getAccuracy()
    print("k = ", i, ":", b)
