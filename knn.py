from classifier import classifier
import numpy as np

class knn(classifier):

    def __init__(self, train_X, train_Labels):
        self.X = train_X
        self.Y = train_Labels
        self.new_Y = []
        self.kn = None

    def fit(self, kn):
        self.kn = kn
        for i in range(len(self.X)):
            label = self.getLabel(self.X.iloc[i])
            self.new_Y.append(label)
        return self.new_Y

    def calAcc(self):
        l1 = len(self.Y)
        l2 = 0
        for i in range(l1):
            if self.Y[i] == self.new_Y[i]:
                l2+=1
        return l2/l1

    def predict(self, test_X):
        hyp_Y = []
        for i in range(len(test_X)):
            label = self.getLabel(test_X.iloc[i])
            hyp_Y.append(label)
        return hyp_Y
    
    def predictOne(self, x):
        return self.getLabel(x)

    def getAllDists(self,x):
        dlist = self.calDist(x)
        index = np.argsort(dlist)
        dlist.sort()
        return dlist, index

    def calDist(self, x):
        x = np.array(x)
        arr = np.array(self.X)
        ds = np.sum((x - arr)**2, axis=1)
        return np.sqrt(ds)


    def getLabel(self,x):
        dlist, index = self.getAllDists(x)
        avg = 0.0
        for i in range(self.kn):
            avg = avg + self.Y[index[i]]
        avg = round(avg / self.kn)
        return avg
