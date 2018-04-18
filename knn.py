from classifier import classifier
import numpy as np
import operator

class knn(classifier):
    def __init__(self, train_X, train_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.hypothesis = None
        self.kn = 3

    def fit(self):
        fit(self.kn)

    def fit(self, kn):
        self.hypothesis = []
        self.kn = kn
        l_test = len(self.test_X)
        for i in range(l_test):
            kNears = self.getKNearest(self.test_X[i])
            result = self.getLabel(kNears)
            self.hypothesis.append(result)
        return self.hypothesis

    def predict(self, pred_X):
        l_test = len(pred_X)
        predict_Y = []
        for i in range(l_test):
            kNears = self.getKNearest(pred_X[i])
            result = self.getLabel(kNears)
            predict_Y.append(result)
        return predict_Y

    def getAccuracy(self):
        correct = 0
        length = len(self.test_X)
        for i in range(length):
            if self.test_Y[i] == self.hypothesis[i]:
                correct += 1
        return (correct/float(length))

    def getDist(self, x1, x2):
        a = np.power((x1 - x2), 2)
        s = np.sum(a)
        return np.sqrt(s)

    def getKNearest(self, test_x):
        dists = []
        for i in range(len(self.train_X)):
            dist = self.getDist(test_x, self.train_X[i])
            dists.append((self.train_Y[i], dist))
        dists.sort(key=operator.itemgetter(1))
        neighbors = []
        for i in range(self.kn):
            neighbors.append(dists[i][0])
        return neighbors

    def getLabel(self, neighbors):
        labels = {}
        for i in range(len(neighbors)):
            label = neighbors[i]
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
        sortedLabels = sorted(labels.items(), key=operator.itemgetter(1), reverse=True)
        return sortedLabels[0][0]
