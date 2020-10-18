
# class for a simple implementation of the KNN algorithm
## Author: Devin Gluesing

import numpy
import random
import time
import os

class KNN():
    def __init__(self, test_file, train_file):
        self.testing_set = numpy.loadtxt(test_file)
        self.training_set = numpy.loadtxt(train_file)

        self.tp = 0 #True Positive
        self.fp = 0 #False Positive
        self.tn = 0 #True Negative
        self.fn = 0 #False Negative

        self.predictions = []
        self.actual = []


    def GeneratePredictions(self, k):
        for test_row in self.testing_set:
            prediction = self.MakePrediction(self.training_set, test_row, k)
            self.predictions.append(prediction)
            self.actual.append(test_row[-1])

    def GatherStats(self):
        for i in range(len(self.predictions)-1):
            currPrediction = self.predictions[i]
            currActual = self.actual[i]
            if  currPrediction == 1 and currActual == 1:
                self.tp += 1
            elif currPrediction == 1 and currActual == -1:
                self.fp += 1
            elif currPrediction == -1 and currActual == -1:
                self.tn += 1
            elif currPrediction == -1 and currActual == 1:
                self.fn +=1
            else:
                print("[-] done fucked up")

    def GenerateTestResults(self, label):
        accuracy = str(self.Accuracy())
        sensitivity = str(self.Sensitivity())
        specificity = str(self.Specificity())
        precision = str(self.Precision())
        info =  "Results for {0}:\n".format(label)
        info += "Accuracy: {0}\n".format(accuracy)
        info += "Sensitivity: {0}\n".format(sensitivity)
        info += "Specificity: {0}\n".format(specificity)
        info += "Precision: {0}\n\n".format(precision)
        return info

    def ClearMemory(self):
        self.predictions = []
        self.actual = []
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def CalculateDistance(self, x, y):
        distance = 0.0
        for i in range(len(x)-1):
            # print(x[i], y[i])
            distance += (x[i] - y[i])**2
        return (distance**(1/2))

    def GetNeighbors(self, training_set, test_row, k):
        distances = []
        for train_row in training_set:
            distance =  self.CalculateDistance(test_row, train_row)
            distances.append((train_row, distance))
        distances.sort(key=lambda tup: tup[1])
        # distances.sort()
        # print(distances)
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][0])
            # neighbors.append(distances[i])
        return neighbors

    def MakePrediction(self, training_set, test_row, k):
        neighbors =  self.GetNeighbors(training_set, test_row, k)
        output_vals = []
        for row in neighbors:
            output_vals.append(row[-1])
        prediction = max(set(output_vals), key=output_vals.count)
        return prediction

    # functions written to be easily readable
    def Accuracy(self):
        top = self.tp + self.tn
        bottom = self.tp + self.fp + self.tn + self.fn
        return (top / bottom)

    def Sensitivity(self):
        bottom = self.tp + self.fn
        return (self.tp / bottom)

    def Specificity(self):
        bottom = self.fp + self.tn
        return (self.tn / bottom)

    def Precision(self):
        bottom = self.tp + self.fp
        return (self.tp / bottom)
