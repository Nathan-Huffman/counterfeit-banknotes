import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn import metrics
import math

# bring in all the data, segmented into portions of 60/20/20
testData = pd.read_csv('test.csv')
trainData = pd.read_csv('training.csv')
validationData = pd.read_csv('validation.csv')
trainValid = pd.read_csv('trainvalid.csv')

trainClassDF = trainData['class']
trainData = trainData.drop('class', axis=1)

validationClassDF = validationData['class']
validationData = validationData.drop('class', axis=1)
trainValidClassDF = trainValid['class']
trainValid = trainValid.drop('class', axis=1)




test = trainValid.to_numpy()
testClass = trainValidClassDF.to_numpy()
sampleSize = 270

stats = []
# Loop through sample sizes from 270 to 50, evaluating models on each sample size
while sampleSize > 40:\
    # get our training data
    # i know it says test data its a misnomer
    testData = testData.sample(n = sampleSize)
    testClassDF = testData['class']
    testDataSample = testData.drop('class', axis=1)
    train = testDataSample.to_numpy()
    trainClass = testClassDF.to_numpy()
    
    # loop through possible k values, evaluating each
    minimumk = 10000
    highestAcc = 0
    for k in range(1, 49):
        predictions = []
        numTestRows, numTestCols = test.shape
        numTrainRows, numTrainCols = train.shape
        # for every test row, calculate the distance between it and every training row
        for testRow in range(0, numTestRows):
            distances = []
            freqZ = 0
            freqO = 0
            match = False
            for i in range(0, numTrainRows):
                # calculate the distance
                distance = euclidean(test[testRow], train[i])
                distances.append([distance, trainClass[i]])

            # sort by distance high to low
            distances.sort()
            
            # add to the frequencies, weighted with the distances themselves
            for j in range(0, k):
                if(distances[j][0] != 0):
                    if(distances[j][1] == 0):
                        freqZ += (1 / math.sqrt(distances[j][0]))
                    else:
                        freqO += (1 / math.sqrt(distances[j][0]))
                else:
                    match = True
                    pred = distances[j][1]

            # 
            if(match != True):
                if(freqO > freqZ):
                    pred = 1
                else:
                    pred = 0

            predictions.append(pred)   
        # if this model had a higher accuracy than the highest previously, keep track of it 
        if(metrics.accuracy_score(testClass, predictions) > highestAcc):
            minimumk = k
            highestAcc = metrics.accuracy_score(testClass, predictions)
            # we want the lowest k with highest accuracy
            # if we max out accuracy early, just stop checking later ks
            if(highestAcc == 1):
                break
    # record some info
    print("For sample size: ", sampleSize)
    print("Minimum k value: ", minimumk, " with accuracy: ", highestAcc)
    stats.append([sampleSize, minimumk, highestAcc])
    
    # confusion matrix stuff commented out for clarity
    # confusion = [[0,0],[0,0]]
    # for i in range(0, numTestRows):
    #     if(predictions[i] == 0 and testClass[i] == 0):
    #         confusion[0][0] += 1
    #     elif(predictions[i] == 0 and testClass[i] == 1):
    #         confusion[1][0] += 1
    #     elif(predictions[i] == 1 and testClass[i] == 0):
    #         confusion[0][1] += 1
    #     elif(predictions[i] == 1 and testClass[i] == 1):
    #         confusion[1][1] += 1
    # tpr = confusion[1][1] / (confusion[1][1] + confusion[1][0])
    # fpr = 1 - (confusion[0][0] / (confusion[0][0] + confusion[0][1]))

    # print("Homebrew accuracy with ", k, " neighbors: ", metrics.accuracy_score(testClass, predictions))
    # print("TPR = ", tpr, " FPR = ", fpr)
    # print(confusion)

    # lower the sample size
    sampleSize = sampleSize - 10


print(stats)