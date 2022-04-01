#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # module that helps splitting training and testing data sets randomly

# Find theta for the normal equation
def find_theta(x, y):
    # apply the formula in Pg. 3 Training Regression Models
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return theta

# Measure the mean squared error
def mse(theta, x, y):
    mse = 0
    x.insert(0, "bias", np.ones(len(x)), True) # adding bias term for the input var
    for i in range(0, len(x)):
        mse += (np.square(theta.T.dot(x.values[i])-y.values[i]))
    mse /= len(x)
    return mse

if __name__ == '__main__':
    inputFile = pd.read_csv("customer_info.csv") # read csv file 
    inputData = pd.DataFrame(inputFile, columns = ["bmi", "age", "children"]) # get the specific input var
    outputData = pd.DataFrame(inputFile, columns = ["charges"]) # get the output var
    testRatio = 0.2 # initial size for training set
    # Creating list of training and testing MSE for ploting generalizaton
    listTrainMSE = []
    listTestMSE = []
    listTrainNumb = []
    listTestNumb = []
    while testRatio < 0.81: # increment until the training setsize reach 80%
        # Splitting training set and test set randomly
        inputTest, inputTrain, outputTest, outputTrain = train_test_split(inputData, outputData, test_size = testRatio)
        # Clone the input training data for adding bias term, so we can keep the og one for plotting
        inputTrainOne = inputTrain.copy()
        inputTrainOne.insert(0, "bias", np.ones(len(inputTrain)), True) # adding bias term
        theta = find_theta(inputTrainOne, outputTrain) # find theta
        # Plotting linear regression for each bmi, age and children attr among 20%, 30%,...,80% size
        attrList = ["bmi", "age", "children"]
        for i in range(len(attrList)):
            plt.figure()
            plt.scatter(inputTrain.iloc[:,i], outputTrain, s = 30, marker = "o")
            plt.plot(inputTrain.iloc[:,i], theta[0] + theta[i+1] * inputTrain.iloc[:,i], color = "red")
            title = "Linear Regression for " + attrList[i] + " attribute with ratio " + str(testRatio)
            plt.title(title)
            plt.show()
        # Calculate mean square error:
        trainMSE = mse(theta, inputTrain, outputTrain)
        testMSE = mse(theta, inputTest, outputTest)
        listTrainMSE.append(trainMSE)
        listTestMSE.append(testMSE)
        listTrainNumb.append(len(inputTrain))
        listTestNumb.append(len(inputTest))
        testRatio += 0.1
    # Plotting the generalization
    plt.figure()
    plt.plot(listTrainNumb, listTrainMSE, color = "red", label = "training set")
    plt.plot(listTestNumb, listTestMSE, color = "blue", label = "validation set")
    title = "modeling and generalization for attribute"
    plt.title(title)
    plt.xlabel("Training set size")
    plt.ylabel("MSE")
    plt.legend()
