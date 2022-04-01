import random
import numpy as np # Linear Algebra
import pandas as pd # Loading and manipulating csv data
import matplotlib.pyplot as plt # Plotting results
from sklearn.model_selection import train_test_split # Use to split training and testing data

LEARNING_RATE = 0.01
CONVERGENCE = 0.01
SAVE_FIGS = False
lamb = -1

def ScatterPlotAllColumns(x, y, title):
    for col in x.columns:
        plt.scatter(x[col], y, s = 30, marker = "o")
        plt.xlabel(col)
        plt.ylabel(y.columns[0])
        plt.title(title)
        
        if SAVE_FIGS:
            plt.savefig("Figures/" + title + " - " + col + ".png")
            plt.clf()
        else:
            plt.show()

def LinePlotAllColumns(x, y, title):
    for col in x.columns:
        plt.plot(x[col], y)
    
    plt.xlabel("Attributes")
    plt.ylabel("Charges")
    plt.title(title)
    
    plt.show()
        

def PlotAllAttributes(x, y, weights, title, removedIndex=-1):
    if removedIndex > -1:
        weights = weights.tolist()
        weights.insert(removedIndex, 0) # Insert zero value so indices aren't mixed up
    
    i = 1
    for col in x.columns:
        if i == removedIndex:
            i += 1
            continue # Skip plotting of removed attribute from regularization
        
        plt.scatter(x[col], y, s = 30, marker = "o")
        plt.plot(x, weights[0] + weights[i] * x, color = "red")
        plt.xlabel(col)
        plt.ylabel(y.columns[0])
        plt.title(title)
        
        if SAVE_FIGS:
            plt.savefig("Figures/" + title + " - " + col + ".png")
            plt.clf()
        else:
            plt.show()
        
        i += 1

# Print MSE for test set and train set
def compareMSE(xTrainOnes, xTest, yTrain, yTest, w, name, typeR = 'None', removedIndex=-1):
    if removedIndex == -1:
        yHat = xTrainOnes.to_numpy().dot(w)
        mse = MSE(yTrain["charges"], yHat)
        print('MSE of the ' + name + ' wo regularization for train: ' + str(mse))

        xTestOnes = xTest.copy()
        xTestOnes.insert(0, "ones", np.ones(len(xTest)), True)
        yHat = xTestOnes.to_numpy().dot(w)
        mse = MSE(yTest["charges"], yHat)
        print('MSE of the ' + name + ' wo regularization for test: ' + str(mse))
        print('\n \n')
    else:
        yHat = xTrainOnes.drop(xTrainOnes.columns[removedIndex], axis = 1).to_numpy().dot(w)
        mse = MSE(yTrain["charges"], yHat)
        print('MSE of the ' + name + ' w ' + typeR + ' regularization for train: ' + str(mse))

        xTestOnes = xTest.copy()
        xTestOnes.insert(0, "ones", np.ones(len(xTest)), True)
        yHat = xTestOnes.drop(xTestOnes.columns[removedIndex], axis = 1).to_numpy().dot(w)
        mse = MSE(yTest["charges"], yHat)
        print('MSE of the ' + name + ' w ' + typeR + ' regularization for test: ' + str(mse))
        print('\n \n')
        
def MSE(yTrain, yHat):
    return np.mean(np.square(yTrain - yHat))

def GradientMSE(xTrain, yTrain, w, lamb=0, reg="None"):
    if reg == "L2":
        return (2 / len(xTrain)) * xTrain.T.dot(xTrain.dot(w) - yTrain) + 2 * lamb * w
    elif reg == "L1":
        return (2 / len(xTrain)) * xTrain.T.dot(xTrain.dot(w) - yTrain) + 2 * lamb * np.sign(w)
    else:
        return (2 / len(xTrain)) * xTrain.T.dot(xTrain.dot(w) - yTrain)

def GradientDescent(xTrain, yTrain, w, lamb=-1, reg="None"):
    if reg=="None":
        gradMSE = GradientMSE(xTrain, yTrain, w, reg)
        return w - LEARNING_RATE * gradMSE, lamb
    else:
        # Bias weight needs to be calculated separately
        gradMSEBias = GradientMSE(xTrain[:,0], yTrain, w[0])
        w[0] = w[0] - LEARNING_RATE * gradMSEBias
        
        if lamb == -1: # First time we will use validation to figure a lambda constant
            lamb = 0.1
            wMin = w.copy()
            wTemp = w.copy()
            lambMin = lamb
            while lamb <= 10.1: # Try a few different lambdas and choose the one resulting in min testMSE
                gradMSER = GradientMSE(xTrain[:,1:], yTrain, w[1:], lamb, reg)
                wTemp[1:] = w[1:] - LEARNING_RATE * gradMSER
                
                # Split the training set into two subset: subtest and subtrain
                xTempTrain, xTempTest, yTempTrain, yTempTest = train_test_split(xTrain, yTrain, test_size = 0.1)
                
                # Find the smaller subtest-mse by the current w-vector, through lambda = 0.1, 1.1, ... 10.1, also get that lambda
                yHatTest = xTempTest.dot(w)
                yHatTestMin = xTempTest.dot(wMin)
                testMSE = MSE(yTempTest, yHatTest)
                testMSEMin = MSE(yTempTest, yHatTestMin)
                if testMSE < testMSEMin: 
                    wMin = w.copy()
                    lambMin = lamb
                # Increment lambda by 1
                lamb += 1
            return wMin, lambMin
        else: # Every other time use the same one
            gradMSER = GradientMSE(xTrain[:,1:], yTrain, w[1:], lamb, reg)
            w[1:] = w[1:] - LEARNING_RATE * gradMSER
            return w, lamb

def BatchGradientDescent(xTrain, yTrain, epochs, reg="None", removedInd=-1):
    global lamb
    m = xTrain.shape[0]
    w = np.ones(xTrain.shape[1]) # "Randomly" fill w with ones
    
    for i in range(epochs):
        yHat = xTrain.dot(w)
        
        trainMSE = MSE(yTrain, yHat)
        if trainMSE < CONVERGENCE:
            break
        
        w, lamb = GradientDescent(xTrain, yTrain, w, lamb, reg)
    
    if reg == "None":
        return w, removedInd
    else:
        # Do not remove bias column/attribute
        wAbs = np.abs(w[1:])
        minIndex = wAbs.argmin() + 1
        xTrain = np.delete(xTrain, minIndex, 1)
        return BatchGradientDescent(xTrain, yTrain, epochs, "None", minIndex)

def StochasticGradientDescent(xTrain, yTrain, epochs, reg="None", removedInd=-1):
    global lamb
    m = xTrain.shape[0]
    w = np.ones(xTrain.shape[1]) # "Randomly" fill w with ones
    
    for i in range(epochs):
        # Choose a random index each impox and train with only that
        randInd = random.randint(0, m-1)
        
        randX = np.array(xTrain[randInd], ndmin=2)
        randY = yTrain.values[randInd]
        
        w, lamb = GradientDescent(randX, randY, w, lamb, reg)
    
    if reg == "None":
        return w, removedInd
    else:
        # Do not remove bias column/attribute
        wAbs = np.abs(w[1:])
        minIndex = wAbs.argmin() + 1
        xTrain = np.delete(xTrain, minIndex, 1)
        return StochasticGradientDescent(xTrain, yTrain, epochs, "None", minIndex)

def MiniBatchGradientDescent(xTrain, yTrain, epochs, reg="None", removedInd=-1):
    global lamb
    m = xTrain.shape[0]
    w = np.ones(xTrain.shape[1]) # "Randomly" fill w with ones
    
    for i in range(epochs):    
        # Fill lists with a random 5% batch of training set
        numBatch = int(m * 0.05)
        randInds = np.random.choice(m, size=numBatch, replace=False)
        xTrainBatch = xTrain[randInds, :]
        yTrainBatch = yTrain.values[randInds]
        
        yHat = xTrainBatch.dot(w)
        
        trainMSE = MSE(yTrainBatch, yHat)
        if trainMSE < CONVERGENCE:
            break
        
        w, lamb = GradientDescent(xTrainBatch, yTrainBatch, w, lamb, reg)
    
    if reg == "None":
        return w, removedInd
    else:
        # Do not remove bias column/attribute
        wAbs = np.abs(w[1:])
        minIndex = wAbs.argmin() + 1
        xTrain = np.delete(xTrain, minIndex, 1)
        return MiniBatchGradientDescent(xTrain, yTrain, epochs, "None", minIndex)

def Main():
    # Load the input file
    inputFile = pd.read_csv(r"customer_info.csv")

    # Dictionaries to convert data to numerical
    sex = {"male": 1, "female": 2}
    smoker = {"no": 1, "yes": 2}

    inputFile.sex = [sex[item] for item in inputFile.sex]
    inputFile.smoker = [smoker[item] for item in inputFile.smoker]

    # Grab the data from the input
    inputData = pd.DataFrame(inputFile, columns= ["age", "sex", "bmi", "children", "smoker"])
    outputData = pd.DataFrame(inputFile, columns= ["charges"])
    
    # Plot all of the data BEFORE proceeding
    ScatterPlotAllColumns(inputData, outputData, "Data Before Normalization")
    
    # Normalize the data
    normalizedInputData = (inputData - inputData.mean()) / inputData.std()
    
    # Plot all of the data AFTER normalization
    ScatterPlotAllColumns(normalizedInputData, outputData, "Data After Normalization")
    
    # Show all attributes on one plot
    LinePlotAllColumns(normalizedInputData, outputData, "Attributes After Normalization")
    
    # Split and save the data 50/50
    xTrain, xTest, yTrain, yTest = train_test_split(normalizedInputData, outputData, test_size=0.5)
    
    # Insert column  of ones to calculate intercept
    xTrainOnes = xTrain.copy()
    xTrainOnes.insert(0, "ones", np.ones(len(xTrain)), True)
    
    # Run the Gradient Descent algorithms
    # wo Regularization
    # Batch Gradient Descent
    # Batch Gradient Descent needs to be performed first so that lambda value can be calculated and set
    w1, removedIndex1 = BatchGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 1000)
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w1, "Batch Gradient Descent")
    PlotAllAttributes(xTrain, yTrain, w1, "Batch - No Regularization", removedIndex1)
    
    # Stochastic Gradient Descent
    w2,removedIndex2 = StochasticGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 10000)
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w2, "Stochastic Gradient Descent")
    PlotAllAttributes(xTrain, yTrain, w2, "Stochastic - No Regularization", removedIndex2)
    
    # Mini-batch Gradient Descent
    w3, removedIndex3 = MiniBatchGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 1000)
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w3, "Mini batch Gradient Descent")
    PlotAllAttributes(xTrain, yTrain, w3, "Mini-Batch - No Regularization", removedIndex3)
    
    #L2 Regularization
    # Batch Gradient Descent
    w4, removedIndex4 = BatchGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 1000, "L2")
    # Divide the data sets into training and test subsets of approximately equal,
    # and train again, using the attributes selected by the L1-regularization step.
    # repeat for other types of GD
    xTrainTemp, xTestTemp, yTrainTemp, yTestTemp = train_test_split(normalizedInputData, outputData, test_size=0.5)
    xTrainOnesTemp = xTrainTemp.copy()
    xTrainOnesTemp.insert(0, "ones", np.ones(len(xTrainTemp)), True)
    xTrainOnesTemp = xTrainOnesTemp.drop(xTrainOnesTemp.columns[removedIndex4], axis = 1)
    w4, ignore = BatchGradientDescent(xTrainOnesTemp.to_numpy(), yTrainTemp["charges"], 1000)
    #print MSE for test and train set
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w4, "Batch Gradient Descent", 'L2', removedIndex4)
    PlotAllAttributes(xTrain, yTrain, w4, "Batch - L2 Regularization", removedIndex4)

    # Stochastic Gradient Descent
    w5, removedIndex5 = StochasticGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 10000, "L2")
    xTrainOnesTemp = xTrainTemp.copy()
    xTrainOnesTemp.insert(0, "ones", np.ones(len(xTrainTemp)), True)
    xTrainOnesTemp = xTrainOnesTemp.drop(xTrainOnesTemp.columns[removedIndex5], axis = 1)
    w5, ignore = StochasticGradientDescent(xTrainOnesTemp.to_numpy(), yTrainTemp["charges"], 10000)
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w5, "Stochastic Gradient Descent", 'L2', removedIndex5)
    PlotAllAttributes(xTrain, yTrain, w5, "Stochastic - L2 Regularization", removedIndex5)
    
    # Mini-batch Gradient Descent
    w6, removedIndex6 = MiniBatchGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 1000, "L2")
    xTrainOnesTemp = xTrainTemp.copy()
    xTrainOnesTemp.insert(0, "ones", np.ones(len(xTrainTemp)), True)
    xTrainOnesTemp = xTrainOnesTemp.drop(xTrainOnesTemp.columns[removedIndex6], axis = 1)
    w6, ignore = MiniBatchGradientDescent(xTrainOnesTemp.to_numpy(), yTrainTemp["charges"], 1000)
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w6, "Mini-batch Gradient Descent", 'L2', removedIndex6)
    PlotAllAttributes(xTrain, yTrain, w6, "Mini-Batch - L2 Regularization", removedIndex6)
    
    #L1 Regularization
    # Batch Gradient Descent
    w7, removedIndex7 = BatchGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 1000, "L1")
    xTrainOnesTemp = xTrainTemp.copy()
    xTrainOnesTemp.insert(0, "ones", np.ones(len(xTrainTemp)), True)
    xTrainOnesTemp = xTrainOnesTemp.drop(xTrainOnesTemp.columns[removedIndex7], axis = 1)
    w7, ignore = BatchGradientDescent(xTrainOnesTemp.to_numpy(), yTrainTemp["charges"], 1000)
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w7, "Batch Gradient Descent", 'L1', removedIndex7)
    PlotAllAttributes(xTrain, yTrain, w7, "Batch - L1 Regularization", removedIndex7)
    
    # Stochastic Gradient Descent
    w8, removedIndex8 = StochasticGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 10000, "L1")
    xTrainOnesTemp = xTrainTemp.copy()
    xTrainOnesTemp.insert(0, "ones", np.ones(len(xTrainTemp)), True)
    xTrainOnesTemp = xTrainOnesTemp.drop(xTrainOnesTemp.columns[removedIndex8], axis = 1)
    w8, ignore = StochasticGradientDescent(xTrainOnesTemp.to_numpy(), yTrainTemp["charges"], 10000)
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w8, "Stochastic Gradient Descent", 'L1', removedIndex8)
    PlotAllAttributes(xTrain, yTrain, w8, "Stochastic - L1 Regularization", removedIndex8)
    
    # Mini-batch Gradient Descent
    w9, removedIndex9 = MiniBatchGradientDescent(xTrainOnes.to_numpy(), yTrain["charges"], 1000, "L1")
    xTrainOnesTemp = xTrainTemp.copy()
    xTrainOnesTemp.insert(0, "ones", np.ones(len(xTrainTemp)), True)
    xTrainOnesTemp = xTrainOnesTemp.drop(xTrainOnesTemp.columns[removedIndex9], axis = 1)
    w9, ignore = MiniBatchGradientDescent(xTrainOnesTemp.to_numpy(), yTrainTemp["charges"], 1000)
    compareMSE(xTrainOnes, xTest, yTrain, yTest, w9, "Mini-batch Gradient Descent", 'L1', removedIndex9)
    PlotAllAttributes(xTrain, yTrain, w9, "Mini-Batch - L1 Regularization", removedIndex9)


if __name__ == "__main__":
    Main()