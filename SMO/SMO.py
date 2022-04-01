import random
import numpy as np # Linear Algebra
import pandas as pd # Loading and manipulating csv data
import matplotlib.pyplot as plt # Plotting results
from sklearn.model_selection import train_test_split # Use to split training and testing data

ALPHAS_MAX = 100
EPS = 0.01

class SMO:
    def __init__(self, xData, yData):
        self.xData = xData
        self.yData = yData
        self.numData = len(yData)
        
        #1: Initialize alphas subject to constraint, b
        alphas = np.random.randint(low=1, high=(ALPHAS_MAX+1), size=self.numData)
        
        positveYs = np.where(yData == 1)[0]
        negativeYs = np.where(yData == -1)[0]
        while np.dot(alphas, yData) != 0:
            difference = np.dot(alphas, yData)
            if (difference < 0):
                # Add to alpha corresponding at positive y
                index = random.choice(positveYs)
                difference *= -1
            elif (difference > 0):
                # Add to alpha corresponding at negative y
                index = random.choice(negativeYs)
            
            if (alphas[index] < ALPHAS_MAX):
                if (difference + alphas[index] > ALPHAS_MAX):
                    alphas[index] += np.random.randint(low=0, high=(ALPHAS_MAX - alphas[index] + 1), size=1)
                else:
                    alphas[index] += difference
        
        self.alphas = alphas
        self.b = 0
        self.weights = [0, 0]
    
    def K(self, x1, x2):
        return np.dot(x1, x2)
    
    def CalcError(self, xi, yi):
        sum = 0
        for j in range(self.numData):
            sum += self.alphas[j] * self.yData[j] * self.K(self.xData[j], xi)
        return sum - yi
    
    def UpdateWeights(self):
        # Step 2: Calculate weight vector
        self.weights[0] = np.sum(self.alphas * self.yData * self.xData[:,0])
        self.weights[1] = np.sum(self.alphas * self.yData * self.xData[:,1])
    
    def CalcKKT(self):
        # Step 3: Calculate KKT Conditions
        KKT = np.zeros(self.numData)
        for i in range(self.numData):
            KKT[i] = self.alphas[i] * (self.yData[i] * (np.dot(self.weights, self.xData[i]) + self.b) - 1)
            #KKT[i] = (self.yData[i] * (np.dot(self.weights, self.xData[i]) + self.b) - 1) # Try without alpha?
        return KKT
    
    def Classify(self, xVal):
        return np.sign(np.sum(self.alphas * self.yData * np.dot(self.xData, xVal)) + self.b)
    
    def Train(self):
        # Step 2: Calculate weight vector - Also happens before updating b below
        self.UpdateWeights()
        done = False
        while not done:
            oldAlphas = self.alphas.copy()
            
            # Step 3: Calculate KKT conditions
            KKT = self.CalcKKT()
            
            E = np.zeros(self.numData)
            for i in range(self.numData):
                E[i] = self.CalcError(self.xData[i], self.yData[i])
            
            # Step 4: Pick x1, x2
            i1 = np.argmax(KKT)
            x1 = self.xData[i1]
            
            e = E[i1] - E
            
            i2 = np.argmax(e)
            x2 = self.xData[i2]
            
            if (i1 == i2):
                break # or else divide by zero
            
            k = self.K(x1, x1) + self.K(x2, x2) - 2 * self.K(x1, x2)
            
            # Step 5: Update alpha2
            oldAlpha2 = self.alphas[i2]
            self.alphas[i2] = oldAlpha2 + ((self.yData[i2] * e[i2]) / k)
            
            # Try clipping alpha2 before updating alpha1?
            # Leads to no change and alphas will be done changing, so loop exits below or else will hang forever
            #if self.alphas[i2] < EPS:
            #    self.alphas[i2] = 0.0
            #elif self.alphas[i2] > ALPHAS_MAX:
            #    self.alphas[i2] = ALPHAS_MAX
            
            # Step 6: Update alpha1
            oldAlpha1 = self.alphas[i1]
            self.alphas[i1] = oldAlpha1 + self.yData[i1] * self.yData[i2] * (oldAlpha2 - self.alphas[i2])
            
            # Step 7: alpha < EPS ? -> 0.0
            above0 = []
            for i in range(self.numData):
                if self.alphas[i] < EPS:
                    self.alphas[i] = 0.0
                else:
                    above0.append(i)
            
            # Step 2: Update weights with new alphas
            self.UpdateWeights()
            
            # Step 8: Calculate b from KKT conditions..
            
            #alphaMaxIndex = np.argmax(self.alphas)
            #self.b = self.yData[alphaMaxIndex] - np.dot(self.xData[alphaMaxIndex], self.xData[alphaMaxIndex])
            
            # This is in the SVM_Notes.pdf, (19). Tried a bunch of different ways of updating b, none really working correctly
            numAbove0 = len(above0)
            totalBs = 0
            for i in above0:
                totalBs += self.yData[i] - np.dot(self.xData[i], self.xData[i])
                #totalBs += (KKT[i] / self.alphas[i] + 1) / self.yData[i] - np.dot(self.weights, self.xData[i])
            self.b = totalBs / numAbove0
            
            print("i1: {}  i2: {}  b: {}".format(i1, i2, self.b))
            
            # If alphas don't change they'll never change again, loop will run forever on same i1 i2
            if np.array_equal(oldAlphas, self.alphas):
                    done = True
                    continue
            
            # Check for full classification
            for i in range(self.numData):              
                if self.yData[i] != self.Classify(self.xData[i]):
                    done = False
                    break
                done = True
        
        print("Alphas")
        print(self.alphas)
        print("Weights: {}".format(self.weights))
        print("b: {}".format(self.b))

def Main():
    inputFile = pd.read_csv(r"data.txt", header=None, delim_whitespace=True)
    
    xData = pd.DataFrame(inputFile, columns=[0, 1]).to_numpy();
    yData = pd.DataFrame(inputFile, columns=[2]).to_numpy();
    
    # Plot data
    marker = "o"
    for classificationValue in [-1, 1]:
        rows = np.where(yData == classificationValue)
        plt.scatter(xData[rows, 0], xData[rows ,1], marker=marker, label="{}".format(classificationValue))
        marker = "*"
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Scatter plot of linearly separable data")
    plt.legend()
    plt.show()
    
    #np.random.seed(0) # Same alphas every time
    
    #xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.20, random_state=1) # Same split every time
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.20)
    
    smo = SMO(xTrain, yTrain)
    
    smo.Train()
    
    numTest = len(yTest)
    numCorrect = 0
    for i in range(numTest):
        if (smo.Classify(xTest[i]) == yTest[i]):
            numCorrect += 1
    
    accuracy = numCorrect / numTest
    print("Accuracy of test set = {}".format(accuracy))

if __name__ == "__main__":
    Main()