# CS 6037 - MACHINE LEARNING
# Assignment 4
#
# Ryan Green
# Minh Nguyen
# Rizul Sharma
# Vincent Tong

import timeit # Timing the functions
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # Plotting results

class DeltaRulePerceptron:
    def __init__(self, trainType):
        self.w = [0.0, 0.0, 0.0] # Weight vector
        self.trainType = trainType # Batch or Stochastic
        self.trainingErrors = [] # Training errors every epoch
        self.numEpochs = 0
        self.learnRate = 0
        self.recentPredictions = [] # Set of most recent epoch predictions (Decision Surface)
        self.inputData = []
        self.outputData = []
        self.numWeightUpdates = 0
    
    def Activation(self, point):
        activation = self.w[0] + self.w[1] * point[0] + self.w[2] + point[1]
        return 1 if activation > 0 else -1
    
    def Predict(self, inputData):
        numDataPoints = len(inputData)
        predictions = np.zeros(numDataPoints)
        
        for i in range(len(inputData)):
            predictions[i] = self.Activation(inputData[i])
        return predictions
    
    def Train(self, inputData, outputData, numEpochs, learnRate, decay=False, decayRate=1, adaptiveRate=False, t=0.0, d=1.0, D=1.0):
        numDataPoints = len(inputData)
        
        # Save some data for plotting
        self.numEpochs = numEpochs
        self.learnRate = learnRate
        self.inputData = inputData
        self.outputData = outputData
        
        for epoch in range(numEpochs):
            predictions = np.zeros(numDataPoints)
            
            for i in range(numDataPoints):
                prediction = self.Activation(inputData[i])
                
                if (self.trainType == "Batch"):
                    predictions[i] = prediction
                elif (self.trainType == "Stochastic"):
                    # Update weights based on this sample
                    if (prediction != outputData[i]):
                        error = (outputData[i] - prediction)
                        
                        # Update bias separately
                        self.w[0] = self.w[0] + learnRate * error
                        self.w[1:] = self.w[1:] + learnRate * error * inputData[i]
                        self.numWeightUpdates += 1;
            
            if (self.trainType == "Batch"):
                # Update weights once at the end of every epoch
                
                trainingError = 0
                deltaW = [0.0, 0.0, 0.0]
                
                for i in range(numDataPoints):
                    prediction = predictions[i]
                    if (prediction != outputData[i]):
                        error = (outputData[i] - prediction)
                        
                        trainingError += (error**2) / 2
                        
                        # Update bias separately
                        deltaW[0] += learnRate * error
                        deltaW[1:] += learnRate * error * inputData[i]
                
                if (adaptiveRate == True and epoch >= 1):
                    prevError = self.trainingErrors[epoch-1]
                    
                    if (trainingError > (prevError + t)): # Training error exceeds previous by a threshold
                        deltaW[:] = [0.0, 0.0, 0.0] # Discard new weights
                        learnRate *= d
                    elif (trainingError < prevError): # Training error smaller than previous
                        learnRate *= D
                        
                
                # Update weights from deltaW
                for i in range(3):
                    self.w[i] += deltaW[i]
                self.numWeightUpdates += 1;
                
                # Save metrics for plotting
                self.trainingErrors.append(trainingError)
                self.recentPredictions = predictions
            
            if decay == True:
                learnRate *= decayRate
    
    def PlotErrorOverEpochs(self, label=""):
        if label == "":
            label = "#Epochs: {}  LearnRate: {}".format(self.numEpochs, self.learnRate)
        
        plt.plot(range(1, self.numEpochs + 1), self.trainingErrors, label=label)
    
    def PlotDecisionSurfaceLine(self, color="red"):
        # This function uses concepts from https://machinelearningmastery.com/plot-a-decision-surface-for-machine-learning/
        minX1, maxX1 = self.inputData[:,0].min()-1, self.inputData[:,0].max()+1
        minX2, maxX2 = self.inputData[:,1].min()-1, self.inputData[:,1].max()+1
        
        x1Grid = np.arange(minX1, maxX1, 0.5)
        x2Grid = np.arange(minX2, maxX2, 0.5)
        
        xCoords, yCoords = np.meshgrid(x1Grid, x2Grid)
        
        r1, r2 = xCoords.flatten(), yCoords.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        
        grid = np.hstack((r1, r2))
        
        predicted = self.Predict(grid)
        
        zVals = predicted.reshape(xCoords.shape)
        
        CS = plt.contour(xCoords, yCoords, zVals, colors=color, linestyles="solid")
        CS.collections[0].set_label("#Epochs: {}".format(self.numEpochs))

# data is randomly generating by formula x + 3y - 2
def GenerateData(numPoints):
    percentPositive = 0
    
    # Ensure that 49-51% are positive (49-51% would be negative)
    while (percentPositive) < 0.49 or percentPositive > 0.51:
        countPositive = 0
        inputData = np.random.randint(low=-40, high=41, size=(numPoints, 2))
        outputData = np.zeros(numPoints)
        
        for i in range(numPoints):
            if (inputData[i][0] + 3 * inputData[i][1] - 2) > 0:
                outputData[i] = 1
                countPositive += 1
            else:
                outputData[i] = -1
        percentPositive = countPositive / numPoints
    
    return inputData, outputData

def Main():
    inputData, outputData = GenerateData(200)
    
    # Problem 1
    # Part A
    model = DeltaRulePerceptron("Batch")
    model.Train(inputData, outputData, 25, 0.01)
    
    model.PlotErrorOverEpochs()
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Error")
    plt.title("Epoch Number vs. Training Error for 25 Epochs")
    plt.show()
    
    # Part B
    colors = ["red", "orange", "blue", "green"]
    colorInd = 0
    learningRate = 0.00075
    for numEpochs in [5, 10, 50, 100]:
        model = DeltaRulePerceptron("Batch")
        model.Train(inputData, outputData, numEpochs, learningRate)
        model.PlotDecisionSurfaceLine(colors[colorInd])
        colorInd += 1
    
    for classificationValue in [-1, 1]:
        rows = np.where(outputData == classificationValue)
        plt.scatter(inputData[rows, 0], inputData[rows ,1], cmap='Paired')
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Surface over different Epochs, LearningRate: {}".format(learningRate))
    plt.show()
    
    # Part C
    minError = 999
    minRate = 0
    for learningRate in [0.1, 0.01, 0.001, 0.0001]:
        model = DeltaRulePerceptron("Batch")
        model.Train(inputData, outputData, 50, learningRate)
        model.PlotErrorOverEpochs()
        if model.trainingErrors[-1] < minError:
            minError = model.trainingErrors[-1]
            minRate = learningRate
    
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Error")
    plt.title("Epoch Number vs. Training Error for different Learning Rates")
    plt.show()
    
    # Part D
    model = DeltaRulePerceptron("Stochastic")
    startTime = timeit.default_timer()
    model.Train(inputData, outputData, 50, minRate)
    stochasticTime = timeit.default_timer() - startTime
    stochasticUpdates = model.numWeightUpdates
    
    model = DeltaRulePerceptron("Batch")
    startTime = timeit.default_timer()
    model.Train(inputData, outputData, 50, minRate)
    batchTime = timeit.default_timer() - startTime
    batchUpdates = model.numWeightUpdates
    
    print("Epochs: 50\nLearning Rate: {}\n".format(minRate))
    print("Stochastic Training:")
    print("\tExecution Time: {}".format(stochasticTime))
    print("\tNum of Weight Updates: {}".format(stochasticUpdates))
    
    print("\nBatch Training:")
    print("\tExecution Time: {}".format(batchTime))
    print("\tNum of Weight Updates: {}".format(batchUpdates))
    
    
    # Problem 2
    # Part A
    startLearnRate = 0.001
    decayRate = 0.9
    
    print("\n\nProblem 2 Part A")
    print("\tStart Learning Rate: {}".format(startLearnRate))
    print("\tDecay Rate: {}".format(decayRate))
    
    model = DeltaRulePerceptron("Batch")
    model.Train(inputData, outputData, 25, startLearnRate, True, decayRate)
    model.PlotErrorOverEpochs("Decaying Learning Rate")
    
    model = DeltaRulePerceptron("Batch")
    model.Train(inputData, outputData, 25, startLearnRate, False)
    model.PlotErrorOverEpochs("Constant Learning Rate")
    
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Error")
    plt.title("Epoch Number vs. Training Error for 25 Epochs - Decaying Rates")
    plt.show()
    
    # Part B
    startLearnRate = 0.001
    t = 0.5
    d = 0.9
    D = 1.02
    
    print("\n\nProblem 2 Part B")
    print("\tStart Learning Rate: {}".format(startLearnRate))
    print("\tt: {}".format(t))
    print("\td: {}".format(d))
    print("\tD: {}".format(D))
    
    model = DeltaRulePerceptron("Batch")
    model.Train(inputData, outputData, 25, startLearnRate, adaptiveRate=True, t=t, d=d, D=D)
    model.PlotErrorOverEpochs("Adaptive Learning Rate")
    
    model = DeltaRulePerceptron("Batch")
    model.Train(inputData, outputData, 25, startLearnRate, adaptiveRate=False)
    model.PlotErrorOverEpochs("Constant Learning Rate")
    
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Error")
    plt.title("Epoch Number vs. Training Error for 25 Epochs - Adaptive Rates")
    plt.show()

if __name__ == "__main__":
    Main()