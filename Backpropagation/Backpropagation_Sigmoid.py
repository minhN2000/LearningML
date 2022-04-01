import math
import numpy as np # Linear Algebra
import pandas as pd # Loading and manipulating csv data
from sklearn.model_selection import train_test_split # Use to split training and testing data

# Concepts from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/ are used for this assignment

class Node: # Or Neuron
    def __init__(self, weights):
        self.weights = weights # first weight (index 0) is bias
        self.result = None
        self.deltaW = None

class NeuralNetwork:
    def __init__(self, nInputs, nHidden, nOutputs, learnRate, squashing):
        self.nInputs = nInputs
        self.nHidden = nHidden
        self.nOutputs = nOutputs
        self.learnRate = learnRate
        self.squashing = squashing
        
        hiddenLayer = [Node(np.random.rand(nInputs+1)) for i in range(nHidden)]
        outputLayer = [Node(np.random.rand(nHidden+1)) for i in range(nOutputs)]
        self.layers = [hiddenLayer, outputLayer]
    
    def Predict(self, inputData):
        predictedData = []
        
        for dataPoint in inputData:
            result = self.ForwardPropagate(dataPoint) # probabilities of each class
            predictedData.append(result.index(max(result))) # Index of max-probability value is classification
        
        return predictedData
    
    def Train(self, inputData, outputData, nEpochs):
        numDataPoints = len(inputData)
        sumSquaredError = 0
        for epoch in range(nEpochs):
            error = 0
            
            # Use a stochastic gradient descent to tune weights for each attribute
            for iData in range(numDataPoints):
                probs = self.ForwardPropagate(inputData[iData])
                expected = np.zeros(self.nOutputs)
                expected[outputData[iData]] = 1 # [1, 0] or [0, 1], to match with probabilities from ForwardPropagate result
                
                error += sum([(expected[i]-probs[i])**2 for i in range(len(expected))]) # Sum of squared estimate of errors
                
                self.BackPropagate(expected)
                
                self.UpdateWeights(inputData[iData])
            
            sumSquaredError = error
        return sumSquaredError
    
    def ForwardPropagate(self, x):
        # Returns result of two nodes in output layer
        input = x
        for layer in self.layers:
            nextInput = []
            
            for neuron in layer:
                act = self.Activation(input, neuron.weights)
                neuron.result = self.Squash(act)
                nextInput.append(neuron.result)
                
            input = nextInput
            
        return input
    
    def BackPropagate(self, expected):
        numLayers = len(self.layers)
        for iLayer in reversed(range(numLayers)): # Start with output back to hidden
            layer = self.layers[iLayer]
            errors = []
            
            if (iLayer == (numLayers - 1)): # Last layer (output)
                for iNeuron in range(len(layer)):
                    neuron = layer[iNeuron]
                    errors.append(neuron.result - expected[iNeuron])
            else: # Hidden layer(s)
                for iNeuron in range(len(layer)):
                    error = 0.0
                    
                    for neuron in self.layers[iLayer+1]:
                        error += (neuron.weights[iNeuron] * neuron.deltaW)
                    
                    errors.append(error)
            
            # Calculate change in weight for each neuron in layer, update only at end
            for iNeuron in range(len(layer)):
                neuron = layer[iNeuron]
                neuron.deltaW = errors[iNeuron] * self.SquashDerivative(neuron.result)
    
    def UpdateWeights(self, x):
        # Use a stochastic gradient descent to tune weights for each attribute
        for iLayer in range(len(self.layers)):
            input = x
            
            if iLayer > 0:
                input = [neuron.result for neuron in self.layers[iLayer-1]]
            
            for neuron in self.layers[iLayer]:
                neuron.weights[0] -= self.learnRate * neuron.deltaW # Update bias separately
                
                for iAttrib in range(len(input)):
                    neuron.weights[iAttrib+1] -= self.learnRate * neuron.deltaW * input[iAttrib]
        
    
    def Activation(self, x, weights):
        # Simple activation function, weighted sum
        act = weights[0] # Bias
        for i in range(len(x)):
            act += weights[i+1] * x[i]
        
        return act
    
    def Squash(self, x):
        if (self.squashing == "Sigmoid"):
            return 1.0 / (1.0 + math.exp(-x))
        elif (self.squashing == "TanH"):
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    
    def SquashDerivative(self, x):
        if (self.squashing == "Sigmoid"):
            return x * (1 - x)
        elif (self.squashing == "TanH"):
            return 1 - x**2

def Accuracy(actual, predicted):
    # numCorrect / numTotal
    numItems = len(actual)
    sum = 0
    for i in range(numItems):
        if (actual[i] == predicted[i]):
            sum += 1
    return sum / numItems

def Main():
    # Load the input file
    inputFile = pd.read_csv(r"data_banknote_authentication.txt", header=None)
    
    inputData = pd.DataFrame(inputFile, columns=[0, 1, 2, 3]).to_numpy()
    outputData = pd.DataFrame(inputFile, columns=[4]).to_numpy()
    
    # Train = 70%,  Validation = 20%,  Test = 10%
    trainInput, intermedInput, trainOutput, intermedOutput = train_test_split(inputData, outputData, test_size=0.3)
    validInput, testInput, validOutput, testOutput = train_test_split(intermedInput, intermedOutput, test_size=0.333)
    
    nAttribs = len(inputData[0])
    nOutputs = 2
    nEpochs = 100
    learnRate = 0.1
    
    for squashing in ["Sigmoid", "TanH"]:
        bestNN = None
        bestAcc = 0
        bestNumHidden = 0
        
        print("Squashing Function: {}".format(squashing))
        for nHidden in reversed(range(1, nAttribs+1)):
            Network = NeuralNetwork(nAttribs, nHidden, nOutputs, learnRate, squashing)
            sumSquaredError = Network.Train(trainInput, trainOutput, nEpochs)
            accuracy = Accuracy(validOutput, Network.Predict(validInput))
            print("\tNumHidden={}, SSE={:.3f}, ValidAcc={:.3f}".format(nHidden, sumSquaredError, accuracy))
            
            if (accuracy >= bestAcc):
                bestNN = Network
                bestAcc = accuracy
                bestNumHidden = nHidden
        
        testAcc = Accuracy(testOutput, bestNN.Predict(testInput))
        print("\nBest model for Squashing Function: {}\n\tNumber Hidden Nodes = {}\n\tValidation Accuracy = {}\n\tTest Accuracy = {}\n\n".format(squashing, bestNumHidden, bestAcc, testAcc))

if __name__ == "__main__":
    Main()