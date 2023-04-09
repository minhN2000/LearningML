"""
GCNN.py

This module contains the implementation of a Graph Convolutional Neural Network (GCN) in Python, using the Keras
library for the Glorot Uniform initializer.

Classes:
- GCNLayer
- GradientDescentOptim
- SoftmaxLayer
- GCNN

"""

import numpy as np
from keras.initializers import glorot_uniform

class GCNLayer:
    """
    A Graph Convolutional Layer of a GCN.
    Input:
        nInputs (int): The number of input nodes.
        nOutputs (int): The number of output nodes.
        activation (callable): Activation function for the layer.
        name (str): The name of the layer.
        W (numpy array): The weight matrix of the layer.

    """
    def __init__(self, nInputs, nOutputs, activation=None, name=''):
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.W = glorot_uniform(self.nOutputs, self.nInputs)
        self.activation = activation
        self.name = name

    def forward(self, A, X, W=None):
        """s
        Perform the forward pass of the layer.

        Args:
            A (numpy array): The adjacency matrix.
            X (numpy array): The input feature matrix.
            W (numpy array): The weight matrix, if None use self.W.

        Returns:
            H (numpy array): The output feature matrix.

        """
        self._X = (A @ X).T  # message passing

        if W is None:
            W = self.W

        H = W @ self._X  # net input
        if self.activation != None:
            H = self.activation(H)
        self._H = H
        return self._H.T

    def backward(self, optim, update=True):
        """
        Perform the backward pass of the layer.

        Args:
            optim (GradientDescentOptim): The gradient descent optimizer.
            update (bool): Whether to update the weights during backpropagation.

        Returns:
            dW + dwDecay (numpy array): The weight update.

        """
        dtanh = 1 - np.asarray(self._H.T)**2
        d2 = np.multiply(optim.out, dtanh)

        optim.out = d2 @ self.W

        dW = np.asanyarray(d2.T @ self._X.T) / optim.bs
        dwDecay = self.W * optim.wd / optim.bs

        if update:
            self.W -= optim.lr * (dW + dwDecay)
            return dW + dwDecay
        
class GradientDescentOptim:
    """
    A gradient descent optimizer with weight decay.
    Args:
        lr (float): The learning rate.
        wDecay (float): The weight decay factor.
        _yHat (numpy array): The predicted output.
        _y (numpy array): The true output.
        _out (numpy array): The output of the optimization step.
        bs (int): The batch size.
        trainNodes (numpy array): The training nodes.

    """
    def __init__(self, lr, wDecay):
        self.lr = lr
        self.wDecay = wDecay
        self._yHat = None
        self._y = None
        self._out = None
        self.bs = None
        self.trainNodes = None

    def __call__(self, yHat, y, trainNodes=None):
        """
        Call the optimizer with the predicted and true outputs.

        Args:
            yHat (numpy array): The predicted output.
            y (numpy array): The true output.
            trainNodes (numpy array): The training nodes, if None use all nodes.
        """
        self.y = y
        self.yHat = yHat

        if trainNodes != None:
            self.trainNodes = trainNodes
        else:
            self.trainNodes = np.arange(yHat.shape[0])

        self.bs = self.trainNodes.shape[0]

    @property
    def out(self):
        """
        Get the output of the optimization step.

        Returns:
            _out (numpy array): The output of the optimization step.

        """
        return self._out
    
    @out.setter
    def out(self, y):
        """
        Set the output of the optimization step.

        Args:
            y (numpy array): The output of the optimization step.

        """
        self._out = y

class SoftmaxLayer:
    """
    A Softmax Layer.
    Inputs:
        nInputs (int): The number of input nodes.
        nOutputs (int): The number of output nodes.
        name (str): The name of the layer.
        W (numpy array): The weight matrix of the layer.
        b (numpy array): The bias vector of the layer.

    """

    def __init__(self, nInputs, nOutputs, name=''):
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.W = glorot_uniform(self.nOutputs, self.nInputs)
        self.b = np.zeros((self.nOutputs, 1))
        self.name = name
        self._X = None

    def shift(self, proj):
        """
        Shift the softmax inputs for numerical stability.

        Args:
            proj (numpy array): The input feature matrix.

        Returns:
            shifted (numpy array): The shifted and normalized input feature matrix.

        """
        shiftX = proj - np.max(proj, axis=0, keepdims=True)
        exps = np.exp(shiftX)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self, X, W=None, b=None):
        """
        Perform the forward pass of the layer.

        Args:
            X (numpy array): The input feature matrix.
            W (numpy array): The weight matrix, if None use self.W.
            b (numpy array): The bias vector, if None use self.b.

        Returns:
            softmax (numpy array): The softmax output.

        """
        self._X = X.T
        if W is None:
            W = self.W
        if b is None:
            b = self.b

        proj = np.asarray(W @ self._X) + b
        return self.shift(proj).T

    def backward(self, optim, update=True):
        """
        Perform the backward pass of the layer.

        Args:
            optim (GradientDescentOptim): The gradient descent optimizer.
            update (bool): Whether to update the weights during backpropagation.

        Returns:
            dW + dWDecay (numpy array): The weight update.
            db (numpy array): The bias update.

        """
        trainMask = np.zeros(optim.yHat.shape[0])
        trainMask[optim.trainNodes] = 1
        trainMask = trainMask.reshape((-1, 1))

        d1 = np.asarray((optim.yHat - optim.y))
        d1 = np.multiply(d1, trainMask)

        optim.out = d1 @ self.W
        dW = (d1.T @ self._X.T) / optim.bs
        db = d1.T.sum(axis=1, keepdims=True) / optim.bs

        dWDecay = self.W * optim.wDecay / optim.bs
        if update:
            self.W -= optim.lr * (dW + dWDecay)
            self.b -= optim.lr * db.reshape(self.b.shape)

        return dW + dWDecay, db.reshape(self.b.shape)

class GCNN:
    """
    A Graph Convolutional Neural Network (GCN) implementation.
    Inputs:
        nInputs (int): The number of input nodes.
        nOutputs (int): The number of output nodes.
        nLayers (int): The number of hidden layers.
        hiddenSizes (list): A list containing the number of nodes in each hidden layer.
        activation (callable): Activation function for the layers.
        layers (list): A list containing the layers of the GCN.

    """

    def __init__(self, nInputs, noutputs, nLayers, hiddenSizes, activation, seed=0):
        self.nInputs = nInputs
        self.nOutputs = noutputs
        self.nLayers = nLayers
        self.hiddenSizes = hiddenSizes

        np.random.seed(seed)

        self.layers = []

        # Add input layer
        self.layers.append(GCNLayer(nInputs, hiddenSizes[0], activation, name='in'))

        # Add hidden layers
        for layer in range(nLayers):
            self.layers.append(GCNLayer(self.layers[-1].W.shape[0], hiddenSizes[layer], activation, name="hidden layer{}".format(layer+1)))

        # Add output layer
        self.layers.append(SoftmaxLayer(hiddenSizes[-1], noutputs, name='out'))

    def embedding(self, A, X):
        """
        Compute the node embeddings.

        Args:
            A (numpy array): The adjacency matrix.
            X (numpy array): The input feature matrix.

        Returns:
            H (numpy array): The embeddings.

        """
        H = X
        for layer in self.layers[:-1]:
            H = layer.forward(A, H)
        return np.asarray(H)

    def forward(self, A, X):
        """
        Perform the forward pass of the GCN.

        Args:
            A (numpy array): The adjacency matrix.
            X (numpy array): The input feature matrix.

        Returns:
            p (numpy array): The softmax probabilities.

        """
        H = self.embedding(A, X)

        p = self.layers[-1].forward(H)

        return np.asarray(p)

    def backward(self, optim, update=True):
        """
        Perform the backward pass of the GCN.

        Args:
            optim (GradientDescentOptim): The gradient descent optimizer.
            update (bool): Whether to update the weights during backpropagation.

        Returns:
            H (numpy array): The weight and bias updates.

        """
        self.layers[-1].backward(optim)
        for layer in reversed(self.layers[:-1]):
            H = layer.backward(optim)
        return H



