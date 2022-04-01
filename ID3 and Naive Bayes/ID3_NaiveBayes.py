import numpy as np # Linear Algebra
import pandas as pd # Loading and manipulating csv data
import matplotlib.pyplot as plt # Plotting results
from sklearn.model_selection import train_test_split # Use to split training and testing data

SAVE_FIGS = False
XCOLNAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
BETA = 1

class TreeNode:
    def __init__(self, x, y, bins):
        self.x = x
        self.y = y
        self.col = None  # Name of column
        self.bins = bins
        self.entropy = CalcEntropy(None, y, None, None)
        self.children = []

def LinePlotAllAccuracies(x, title):
    for i in range(len(x)):
        plt.plot(range(1, 6), x[i], label='Bin Size: {}'.format((i + 1) * 5))
    plt.legend()
    
    plt.xticks(np.arange(1, 6, 1.0))
    plt.xlabel("Run Number")
    plt.ylabel("Accuracy")
    plt.title(title)

    plt.show()

def PlotAllFScores(allPosNegList, title):
    allFScores = []
    for i in range(len(allPosNegList)):
        fScore = []
        for run in range(5):
            precision = allPosNegList[i][run][0] / (allPosNegList[i][run][0] + allPosNegList[i][run][2])
            recall = allPosNegList[i][run][0] / (allPosNegList[i][run][0] + allPosNegList[i][run][3])
            try:
                fScore.append(((1 + pow(BETA, 2)) * precision * recall) / (pow(BETA, 2) * precision + recall))
            except ZeroDivisionError:
                fScore.append(0)
        allFScores.append(fScore)
    for i in range(len(allPosNegList)):
        plt.plot(range(1, 6), allFScores[i], label='Bin Size: {}'.format((i + 1) * 5))
    plt.legend()
    
    plt.xticks(np.arange(1, 6, 1.0))
    plt.xlabel("Run Number")
    plt.ylabel("F Scores")
    plt.title(title)

    plt.show()

def PlotAllRocScores(allPosNegList, title):
    allFalsePos = []
    allTruePos = []
    for i in range(len(allPosNegList)):
        falsePos = []
        truePos = []
        for col in range(5):
            falsePos.append(allPosNegList[i][col][2] / (allPosNegList[i][col][1] + allPosNegList[i][col][2]))
            truePos.append(allPosNegList[i][col][0] / (allPosNegList[i][col][0] + allPosNegList[i][col][3]))
        allFalsePos.append(falsePos)
        allTruePos.append(truePos)
    for i in range(len(allPosNegList)):
        plt.plot(allFalsePos[i], allTruePos[i], label='Bin Size: {}'.format((i + 1) * 5))
    plt.legend()

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)

    plt.show()

def CalcEntropy(x, y, bins, prob):
    numPos = 0
    numNeg = 0
    numTotal = y.size
    if numTotal == 0:
        return 1
    if bins is None:
        for i in range(numTotal):
            if y[i] == 1:
                numPos += 1
            else:
                numNeg += 1
        if numPos == 0 or numNeg == 0:
            entropy = 0
        else:
            posFrac = numPos / numTotal
            negFrac = numNeg / numTotal
            entropy = -1 * posFrac * np.log2(posFrac) - negFrac * np.log2(negFrac)
        return entropy
    else:
        entropyList = []
        for i in range(len(bins) - 1):
            if i < len(bins) - 2:
                yBin = y[np.where((x >= bins[i]) & (x < bins[i + 1]))]
            else:
                yBin = y[np.where((x >= bins[i]) & (x <= bins[i + 1]))]
            entropyList.append(CalcEntropy(None, yBin, None, None))
        entropyList = np.array(entropyList)
        return entropyList.dot(prob) / numTotal

def ID3Helper(x, y, colNames, binSize):
    minEnt = 1
    for ind in range(len(colNames)):
        prob, bins = np.histogram(x[:,ind], binSize)
        node = TreeNode(x.copy(), y.copy(), bins)
        entropy = CalcEntropy(x[:,ind], y.flatten(), bins, prob)
        if minEnt > entropy:
            minEnt = entropy
            root = node
            root.col = colNames[ind]
            if minEnt == 0:
                break;
    return root

def ID3(x, y, colNames, binSize, root=None):
    if root is None:
        root = ID3Helper(x, y, colNames, binSize)
        colNames.remove(root.col)
        return ID3(x, y, colNames, binSize, root)
    else:
        prob, bins = np.histogram(x[:,XCOLNAMES.index(root.col)], binSize)
        # Split the set into bins
        for binInd in range(binSize):
            if binInd < binSize - 1:
                indices = np.where((x[:,XCOLNAMES.index(root.col)] >= bins[binInd]) & (x[:,XCOLNAMES.index(root.col)] < bins[binInd + 1]))
            else:
                indices = np.where((x[:,XCOLNAMES.index(root.col)] >= bins[binInd]) & (x[:,XCOLNAMES.index(root.col)] <= bins[binInd + 1]))
            xBin = x[indices]
            yBin = y[indices]
            if yBin.size < 1:
                root.children.append(2)  # Just classify as negative just in case a test case hits this
                continue
            if len(colNames) <= 1:
                # Determine whether the node is 1 or 2
                numPos = 0
                numNeg = 0
                for i in range(y.size):
                    if y[i] == 1:
                        numPos += 1
                    else:
                        numNeg += 1
                if numPos > numNeg:
                    root.children.append(1)
                else:
                    root.children.append(2)
                continue
            node = ID3Helper(xBin, yBin, colNames, binSize)
            nextColNames = colNames.copy()
            nextColNames.remove(node.col)
            if node.entropy == 0:
                root.children.append(yBin[0])
            else:
                root.children.append(node)
                return ID3(xBin, yBin, nextColNames, binSize, node)
        return root

def TestID3Accuracy(root, x, y):
    posNegList = [0, 0, 0, 0]  # TP, TN, FP, FN
    for i in range(y.size):
        node = root
        while isinstance(node, TreeNode):
            for ind in range(len(node.bins) - 2):
                if x[i][XCOLNAMES.index(node.col)] >= node.bins[ind] and x[i][XCOLNAMES.index(node.col)] < node.bins[ind + 1]:
                    break
            node = node.children[ind]
        if node == y[i]:
            if y[i] == 1:
                posNegList[0] += 1
            else:
                posNegList[1] += 1
        else:
            if y[i] == 1:
                posNegList[2] += 1
            else:
                posNegList[3] += 1
    return posNegList

def GetID3Values(root, x):
    y = []
    for i in range(x.shape[0]):
        node = root
        while isinstance(node, TreeNode):
            for ind in range(len(node.bins) - 2):
                if x[i][XCOLNAMES.index(node.col)] >= node.bins[ind] and x[i][XCOLNAMES.index(node.col)] < node.bins[ind + 1]:
                    break
            node = node.children[ind]
        y.append(node)
    return y

def NaiveBayes(x, y, binSize):
    probList = []
    binsList = []
    for ind in range(len(XCOLNAMES)):
        probListRow = []
        prob, bins = np.histogram(x[:,ind], binSize)
        binsList.append(bins)
        for i in range(len(bins) - 1):
            if i < len(bins) - 2:
                yBin = y[np.where((x[:,ind] >= bins[i]) & (x[:,ind] < bins[i + 1]))]
            else:
                yBin = y[np.where((x[:,ind] >= bins[i]) & (x[:,ind] <= bins[i + 1]))]

            numPos = 0
            numTotal = yBin.size
            if numTotal < 1:
                probListRow.append(0)
                continue
            for i in range(numTotal):
                if yBin[i] == 1:
                    numPos += 1
            probListRow.append(numPos / numTotal)
        probList.append(probListRow)
    return probList, binsList

def TestNaiveBayesAccuracy(probTable, x, y, binsTable):
    posNegList = [0, 0, 0, 0]  # TP, TN, FP, FN
    for i in range(y.size):
        probSetosa = 1
        probNotSetosa = 1
        for col in range(len(XCOLNAMES)):
            for ind in range(len(binsTable[col]) - 2):
                if x[i][col] >= binsTable[col][ind] and x[i][col] < binsTable[col][ind + 1]:
                    break
            probSetosa *= probTable[col][ind]
            probNotSetosa *= 1 - probTable[col][ind]
        if probSetosa > probNotSetosa and y[i] == 1 or probSetosa <= probNotSetosa and y[i] == 2:
            if y[i] == 1:
                posNegList[0] += 1
            else:
                posNegList[1] += 1
        else:
            if y[i] == 1:
                posNegList[2] += 1
            else:
                posNegList[3] += 1
    return posNegList

def GetNaiveBayesValues(probTable, x, binsTable):
    y = []
    for i in range(x.shape[0]):
        probSetosa = 0
        probNotSetosa = 0
        for col in range(len(XCOLNAMES)):
            for ind in range(len(binsTable[col]) - 2):
                if x[i][col] >= binsTable[col][ind] and x[i][col] < binsTable[col][ind + 1]:
                    break
            probSetosa += probTable[col][ind]
            probNotSetosa += 1 - probTable[col][ind]
        if probSetosa > probNotSetosa:
            y.append(1)
        else:
            y.append(2)
    return y

def Main():
    # Load the input file
    inputFile = pd.read_csv(r"iris.csv")

    # Dictionaries to convert data to numerical (only differentiating between setosa and non-setosa)
    iris = {"setosa": 1, "versicolor": 2, "virginica": 2}

    inputFile.species = [iris[item] for item in inputFile.species]

    # Grab the data from the input
    inputData = pd.DataFrame(inputFile, columns=XCOLNAMES)
    outputData = pd.DataFrame(inputFile, columns=["species"])
    
    xTrains = []
    xTests = []
    yTrains = []
    yTests = []
    
    for i in range(5):
         # Split and save the data (test size is 33%)
        xTrain, xTest, yTrain, yTest = train_test_split(inputData, outputData, test_size=0.33)
        xTrains.append(xTrain.to_numpy())
        xTests.append(xTest.to_numpy())
        yTrains.append(yTrain.to_numpy())
        yTests.append(yTest.to_numpy())

    print('ID3 Algorithm:')
    allAccuracies = []
    allPosNegList = []
    rootList = []
    yTruthsID3 = []

    for binSize in range(5, 21, 5):
        posNegRow = []
        accuracyList = []
        yTruthRow = []
        rootRow = []
        for i in range(5):
            root = ID3(xTrains[i].copy(), yTrains[i].copy().flatten(), XCOLNAMES.copy(), binSize)
            rootRow.append(root)
            yTruthRow.append(GetID3Values(root, xTests[i]))
            posNegList = TestID3Accuracy(root, xTests[i], yTests[i].flatten())
            accuracyList.append((posNegList[0] + posNegList[1]) / (posNegList[0] + posNegList[1] + posNegList[2] + posNegList[3]))
            posNegRow.append(posNegList)

        accuracyList = np.array(accuracyList)
        print('Bins: {}'.format(binSize))
        print('\tAccuracies: {}'.format(accuracyList))
        print('\tMin Acc: {}'.format(np.amin(accuracyList)))
        print('\tMax Acc: {}'.format(np.amax(accuracyList)))
        print('\tAvg Acc: {}'.format(np.average(accuracyList)))

        allAccuracies.append(accuracyList)
        allPosNegList.append(posNegRow)
        rootList.append(rootRow)
        yTruthsID3.append(yTruthRow)

    LinePlotAllAccuracies(allAccuracies, 'ID3 Algorithm Accuracies')
    PlotAllFScores(allPosNegList, 'ID3 Algorithm F Scores')
    PlotAllRocScores(allPosNegList, 'ID3 Algorithm ROC Curves')

    print('Naive Bayes Algorithm:')
    allAccuracies = []
    allPosNegList = []
    probTableList = []
    binsTableList = []
    yTruthsBayes = []

    for binSize in range(5, 21, 5):
        accuracyList = []
        posNegRow = []
        probTableRow = []
        binsTableRow = []
        yTruthRow = []

        for i in range(5):
            probTable, binsTable = NaiveBayes(xTrains[i].copy(), yTrains[i].copy().flatten(), binSize)
            probTableRow.append(probTable)
            binsTableRow.append(binsTable)
            yTruthRow.append(GetNaiveBayesValues(probTable, xTests[i], binsTable))
            posNegList = TestNaiveBayesAccuracy(probTable, xTests[i], yTests[i].flatten(), binsTable)
            accuracyList.append((posNegList[0] + posNegList[1]) / (posNegList[0] + posNegList[1] + posNegList[2] + posNegList[3]))
            posNegRow.append(posNegList)

        accuracyList = np.array(accuracyList)
        print('Bins: {}'.format(binSize))
        print('\tAccuracies: {}'.format(accuracyList))
        print('\tMin Acc: {}'.format(np.amin(accuracyList)))
        print('\tMax Acc: {}'.format(np.amax(accuracyList)))
        print('\tAvg Acc: {}'.format(np.average(accuracyList)))

        allAccuracies.append(accuracyList)
        allPosNegList.append(posNegRow)
        probTableList.append(probTableRow)
        binsTableList.append(binsTableRow)
        yTruthsBayes.append(yTruthRow)

    LinePlotAllAccuracies(allAccuracies, 'Naive Bayes Algorithm Accuracies')
    PlotAllFScores(allPosNegList, 'Naive Bayes Algorithm F Scores')
    PlotAllRocScores(allPosNegList, 'Naive Bayes Algorithm ROC Curves')

    allPosNegListID3Truth = []
    allPosNegListBayesTruth = []
    for binSize in range(5, 21, 5):
        posNegRowID3Truth = []
        posNegRowBayesTruth = []
        for i in range(5):
            ind = int(binSize / 5 - 1)
            xTestBayes = xTests[i]
            yTruthID3 = np.array(yTruthsID3[ind][i])
            posNegRowID3Truth.append(TestNaiveBayesAccuracy(probTableList[ind][i], xTestBayes, yTruthID3, binsTableList[ind][i]))

            xTestID3 = xTests[i]
            yTruthBayes = np.array(yTruthsBayes[ind][i])
            posNegRowBayesTruth.append(TestID3Accuracy(rootList[ind][i], xTestID3, yTruthBayes))

        allPosNegListID3Truth.append(posNegRowID3Truth)
        allPosNegListBayesTruth.append(posNegRowBayesTruth)

    PlotAllFScores(allPosNegListID3Truth, 'Naive Bayes Algorithm, ID3 Truth F Scores')
    PlotAllFScores(allPosNegListBayesTruth, 'ID3 Algorithm, Naive Bayes Truth F Scores')


if __name__ == "__main__":
    Main()