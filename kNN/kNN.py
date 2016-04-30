from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inVec, trainingDataSet, trainingLabels, k):
	'''
	the input is a vector
	'''
	dataSetSize = trainingDataSet.shape[0]
	diffMat = tile(inVec, (dataSetSize, 1)) - trainingDataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteLabel = trainingLabels[sortedDistIndicies[i]]
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]


def main():
	group, labels = createDataSet()
	print classify0([0, 0], group, labels, 3)


if __name__ == "__main__":
	main()