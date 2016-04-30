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


def file2matrix(filename):
	'''
	read file into matrix
	'''
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector


def autoNorm(dataSet):
	'''
	Normalization
	'''
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet / tile(ranges, (m, 1))
	return normDataSet, ranges, minVals


def datingClassTest():
	'''
	hoRatio = the ratio of test set
	'''
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 5)
		print "the classifier came back with %d, the real answer is: %d" % (classifierResult, datingLabels[i])
		if(classifierResult != datingLabels[i]):
			errorCount += 1.0
	print "the total error rate is: %f" % (errorCount / float(numTestVecs))


def datingPlotTest():
	datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
	plt.show()


if __name__ == "__main__":
	# datingPlotTest()
	datingClassTest()