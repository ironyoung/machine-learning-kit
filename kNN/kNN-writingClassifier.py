from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


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


def img2vector(filename, height, width):
	'''
	the height and width of the image
	'''
	size = height * width
	retVec = zeros((1, 1024))
	fr = open(filename)
	for i in range(height):
		lineStr = fr.readline()
		for j in range(width):
			retVec[0, i*height + j] = int(lineStr[j])
	return retVec


def handwritingClassTest(height, width):
	'''
	the height and width of the image
	'''
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m, int(height * width)))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNameStr = int(fileStr.split('_')[0])
		hwLabels.append(classNameStr)
		trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr, height, width)

	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr  = int(fileStr.split('_')[0])
		vecUnderTest = img2vector('testDigits/%s' % fileNameStr, height, width)
		classifierResult = classify0(vecUnderTest, trainingMat, hwLabels, 5)
		print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
		if(classifierResult != classNumStr):
			errorCount += 1

	print "\nthe total number of errors is: %d" % errorCount
	print "\nthe total error rate is: %f" % (errorCount/float(mTest))	


if __name__ == "__main__":
	'''
	input the height and width of the image
	'''
	handwritingClassTest(32, 32)