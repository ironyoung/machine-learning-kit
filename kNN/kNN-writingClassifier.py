from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import kNN


def img2vector(filename, height, width):
	'''
	the height and width of the digital image
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
		'''
		the file name consists of 'class name'_'...'
		'''
		classNumStr  = int(fileStr.split('_')[0])
		vecUnderTest = img2vector('testDigits/%s' % fileNameStr, height, width)
		classifierResult = kNN.classify0(vecUnderTest, trainingMat, hwLabels, 5)
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