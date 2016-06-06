from numpy import *

def loadDataSet():
	dataMat  = [];
	labelMat = [];
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0,  float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat


def sigmoid(inX):
	return 1.0/(1.0 + exp(-inX))


def gradAscent(dataMatIn, classLabels, maxCycles=500):
	dataMatrix 	= mat(dataMatIn)
	labelMat 	= mat(classLabels).transpose()
	m, n 	= shape(dataMatrix)
	alpha	= 0.001
	weights = ones((n, 1))
	for k in range(maxCycles):
		h 	= sigmoid(dataMatrix * weights)
		error 	= (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights


# without iteration
def stocGradAscent0(dataMatrix, classLabels):
	dataMatrix	= array(dataMatrix)
	m, n 	= shape(dataMatrix)
	alpha	= 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i] * weights))
		error 	= classLabels[i] - h
		weights	= weights + alpha * error * dataMatrix[i]
	return weights


# with iteration
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	dataMatrix	= array(dataMatrix)
	m, n 	= shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex	= range(m)
		for i in range(m):
			alpha = 4 / (1.0 + i + j) + 0.01	# modify learning rate "alpha" every iteration
			randIndex = int(random.uniform(0, len(dataIndex)))
			h 	= sigmoid(sum(dataMatrix[randIndex] * weights))
			error = classLabels[randIndex] - h
			weights	= weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights