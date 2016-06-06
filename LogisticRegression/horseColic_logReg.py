from logReg import *

def classifyVector(inX, weights):
	prob = sigmoid(sum(inX * weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0


def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest	= open('horseColicTest.txt')
	trainingSet		= []
	trainingLabel	= []

	for line in frTrain.readlines():
		currLine= line.strip().split('\t')
		lineArr	= []

		for i in range(21):
			lineArr.append(float(currLine[i]))

		trainingSet.append(lineArr)
		trainingLabel.append(float(currLine[21]))

	# trainWeights= gradAscent(array(trainingSet), trainingLabel, 500)
	trainWeights= stocGradAscent1(array(trainingSet), trainingLabel, 500)
	errorCount 	= 0
	numTestVec	= 0.0

	for line in frTest.readlines():
		numTestVec	+= 1.0
		currLine	= line.strip().split('\t')
		lineArr		= []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
			errorCount	+= 1

	errorRate	= float(errorCount)/numTestVec
	print "the error rate of this test is: %f" % errorRate
	return errorRate


def multiTest():
	numTests	= 10
	errorRateSum= 0.0
	for k in range(numTests):
		errorRateSum	+= colicTest()
	print "after %d iterations the average error rate is: %f" %(numTests, errorRateSum/float(numTests))


def main():
	multiTest()


if __name__ == '__main__':
	main()