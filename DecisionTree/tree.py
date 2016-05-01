from math import log
import operator
import pickle


def createDataSet():
	dataSet = [[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels


def calcShannonEnt(dataSet):
	numDataSet = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1

	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numDataSet
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt


def splitDataSet(dataSet, axis, value):
	'''
	
	Split data set, and return the reduced dataSet:
	if dataSet[i][axis] == value
		then delete the feature: [axis]
	'''
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


def chooseBestFeatureSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1	# the last feature is label
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		featList = set(featList)		# duplicate removal
		newEntropy = 0.0
		for value in featList:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = float(len(subDataSet))/len(dataSet)
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		# print "infoGain %f" % infoGain
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature


def majorityCnt(classlist):
	classCount = {}
	for vote in classlist:
		classCount[vote] = classCount.get(vote, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	'''
	ID3-decision tree delete one feature per decision
	when the number of features is 0 but exists more than one class, return the major class 
	'''
	labelsTemp = labels[:]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureSplit(dataSet)
	bestFeatLabel = labelsTemp[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labelsTemp[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	featValues = set(featValues)
	for value in featValues:
		subLabels = labelsTemp[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

	return myTree


def classify(inputTree, featLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel


def storeTree(inputTree, filename):
	fw = open(filename, 'w')
	pickle.dump(inputTree, fw)
	fw.close()


def grabTree(filename):
	fr = open(filename, 'r')
	return pickle.load(fr)


def main():
	myData, labels = createDataSet()
	myTree = createTree(myData, labels)
	print myTree
	print "[0, 1] classification: %s" % classify(myTree, labels, [0, 1])


if __name__ == "__main__":
	main()