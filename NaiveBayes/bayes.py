# naive bayes: 1.independent features; 2.equally important features

from numpy import *

def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
				   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
   	classVec = [0, 1, 0, 1, 0, 1]
   	return postingList, classVec


def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)


# word2vec: set
def setOfWords2Vec(vocabList, inputSet):
	returnVec = len(vocabList) * [0]
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "the word: %s is not in the vocabulary" % (word)
	return returnVec


# bag-of-words model
def bagOfWords2Vec(vocabList, inputSet):
	returnVec = len(vocabList) * [0]
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else:
			print "the word: %s is not in the vocabulary" % (word)
	return returnVec


# improvements: 
# --- 1. p(w_i|c_j) = 0 => p(w|c_j) = p(w_1|c_j) * p(w_2|c_j) * p(w_3|c_j) * ... = 0
# --- 2. underflow with tiny probabilities
# ---
def trainNB0(trainMatrix, trainClass):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainClass) / float(numTrainDocs)
	# p0Num = zeros(numWords)		# return numpy.array, change to ones()
	# p1Num = zeros(numWords)		# return numpy.array, change to ones()
	# p0Denom = 0.0
	# p1Denom = 0.0
# improvement NO.1:
	p0Num = ones(numWords)			# return numpy.array
	p1Num = ones(numWords)			# return numpy.array
	p0Denom = 2.0
	p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainClass[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	# p1Vect = p1Num / p1Denom		$ change to log()
	# p0Vect = p0Num / p0Denom		$ change to log()
# improvement NO.2, avoid underflow => log():
	p1Vect = log(p1Num / p1Denom)
	p0Vect = log(p0Num / p0Denom)

	return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	#multiply = log sum
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)	# just for binary classification
	if p1 > p0:
		return 1
	else:
		return 0


def testingNB():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for doc in listOPosts:
		#trainMat.append(setOfWords2Vec(myVocabList, doc))
		trainMat.append(bagOfWords2Vec(myVocabList, doc))

	p0V, p1V, pAb = trainNB0(trainMat, listClasses)
	
	print "[p(wi|c0)] : %s" % p0V
	print "[p(wi|c1)] : %s" % p1V
	print "p(c1) = %f" % pAb

	testEntry = ['love', 'my', 'dalmation']
	#thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

	testEntry = ['stupid', 'garbage']
	#thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


if __name__ == "__main__":
	testingNB()