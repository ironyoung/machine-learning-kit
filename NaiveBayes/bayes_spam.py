import re
import bayes
from numpy import *


# step: 1. split() 2. lower()
def textParse(text):
	listOfTokens = re.split(r'\W*', text)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
	docList 	= []
	classList 	= []
	fullText 	= []
	for i in range(1, 26):
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)

		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)

	vocabList 	= bayes.createVocabList(docList)
	trainingSet = range(50)
	testSet 	= []

	# randomly split data set into 2 sets: test set, and training set
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))	# random int 0~len
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])		# split

	trainMat = [];
	trainClasses = [];
	for docIndex in trainingSet:
		trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])

	p0V, p1V, pSpam = bayes.trainNB0(array(trainMat), array(trainClasses))
	errorCount = 0

	for docIndex in testSet:
		wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])
		if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
			print "error word: %s" % (docList[docIndex])
	print "error rate is: %f", float(errorCount) / len(testSet)


if __name__ == "__main__":
	spamTest()