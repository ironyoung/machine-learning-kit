# machine-learning-kit
Please install some Python libraries (*numpy*, *scipy*, *matplotlib*) at first.
### KNN
  - core function: **classify0 ()** in *kNN.py*
  - example 1: classify dating person. Training set is *datingTestSet.txt*
```
python kNN-datingClassifier.py
```
  - example 2: hand writing recognition. Training set (digital) and test set (digital) are in file folders: *trainingDigits* and *testDigits*
```
python kNN-writingClassifier.py
```
### Decision Tree
  - the structure of decision tree is composed of dicts:
  ```
  {A:{B:{...}}}
  ```
  - interfaces in *tree.py*
    - create decision tree: **createTree (dataSet, labels)**
    - save decision tree into file: **storeTree (inputTree, filename)**
    - read decision tree from file: **grabTree (filename)**
    - classifier: **classify (inputTree, featLabels, testVec)**
  - interfaces in *treePlot.py*
    - plot tree: **createPlot (inputTree)**
  - example 1: classify glasses. Training set is *lenses.txt*
```
python tree-glassClassifier.py
```
### Naive Bayes
  - interfaces in *bayes.py*
    - create vocabulary list, without repeat words: **createVocabList (dataSet)**
    - set-of-words (exist or not, 0 or 1): **setOfWords2Vec(vocabList, inputSet)**
    - bag-of-words (occurrences of words): **bagOfWords2Vec(vocabList, inputSet)**
    - train: **trainNB0(trainMatrix, trainClass)**
    - classifier: **classifyNB(vec2Classify, p0Vec, p1Vec, pClass1)**
    - test: **testingNB()**
  - example 1: classify spams. Data set is in file folder: *email* (randomly split data set into training set and test set)
```
python bayes_spam.py
```
### Logistic Regression
  - interfaces in *logReg.py*
    - load test data of *testSet.txt*: **loadDataSet()**
    - batch gradient ascend method: **gradAscent(dataMatIn, classLabels, maxCycles=500)**
    - stochastic gradient ascend method: **stocGradAscent0(dataMatrix, classLabels)**, **stocGradAscent1(dataMatrix, classLabels, numIter=150)**
  - example 1: horse colic classifier. Training data is *horseColicTraining.txt* and test data is *horseColicTest.txt*
```
python horseColic_logReg.py
```
