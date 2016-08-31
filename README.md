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
### SVM (SMO)
#### Please read this English paper: <[Improvements to Platt's SMO Algorithm for SVM Classifier Design](http://www.mitpressjournals.org/doi/abs/10.1162/089976601300014493#.V8a65_l95aQ)> or this Chinese book: 《[统计学习方法](https://book.douban.com/subject/10590856/)》, and realize meanings of SVM and SMO first.
  - main interfaces in *findSupportVector.py*
    - load test data of *testSet.txt*: **loadDataSet(filename)**
    - find random j (!= i): **selectJrand(i, m)**
    - simple SMO algorithm (not Platt SMO): **smoSimple(dataMatIn, classLabels, C, toler, maxIter)**
  - example 1: find support vectors in *testSet.txt*:
```
python findSupportVector.py
```
  - main interfaces in *findPlattSupportVector.py*
    - load test data of *testSet.txt*: **loadDataSet(filename)**
    - find random j (!= i): **selectJrand(i,oS,Ei)**
    - Platt SMO algorithm without kernel function: **smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0))**
    - compute weights from alphas **calcWs(alphas, dataArr, classLabels)**
  - example 2: find support vectors in *testSet.txt*:
```
python findPlattSupportVector.py
```
  - main interfaces in *smoKernelPlattClassifier.py*
    - load test data of *testSet.txt*: **loadDataSet(filename)**
    - find random j (!= i): **selectJrand(i,m)**
    - Platt SMO algorithm with RBF kernel function: **smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0))**
    - compute the kernel or transform data to a higher dimensional: **spacekernelTrans(X, A, kTup)**
    - compute weights from alphas **calcWs(alphas, dataArr, classLabels)**
  - example 3: find support vectors in *testSet.txt*:
```
python smoKernelPlattClassifier.py
```
  - example 4: hand writing recognition. Training set (digital) and test set (digital) are in file folders: *trainingDigits* and *testDigits*:
```
python smo-writingClassifier.py
```
