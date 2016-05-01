# machine-learning-kit
### KNN
  - core function: **classify0 ()** in *kNN.py*
  - the 1st example: classify dating person, and training set is in the **file**: *datingTestSet.txt*
```
python kNN-datingClassifier.py
```
  - the 2nd example: hand writing recognition, and training set (digital) and test set (digital) are in the **folders**
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
  - the 1st example: classify glasses, and training set is in the **file**: *lenses.txt*
```
python tree-glassClassifier.py
```
