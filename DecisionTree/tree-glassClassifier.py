import tree
import treePlotter


fr = open('lenses.txt')
lenses = [example.strip().split('\t') for example in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']	# features
lenseTree = tree.createTree(lenses, lensesLabels)
print lenseTree

treePlotter.createPlot(lenseTree)