# DecisionTreeClassifier

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
#from io import StringIO
#import pydot
#from IPython.core.display import Image
#from sklearn.tree import export_graphviz

iris = load_iris()
X = iris.data[:, [1, 2]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0).fit(X_train, y_train)

test_idx=range(105,150)
resolution=0.01
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = mpl.colors.ListedColormap(colors[:len(np.unique(y_combined))])

x1_min, x1_max = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
x2_min, x2_max = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
Z = tree.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], s=80, label=cl)
plt.show()

#def draw_decision_tree(classifier):
    #dot_buf = StringIO()
    #export_graphviz(classifier, out_file=dot_buf, feature_names=iris.feature_names)
    #graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    #image = graph.create_png()
    #return Image(image)

#draw_decision_tree(tree)
print(confusion_matrix(y, tree.predict(X)))