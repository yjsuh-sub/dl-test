# Ligistic Linear Regression

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pylab as plt

X0, y = make_classification(n_features=1, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state=4)
model = LogisticRegression().fit(X0, y)

xx = np.linspace(-3, 3, 100)
sigm = 1.0/(1 + np.exp(-model.coef_[0][0]*xx - model.intercept_[0]))

plt.subplot(211)
plt.plot(xx, sigm)
plt.scatter(X0, y, marker='o', c=y, s=100)
plt.scatter(X0[0], model.predict(X0[:1]), marker='o', s=300, c='r', lw=5, alpha=0.5)
plt.plot(xx, model.predict(xx[:, np.newaxis]) > 0.5, lw=2)
plt.scatter(X0[0], model.predict_proba(X0[:1])[0][1], marker='x', s=300, c='r', lw=5, alpha=0.5)
plt.axvline(X0[0], c='r', lw=2, alpha=0.5)
plt.xlim(-3, 3)
plt.subplot(212)
plt.bar(model.classes_, model.predict_proba(X0[:1])[0], align="center")
plt.xlim(-1, 2)
plt.gca().xaxis.grid(False)
plt.xticks(model.classes_)
plt.title("conditional probability")
plt.tight_layout()
plt.show()

print(X0.shape, y.shape)
print(model.predict_proba(X0[:1]))