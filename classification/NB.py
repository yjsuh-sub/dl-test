# Naive Bayesian model

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt

news = fetch_20newsgroups(subset="all")
model = Pipeline([
    ('vect', TfidfVectorizer(stop_words="english")),
    ('nb', MultinomialNB()),
])
model.fit(news.data, news.target)

x1 = news.data[:1]
x2 = news.data[1:2]
y1 = model.predict(x1)[0]
y2 = model.predict(x2)[0]

print(model.classes_)

plt.figure(figsize=(10, 15))
plt.subplot(411)
plt.title('data 1')
plt.bar(model.classes_, model.predict_proba(x1)[0], align="center")
plt.xlim(-1, 20)
plt.gca().xaxis.grid(False)
plt.xticks(model.classes_)
plt.subplot(412)
plt.bar(model.classes_, model.predict_log_proba(x1)[0], align="center")
plt.xlim(-1, 20)
plt.gca().xaxis.grid(False)
plt.xticks(model.classes_)
plt.subplot(413)
plt.title('data 2')
plt.bar(model.classes_, model.predict_proba(x2)[0], align="center")
plt.xlim(-1, 20)
plt.gca().xaxis.grid(False)
plt.xticks(model.classes_)
plt.subplot(414)
plt.bar(model.classes_, model.predict_log_proba(x2 )[0], align="center")
plt.xlim(-1, 20)
plt.gca().xaxis.grid(False)
plt.xticks(model.classes_)
plt.tight_layout(h_pad=3)
plt.show()