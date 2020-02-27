import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, cluster
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from spherecluster import SphericalKMeans

stemmer = SnowballStemmer('portuguese', ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

path = 'dados'
path_stop_words = 'stopwords_pt.txt'

categories = [
	'Nao',
	'Sim',
]


with open(path_stop_words, 'r') as f:
	stop_words = f.read().splitlines()

print('Loading data...')
dataset = load_files(os.path.join(path, 'train'), categories=categories, encoding='utf-8', random_state=42)
data_test = load_files(os.path.join(path, 'test'), categories=categories, encoding='utf-8', random_state=42)

X = dataset.data
labels_true = dataset.target

print('Preprocessing data...')
count_vect = StemmedCountVectorizer(ngram_range=(1,4), stop_words=stop_words, max_df=0.6, token_pattern='[^\d\W]{3,}')

X_counts = count_vect.fit_transform(X)
X_counts_test = count_vect.transform(data_test.data)


print(X_counts.shape)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_tfidf_test = tfidf_transformer.transform(X_counts_test)

kbest = SelectKBest(f_classif, k=35)

X_tfidf = kbest.fit_transform(X_tfidf, labels_true)
X_tfidf_test = kbest.transform(X_tfidf_test)


print('\nFitting KNN...\n')
clt = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
clt.fit(X_tfidf, labels_true)
train_labels = clt.predict(X_tfidf)


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(train_labels)) - (1 if -1 in train_labels else 0)

count_vec = list()
idx = [np.where(train_labels == j)[0] for j in range(n_clusters_)]

for i in range(n_clusters_):
	unique_values, count = np.unique(dataset.target[idx[i]], return_counts=True)
	count_vec.append(count)

for i in range(n_clusters_):
	print('Cluster %d: %d samples' % (i, np.where(train_labels == i)[0].shape[0]), count_vec[i])

unique_labels = set(train_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

print('\nNumber of clusters: %d\n' % n_clusters_)

print("Train data:")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(dataset.target, train_labels))
print("Completeness: %0.3f" % metrics.completeness_score(dataset.target, train_labels))
print("V-measure: %0.3f" % metrics.v_measure_score(dataset.target, train_labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(dataset.target, train_labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(dataset.target, train_labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_tfidf, dataset.target))
print("KNN Score: %0.4f" % clt.score(X_tfidf, dataset.target))

print('\nReducing data dimensionality...')
data_train = TSNE(n_components=2, random_state=42).fit_transform(X_tfidf.toarray())

print('Making predictions...\n')
labels = clt.predict(X_tfidf_test)

count_vec = list()
idx = [np.where(labels == j)[0] for j in range(n_clusters_)]

for i in range(n_clusters_):
	unique_values, count = np.unique(data_test.target[idx[i]], return_counts=True)
	count_vec.append(count)

for i in range(n_clusters_):
	print('Cluster %d: %d samples' % (i, np.where(labels == i)[0].shape[0]), count_vec[i])

print("\n\nTest data:")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(data_test.target, labels))
print("Completeness: %0.3f" % metrics.completeness_score(data_test.target, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(data_test.target, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(data_test.target, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(data_test.target, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_tfidf_test, data_test.target))
print("KNN Score: %0.4f" % clt.score(X_tfidf_test, data_test.target))

print('\nReducing data dimensionality...')
data = TSNE(n_components=2, random_state=42).fit_transform(X_tfidf_test.toarray())

legend_elements = [Line2D([0], [0], color='w', markerfacecolor=tuple(colors[0]), marker='s', label=f'Pred. {data_test.target_names[0]}', markersize=6),
					Line2D([0], [0], color='w', markerfacecolor=tuple(colors[1]), marker='s', label=f'Pred. {data_test.target_names[1]}', markersize=6),
					Line2D([0], [0], color='w', markerfacecolor=None, markeredgecolor='k', marker='o', label=f'Class {data_test.target_names[0]}', markersize=6),
					Line2D([0], [0], color='w', markerfacecolor=None, markeredgecolor='k', marker='v', label=f'Class {data_test.target_names[1]}', markersize=6)]

print('Plotting data...')

for i in range(len(train_labels)):
	if labels_true[i] == 0:
		plt.plot(data_train[i, 0], data_train[i, 1], 'o', markerfacecolor=tuple(colors[train_labels[i]]), markeredgecolor='k', markersize=8)
	else:
		plt.plot(data_train[i, 0], data_train[i, 1], 'v', markerfacecolor=tuple(colors[train_labels[i]]), markeredgecolor='k', markersize=8)

plt.legend(handles=legend_elements, loc="lower left")
plt.savefig('train.eps', format='eps')
plt.clf()


for i in range(len(labels)):
	if data_test.target[i] == 0:
		plt.plot(data[i, 0], data[i, 1], 'o', markerfacecolor=tuple(colors[labels[i]]), markeredgecolor='k', markersize=8)
	else:
		plt.plot(data[i, 0], data[i, 1], 'v', markerfacecolor=tuple(colors[labels[i]]), markeredgecolor='k', markersize=8)

plt.legend(handles=legend_elements, loc="lower left")
plt.savefig('pred.eps', format='eps')
plt.clf()
