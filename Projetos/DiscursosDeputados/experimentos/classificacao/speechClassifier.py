#!/usr/bin/env python

from argparse import ArgumentParser

from statistics import stdev, mean
import numpy as np

from nltk.stem.snowball import SnowballStemmer

from sklearn import svm
from sklearn.metrics import classification_report, f1_score
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

# Arguments
parser = ArgumentParser(description='Classificador de opinião (contrária ou a favor) em discursos de deputado')

parser.add_argument('path', help='Caminho para ler os documentos de texto. Espera-se que os arquivos estejam organizados em diretórios que dizem respeito à sua classe.')
parser.add_argument('--stopwords', help='Caminho relativo para arquivo com as stopwords a serem removidas. Cada linha deve representar uma palavra diferente.')
parser.add_argument('--k_folds','-k', metavar='K', type=int, default=10, help='Número de folds utilizadas durante o treinamento.')
parser.add_argument('--save_misclassifications', action='store_true', help='Salva um arquivo contendo os nomes dos arquivos que foram classificados incorretamente em qualquer uma das K folds.')

args = parser.parse_args()

# Constants
stemmer = SnowballStemmer('portuguese', ignore_stopwords=True)
categories = ['Nao', 'Sim']
path = args.path
path_stop_words = args.stopwords
K = args.k_folds
pickle_file = 'model.pickle'

# Variables
n_fold = 1
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
f1 = []
acc = []
miss_list = []

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


# TODO: Verificar a qualidade da lista de stop words para esta aplicação
with open(path_stop_words, 'r') as f:
	stop_words = f.read().splitlines()


speeches = load_files(path, categories=categories, encoding='utf-8', random_state=42)
speeches.data = np.array(speeches.data)
speeches.target = np.array(speeches.target)


for train_idx, test_idx in kf.split(speeches.data, speeches.target):

	print('\nFold', n_fold)
	n_fold += 1

	X_train = np.array(speeches.data[train_idx])
	y_train = np.array(speeches.target[train_idx])

	X_test = np.array(speeches.data[test_idx])
	y_test = np.array(speeches.target[test_idx])

	count_vect = StemmedCountVectorizer(ngram_range=(1,4), stop_words=stop_words)
	X_train_counts = count_vect.fit_transform(X_train)
	X_test_counts = count_vect.transform(X_test)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)


	# Train instances count
	_, count = np.unique(y_train, return_counts=True)

	# Test instances count
	#_, count = np.unique(y_test, return_counts=True)

	weights = [1 - count[0] / len(X_train), 1 - count[1] / len(X_train)]

	sample_weight = np.array(len(y_train) * [weights[1]])
	sample_weight[np.where(np.array(y_train) == 0)[0]] = weights[0]

	clf = MultinomialNB().fit(X_train_tfidf, y_train, sample_weight=sample_weight)

	predicted = clf.predict(X_test_tfidf)

	if args.save_misclassifications:
		miss_list.extend(speeches.filenames[test_idx[predicted != y_test]])

	print(classification_report(y_test, predicted, target_names=speeches.target_names))

	acc.append(np.mean(predicted == y_test))
	print('Accuracy:', acc[-1])

	f1.append(f1_score(y_test, predicted, labels=speeches.target_names))
	print('F1 Score:', f1[-1])


print('\n\nAccuracy Mean:\t\t', mean(acc))
print('Accuracy Std. Dev.:\t', stdev(acc))

print('F1 Score Mean:\t\t', mean(f1))
print('F1 Score Std. Dev.:\t', stdev(f1))

if args.save_misclassifications:
	unique = set(miss_list)
	with open('misclassifications.txt', 'w') as f:
		for file in unique:
			f.write(str(file) + '\n')
