#!/usr/bin/env python

import numpy as np
import os

from argparse import ArgumentParser
from statistics import stdev, mean
from joblib import dump, load

from nltk.stem.snowball import SnowballStemmer

from sklearn import svm
from sklearn.metrics import classification_report, f1_score
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.model_selection import GridSearchCV

# Arguments
parser = ArgumentParser(description='Classificador de opinião (contrária ou a favor) em discursos de deputados')

parser.add_argument('path', help='Caminho para ler os documentos de texto. Espera-se que os arquivos estejam organizados em subdiretórios train e test.')
parser.add_argument('stopwords', help='Caminho relativo para arquivo com as stopwords a serem removidas. Cada linha deve representar uma palavra diferente.')
parser.add_argument('-c', '--classifier', choices=['svm', 'mnb', 'lr'], default='svm', help='Classificador a ser utilizado. Possiveis opções são: svm (Support Vector Machines), mnb (Multinomial Naive Bates) e lr (Logistic Regression)')
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
best_f1 = 0

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


with open(path_stop_words, 'r') as f:
	stop_words = f.read().splitlines()


speeches = load_files(os.path.join(path, 'train'), categories=categories, encoding='utf-8', random_state=42)

idx = np.argsort(speeches.filenames)


# Garante que toda run seja exatamente igual, partindo do pressuposto que existe a possibilidade de a lista de arquivos retornar de diferentes maneiras
speeches.data = np.array(speeches.data)[idx]
speeches.target = np.array(speeches.target)[idx]
speeches.filenames = np.array(speeches.filenames)[idx]

count_vect = StemmedCountVectorizer(ngram_range=(1,4), stop_words=stop_words, max_df=0.6, token_pattern='[^\d\W]{3,}').fit(speeches.data)
X = count_vect.transform(speeches.data)
tfidf_transformer = TfidfTransformer().fit(X)

for train_idx, test_idx in kf.split(speeches.data, speeches.target):

	print('\nFold', n_fold)
	n_fold += 1


	X_train = np.array(speeches.data[train_idx])
	y_train = np.array(speeches.target[train_idx])

	X_test = np.array(speeches.data[test_idx])
	y_test = np.array(speeches.target[test_idx])


	X_train_counts = count_vect.transform(X_train)
	X_test_counts = count_vect.transform(X_test)

	X_train_tfidf = tfidf_transformer.transform(X_train_counts)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)

	# Train instances count
	_, count = np.unique(y_train, return_counts=True)

	# Test instances count
	#_, count = np.unique(y_test, return_counts=True)

	weights = [1 - count[0] / len(X_train), 1 - count[1] / len(X_train)]

	sample_weight = np.array(len(y_train) * [weights[1]])
	sample_weight[np.where(np.array(y_train) == 0)[0]] = weights[0]

	if args.classifier == 'svm':
		clf = SGDClassifier(n_jobs=6).fit(X_train_tfidf, y_train, sample_weight=sample_weight)
	elif args.classifier == 'mnb':
		clf = MultinomialNB().fit(X_train_tfidf, y_train, sample_weight=sample_weight)
	elif args.classifier == 'lr':
		clf = LogisticRegression().fit(X_train_tfidf, y_train, sample_weight=sample_weight)

	predicted = clf.predict(X_test_tfidf)

	print(classification_report(y_test, predicted, target_names=speeches.target_names))

	#if args.save_misclassifications:
		#miss_list.extend(speeches.filenames[test_idx][predicted != y_test])

	acc.append(np.mean(predicted == y_test))
	print('Accuracy:', acc[-1])

	f1.append(f1_score(y_test, predicted, labels=speeches.target_names))
	print('F1 Score:', f1[-1])

	if f1[-1] > best_f1:
		best_f1 = f1[-1]
		dump(clf, 'clf.joblib')

print('\n\nCross-validation training stats:')
print('Accuracy Mean:\t\t', mean(acc))
print('Accuracy Std. Dev.:\t', stdev(acc))

print('F1 Score Mean:\t\t', mean(f1))
print('F1 Score Std. Dev.:\t', stdev(f1))
print('\n\n')

test_data = load_files(os.path.join(path, 'test'), categories=categories, encoding='utf-8', random_state=42)

X_test_counts = count_vect.transform(test_data.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

clf = load('clf.joblib')

predicted = clf.predict(X_test_tfidf)

print(classification_report(test_data.target, predicted, target_names=test_data.target_names))

if args.save_misclassifications:
	miss_list.extend(test_data.filenames[predicted != test_data.target])

acc.append(np.mean(predicted == test_data.target))
f1.append(f1_score(test_data.target, predicted, labels=test_data.target_names))

print('Test stats:')
print('Accuracy:\t\t', acc[-1])
print('F1 Score:\t\t', f1[-1])

if args.save_misclassifications:
	unique = set(miss_list)
	with open('misclassifications.txt', 'w') as f:
		for file in unique:
			f.write(str(file) + '\n')
