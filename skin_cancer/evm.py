#!/usr/bin/env python

import EVM

from argparse import ArgumentParser
import numpy as np
import os
from multiprocessing import Pool
import scipy
import csv
import pickle
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from pylmnn import LargeMarginNearestNeighbor as LMNN

np.random.seed(42)

CMAP = 'BuGn'

path = '/home/users/datasets/skin_cancer_2019/baseline_skin_2019.csv'

def fit(x_train, y_train, params):
		evms = {}
		for cl in np.unique(y_train): # train one evm for each positive class
				print('training evm for class', cl)
				#separate the positive class from the rest
				positives = [x for i,x in enumerate(x_train) if y_train[i] == cl]
				negatives = [x for i,x in enumerate(x_train) if y_train[i] != cl]

				evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.cosine)
				#evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.euclidean)
				evm.train(positives = positives, negatives = negatives, parallel = os.cpu_count())

				evms[cl] = evm

		return evms


def predict(evms, x_test, params):
		print('predicting samples')
		predictions = []
		probabilities_per_evm = {}
		for key, evm in evms.items():
				print('model ' + str(key))
				probabilities = evm.probabilities(x_test, parallel=os.cpu_count())
				max_prob = np.amax(probabilities, axis = 1)
				probabilities_per_evm[key] = max_prob

		# * operator unpacks values and keys into an array (python3 only)
		keys = [*probabilities_per_evm.keys()]
		values = np.array([*probabilities_per_evm.values()])
		#values = np.array(probabilities_per_evm.values())

		max_i = np.amax(values, axis = 0)
		argmax_i = np.argmax(values, axis=0)

		for m, argm in zip(max_i, argmax_i):
				if m > params['classification_threshold']:
						predictions.append(keys[argm])
				else:
						predictions.append(0)

		return predictions

begin = time.clock()

X = np.load('X.npy')
y = np.load('y.npy')
filenames = np.load('filenames.npy')
files_index = range(0, X.shape[0])

parser = ArgumentParser()
parser.add_argument('path', metavar='PATH', help='Path to read/write files from/to.')
parser.add_argument('--tail-size', default=10, help='EVM Tail Size. Default: 10')
parser.add_argument('--cover-threshold', default=0.5, help='EVM Cover Threshold. Default: 0.5')
parser.add_argument('--classification-threshold', default=0	, help='EVM Classification Threshold. Default: 0.0')
parser.add_argument('--test-split', default=0.3, help='Test split. Default: 0.30')
parser.add_argument('--evaluate-only', action='store_true',help='Evaluate existing EVMs and save results to PATH.')
parser.add_argument('--dont-stratify', action='store_true', help='Split data without stratification.')
parser.add_argument('--lmnn-transform', action='store_true', help='Perform LMNN transform on features.')
parser.add_argument('--k-folds', default=1, metavar='K', help='Number of K Folds.')
parser.add_argument('--quickie', action='store_true', help='Perform quick tests with only a minor part of the dataset.')

args = parser.parse_args()

PATH = args.path
TAIL_SIZE = args.tail_size
COVER_THRESHOLD = args.cover_threshold
CLASSIFICATION_THRESHOLD = args.classification_threshold
TEST_SPLIT = args.test_split
K = args.k_folds
TRAIN = None
STRATIFY = None
QUICKIE = args.quickie

if args.evaluate_only:
	TRAIN = False
	if not os.path.isfile(os.path.join(PATH, 'evms.pickle')):
		print('evms.pickle file not found!')
		exit(0)
else:
	TRAIN = True
	#create directory if it doesn't exists
	if not os.path.isdir(PATH):
		os.makedirs(PATH)

if args.dont_stratify:
	STRATIFY = None
else:
	STRATIFY = y

if args.lmnn_transform:
	LMNN_TRANSFORM = True
else:
	LMNN_TRANSFORM = False


params = {}
params['tail_size'] = TAIL_SIZE
params['cover_threshold'] = COVER_THRESHOLD
params['classification_threshold'] = CLASSIFICATION_THRESHOLD


labels = {	#'UNKN': 0,
			'AK'  : 1,
			'BCC' : 2,
			'BKL' : 3,
			'DF'  : 4,
			'MEL' : 5,
			'NV'  : 6,
			'SCC' : 7,
			'VASC': 8}



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, stratify=STRATIFY)
files_train, files_test, _, _ = train_test_split(files_index, y, test_size=TEST_SPLIT, stratify=STRATIFY)


if QUICKIE:
	X_train = X_train[:300]
	y_train = y_train[:300]

	X_test = X_test[:100]
	y_test = y_test[:100]

	files_train = files_train[:300]
	files_test = files_test[:100]

if LMNN_TRANSFORM:
	print('transforming features..')
	lmnn = LMNN(n_neighbors=3, max_iter=3, n_components=180, verbose=1, n_jobs=os.cpu_count())
	lmnn.fit(X_train, y_train)
	X_train = lmnn.transform(X_train)
	X_test = lmnn.transform(X_test)

if TRAIN:
	print('traning models..')
	evms = fit(X_train, y_train, params)
	#save evms
	dbfile = open(os.path.join(PATH,'evms.pickle'), 'wb')
	pickle.dump(evms, dbfile)
else:
	print('loading models..')
	evms = pickle.load(open(os.path.join(PATH,'evms.pickle'), 'rb'))



# make predictionsk
y_pred = predict(evms, X_test, params)


report = classification_report(y_test, y_pred, target_names=[*labels.keys()])
cf_matrix = confusion_matrix(y_test, y_pred, labels=[*labels.values()])

print(report)
print(cf_matrix)

with open(os.path.join(PATH,'report.txt'), 'w') as res:
	res.write('Classification Report\n\n')
	res.write(report)

with open(os.path.join(PATH,'predictions.csv'), 'w') as preds:
	preds.write('filename,ground_truth,prediction\n')
	for i, pred in enumerate(y_pred):
		preds.write(str(filenames[files_test[i]]) + ',' + str(y_test[i]) + ',' + str(pred) + '\n')

sns.heatmap(cf_matrix, annot=True, fmt='d', cmap=CMAP, xticklabels=[*labels.keys()], yticklabels=[*labels.keys()])
plt.ylabel('Ground Truth')
plt.xlabel('Predictions')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(PATH,'confusion_matrix.png'))

#Limpa o plot atual liberando espaço para o prox
plt.clf()

normalized_cm = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(normalized_cm, annot=True, fmt='.2f', cmap=CMAP, xticklabels=[*labels.keys()], yticklabels=[*labels.keys()])
plt.ylabel('Ground Truth')
plt.xlabel('Predictions')
plt.title('Normalized Confusion Matrix')
plt.savefig(os.path.join(PATH,'normalized_confusion_matrix.png'))

with open(os.path.join(PATH,'info.txt'), 'w') as info:
	info.write('EVM Parameters\n')
	info.write('Tail size: {}\n'.format(TAIL_SIZE))
	info.write('Cover Threshold: {}\n'.format(COVER_THRESHOLD))
	info.write('Classification Threshold: {}\n'.format(CLASSIFICATION_THRESHOLD))

	info.write('\nLMNN Transform: {}\n'.format('True' if args.lmnn_transform else 'False'))
	info.write('Stratified: {}\n'.format(('False' if args.dont_stratify else 'True')))
	info.write('Test split: {}\n'.format(TEST_SPLIT))
	info.write('K Fold: {}\n'.format(K))

	info.write('\nTime taken: {:.2f} min\n'.format(float(time.clock() - begin) / 60.0))
