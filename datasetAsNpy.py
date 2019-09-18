# Cria dois conjuntos de arquivos npy: train e test. Cada um composto por seu correspondente X, contendo os caminhos para as imagens
# a serem lidas e y com suas respectivas labels.

import os
import numpy as np
import argparse


def readDataset(path):
    X = []
    y = []

    folders = os.listdir(path)
    folders.sort()

    for f in folders:
        for file in os.listdir(''.join((path, '/', f))):
            X.append(''.join((path, '/', f, '/', file)))
            y.append(int(f))

    return np.array(X), np.array(y)

parser = argparse.ArgumentParser(description='Reads images from PATH and stores data as numpy arrays. PATH must contain folders \
    \'test\' and \'train\'. Inside these folders, data must be organized in subfolders that will name its content class.')

parser.add_argument('path', metavar='PATH', help='Path to read files from.')

args = parser.parse_args()

PATH = args.path

train_path = PATH + '/train'
test_path = PATH + '/test'

X_train, y_train = readDataset(train_path)
X_test, y_test = readDataset(test_path)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
