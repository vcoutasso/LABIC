#!/usr/bin/env python3

import cv2
import os
from argparse import ArgumentParser

parser = ArgumentParser(description='Resizes images and saves them to OUTPUT. Original directory structure is preserved.')

parser.add_argument('size', metavar='SIZE', help='New image size (WIDTHxHEIGHT).')
parser.add_argument('path', metavar='PATH', help='PATH to images that will be resized.')
parser.add_argument('-o', '--output', metavar='OUTPUT', default='resizedImages_output', help='Output directory.')
parser.add_argument('-ff', '--file-format', default='jpg', help='File format of images that will be read. The script only supports one file format at a time.')

args = parser.parse_args()

SIZE = args.size
PATH = args.path
OUTPUT_DIR = args.output
FILE_FORMAT = args.file_format

# Faz com que os delimitadores x e X sejam validos
SIZE = SIZE.lower()

w, h = SIZE.split('x')

WIDTH = int(w)
HEIGHT = int(h)


try:
	os.listdir(PATH)
except FileNotFoundError:
	print('Error: Invalid path')
	exit(0)

# Inclui os caminhos absolutos para todas as images com o formato adequado.
images = []
# Inclui tambem o diretorio principal caso haja imagens nele. Apenas os caminhos relativos a PATH sao armazenados
directories = ['']

print('Creating directory structure...\n')
for root, dirs, files in os.walk(PATH):
	# Lista os diretorios em PATH 
	for d in dirs:
		directories.append(os.path.join(root.replace(PATH, ''), d))

	# Cria os diretorios onde as novas images serao salvas
		try:
			os.makedirs(os.path.join(OUTPUT_DIR, root.replace(PATH, ''), d))
		except FileExistsError:
			pass

	# Armazena os caminhos absolutos para as imagens de interesse
	for f in files:
		if f.endswith(FILE_FORMAT):
			images.append(os.path.join(root, f))

count = 0
total = len(images)

print('Reading and resizing images...\n')

for d in directories:
	images = []
	filelist = os.listdir(os.path.join(PATH, d))

	for f in filelist:
		if f.endswith(FILE_FORMAT):
			images.append(os.path.join(PATH, d, f))

	for im in images:
		img = cv2.imread(im)
		resized_img = cv2.resize(img, (WIDTH, HEIGHT))
		cv2.imwrite(os.path.join(OUTPUT_DIR, im.replace(PATH, '')), resized_img)
		count += 1

		if count % 1000 == 0:
			print('%d out of %d images resized' % (count, total))
