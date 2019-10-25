#!/usr/bin/env python

import os
import cv2
import numpy as np
from sklearn.neighbors import DistanceMetric
import time
import sys
from multiprocessing import Pool, current_process
import argparse
import signal
import tqdm
import string
from PIL import ImageEnhance
import PIL

# Variaveis globais
INPUT_DIR = None
OUTPUT_DIR = None
SIZE = None
SCALE = None
WORKERS = None
OUTPUT_FILE = None
FILE_FORMAT = None
VERBOSE = None
HIDE_BAR = None



# Funcao principal para calcular as cores e gerar a imagem de saida para cada imagem de entrada
def pixel_color(file):

	start_time = time.time()

	if file.endswith(FILE_FORMAT): #formato das imagens

		aux1 = str(INPUT_DIR) + str(file)	#path entrada
		aux2 = str(OUTPUT_DIR) + str(file)	#path saida

		centroids_labels = ['red','red','red','orange', 'orange', 'orange', 'yellow', 'yellow', 'yellow', 'green', 'green', 'green','blue', 'blue','blue','blue','purple','pink','pink','pink','pink','brown','orange','black','gray','silver','white','white']
		# red: maroon, firebrick e  red. orange: orange red, dark orange e orange. yellow: gold, golden rod e yellow. green: yellow green, dark green e lime.
		# blue: deep sky blue, dark blue, blue e steel blue. purple: blue violet, medium purple e purple. pink: magenta/fuchsia, deep pink e pink.
		# brown: saddle brown, sienna e chocolate. black: black. gray: gray. silver: silver. white: gainsboro e white
		centroids = [[128, 0, 0],[178, 34, 34],[255, 0, 0],[255, 69, 0],[255, 140, 0],[255, 165, 0],[255, 215, 0],[218, 165, 32],[255, 255, 0],
		[154, 205, 50],[0, 100, 0],[0, 255, 0],[0, 191, 255],[0, 0, 139],[0, 0, 255], [70, 130, 180], [128, 0, 128],[255, 0, 255],[255, 20, 147],[255,192,203], [199, 21, 133],
		[70, 40, 20], [210, 105, 30], [0, 0, 0], [105, 105, 105], [192, 192, 192], [220, 220, 220], [255, 255, 255]]

		image = cv2.imread(aux1, cv2.IMREAD_UNCHANGED)
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)


		#image = cv2.medianBlur(image,5) # Aplica filtro 5x5 na imagem

		relevantes = list() # pixels que nao sao transparentes

		if SIZE != None:
			image = cv2.resize(image, SIZE)
		elif SCALE != None:
			image = cv2.resize(image, (int(image.shape[0] * SCALE / 100), int(image.shape[1] * SCALE / 100)))

		for i in range(image.shape[0]):
			for j in range(image.shape[1]):
				if (image[i][j][3] > 32):
					relevantes.append([image[i][j][0], image[i][j][1], image[i][j][2]])

		Z = np.float32(relevantes)

		K = 5

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

		ret,label,center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

		center = np.uint8(center)
		res = center[label.flatten()]

		total = len(res)
		unique, counts = np.unique(res, axis=0, return_counts=True)

		dist = DistanceMetric.get_metric('euclidean') #Define qual distancia usar(euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis)
		distances = list()
		indexes = list()

		for i in range(K):
			aux = list()
			for j in range(len(centroids)):
				d = [unique[i], centroids[j]]
				aux.append(np.amax(dist.pairwise(d)))
			distances.append(min(aux))
			indexes.append(aux.index(min(aux)))

		colors = list()

		for i in indexes:
			try:
				colors.index(centroids_labels[i])
			except:
				colors.append(centroids_labels[i])

		frequency = [0] * len(colors)

		for i in range(len(colors)):
			for j in range(len(indexes)):
				if (colors[i] == centroids_labels[indexes[j]]):
					frequency[i] += counts[j] / total * 100


		csv = '{} dominant color: {}'.format(file, colors[frequency.index(max(frequency))])
		print(csv)

		if (file == 'image.png'):
			for i in range(len(colors)):
				print('{} frequency - {:.2f}%'.format(colors[i], frequency[i]))

			for i in range(K):
				print('{} Frequency: {:.2f}% - Closest color: {}'.format(unique[i], counts[i] / total * 100, centroids_labels[indexes[i]]))
			print('\n\n')

		return csv



def get_args():
	# Argumentos para serem passados ao script
	parser = argparse.ArgumentParser(prog='PixelColor')
	parser.add_argument('path', metavar='PATH',  help='path to directory to read images from')
	parser.add_argument('-c', '--cpus', help='determines how many CPUs to use. default=all', default='all')
	parser.add_argument('-sc', '--scale', help='scale image by a factor of SCALE per cent. default=no scaling', default=None, type=int)
	parser.add_argument('-s', '--size', help='resizes all images to SIZE (WIDTHxHEIGHT). overrides --scale. default=no resizing', default=None)
	parser.add_argument('-od', '--output_dir', help='directory to write modified images to. default=pixelcolor_output/', default='pixelcolor_output/')
	parser.add_argument('-of', '--output_file', help='file to write output to. default=output.csv', default='output.csv')
	parser.add_argument('-ff', '--file_format', help='format of images that will be read. default=png', default='png')
	parser.add_argument('-v', '--verbose', help='displays information of time taken to process each individual image', action='store_true')
	parser.add_argument('-hb', '--hide_bar', help='hides progress bar', action='store_true')

	# Recebe argumentos
	args = parser.parse_args()

	return args

# Funcao principal
def main():

	global INPUT_DIR
	global OUTPUT_DIR
	global SIZE
	global SCALE
	global WORKERS
	global OUTPUT_FILE
	global FILE_FORMAT
	global VERBOSE
	global HIDE_BAR

	args = get_args()


	INPUT_DIR = args.path
	print(INPUT_DIR)

	if args.cpus == 'all':
		WORKERS = os.cpu_count()
	else:
		WORKERS = int(args.cpus)

	if args.verbose is True:
		VERBOSE = True
	else:
		VERBOSE = False

	if args.hide_bar is True:
		HIDE_BAR = True
	else:
		HIDE_BAR = False

	if args.size != None:
		args.size.lower()
		x = args.size.index('x')
		width = int(args.size[:x])
		height = int(args.size[x+1:])
		SIZE = (width, height)
		print('Images will be of size {}x{}\n'.format(SIZE[0], SIZE[1]))
	else:
		SIZE = None

	if args.scale != None:
		SCALE = args.scale
		print('Images will be resized to {}% of their original size\n'.format(SCALE))
	else:
		SCALE = None

	if args.output_dir == 'pixelcolor_output/':
		OUTPUT_DIR = 'pixelcolor_output/'
	else:
		OUTPUT_DIR = args.output_dir

	if args.output_file == 'output.csv':
		OUTPUT_FILE = 'output.csv'
	else:
		OUTPUT_FILE = args.output_file

	if args.file_format == 'png':
		FILE_FORMAT = '.png'
	else:
		FILE_FORMAT = '.' + args.file_format


	# Cria diretorio que contem as imagens de saida
	try:
		os.mkdir(OUTPUT_DIR)
	except:
		pass

	saida = open(OUTPUT_FILE, 'w')
	#saida.write('arquivo,black,red,green,blue,yellow,magenta,cyan,maroon,purple,orange,gray,white\n')

	total_time = time.time()

	try:
		filelist = sorted(os.listdir(INPUT_DIR)) # Lista com todos os arquivos no diretorio
	except FileNotFoundError:
		print("Invalid directory!")
		return

	if (len(filelist) == 0):
		print("Empty directory!")
		return

	sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

	# Cria a pool e coloca os workers pra trabalhar
	print("Initializing {} workers..\n".format(WORKERS))

	try:
		with Pool(WORKERS) as p:
			signal.signal(signal.SIGINT, sigint_handler)
			if VERBOSE is True or HIDE_BAR is True:
				result = p.map(pixel_color, filelist)
			else:
				result = list(tqdm.tqdm(p.imap(pixel_color, filelist), total=len(filelist), bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}, {rate_fmt}{postfix}]'))
	except KeyboardInterrupt:
		print("\nCaught KeyboardInterrupt, terminating workers")
		p.terminate()
		sys.exit(0)
	else:
		p.close()
	p.join()

	# Salva os resultados no arquivo OUTPUT_FILE
	for i in result:
		saida.write('{}\n'.format(str(i)))

	saida.close()

	print("\nAll done! Total time taken: %.2f seconds" % (time.time() - total_time))



if __name__ == '__main__':
	main()
