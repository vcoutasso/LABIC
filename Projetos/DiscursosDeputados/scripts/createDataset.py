import os
import json
import shutil
from argparse import ArgumentParser

votacao = None

parser = ArgumentParser(description='Script para criar um dataset onde as classes são os votos dos deputados contendo todos os respectivos discursos presentes no caminho de entrada')
parser.add_argument('input', help='Caminho contendo os discursos. Espera se que haja um subdiretorio para cada deputado contendo seus discursos')
parser.add_argument('output', help='Caminho para armazenar o dataset criado')
parser.add_argument('votacao', help='Caminho para arquivo json contendo os votos que serão usados como classe')

args = parser.parse_args()

if not os.path.exists(args.output):
	os.mkdir(args.output)

if not os.path.exists(os.path.join(args.output, 'Sim')):
	os.mkdir(os.path.join(args.output, 'Sim'))

if not os.path.exists(os.path.join(args.output, 'Nao')):
	os.mkdir(os.path.join(args.output, 'Nao'))

if not os.path.exists(os.path.join(args.output, 'Abstencao')):
	os.mkdir(os.path.join(args.output, 'Abstencao'))

if not os.path.exists(os.path.join(args.output, 'Obstrucao')):
	os.mkdir(os.path.join(args.output, 'Obstrucao'))


with open(args.votacao) as f:
	votacao = json.load(f)

for root,dir,file in os.walk(args.input):
	try:
		id = root.split('/')[-1]
	# Se for o diretorio raiz
	except:
		continue

	try:
		voto = list(filter(lambda x : x['ideCadastro'] == id, votacao['votos']))[0]['Voto']


		if voto == 'Sim':
			dir = 'Sim'
		elif voto == 'Não':
			dir = 'Nao'
		elif voto == '-':
			dir = 'Abstencao'
		# Não sei se é possível votar 'obstrução', pois não houve nenhum, mas por via das dúvidas...
		else:
			dir = 'Obstrucao'

		for n,f in enumerate(file):
			print('Copying',  f)
			shutil.copy(os.path.join(root, f), os.path.join(args.output, dir))

	# Se o deputado não votou
	except:
		continue

	continue
