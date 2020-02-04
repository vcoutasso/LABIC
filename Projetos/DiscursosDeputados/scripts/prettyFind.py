#!/usr/bin/env python

from pymongo import MongoClient
from argparse import ArgumentParser
from datetime import date
import os

parser = ArgumentParser(description='Script para encontrar discursos de deputados no banco de dados por meio de regular expressions e campo de interesse.')

parser.add_argument('campo', default='transcricao', nargs='?', help='Campo de interesse do discurso e.g. transcricao')
parser.add_argument('regex', help='Regular expression usada para filtrar resultados')
parser.add_argument('--print', '-p', action='store_true', help='Regular expression usada para filtrar resultados')
parser.add_argument('--save', '-s', default=None, help='Diretorio para salvar o resultado da busca em arquivos dedicados')
parser.add_argument('--from-date', '-f', default=None, help='Data minima, no padrao ISO 8601 (YYYY-MM-DD), para considerar um resultado')
parser.add_argument('--until-date', '-u', default=None, help='Data maxima, no padrao ISO 8601 (YYYY-MM-DD), para considerar um resultado')

args = parser.parse_args()

client = MongoClient()
db = client.CamaraDosDeputados
discursos = db.Discursos
deputados = db.Deputados

if args.save:
	if not os.path.exists(args.save):
		os.mkdir(args.save)

for i,discurso in enumerate(discursos.find({args.campo:{'$regex':args.regex}})):
	deputado = next(deputados.find({'ideCadastro':discurso['ideCadastro']}))
	id = deputado['ideCadastro']
	data = discurso['dataHoraInicio'].partition('T')[0]

	if args.from_date:
		if date.fromisoformat(data) < date.fromisoformat(args.from_date):
			continue
	if args.until_date:
		if date.fromisoformat(data) > date.fromisoformat(args.until_date):
			continue

	if args.print:
		print("Discurso nº %d por %s do %s em %s" % (i, deputado['nomeParlamentar'], deputado['partido'], data))
		print('SUMARIO\n', discurso['sumario'])
		print('TRANSCRIÇÃO\n', discurso['transcricao'])
		print()

	if args.save:
		if not os.path.exists(os.path.join(args.save, id)):
			os.mkdir(os.path.join(args.save, id))
		with open(os.path.join(args.save, ''.join((id, '/', str(i), '.txt'))), 'w') as f:
			f.write('SUMARIO\n' + discurso['sumario'])
			f.write('\nTRANSCRIÇÃO\n' + discurso['transcricao'])
			#f.write(discurso['transcricao'])
