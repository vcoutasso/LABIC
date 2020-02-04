#!/usr/bin/env python

import os
import json

idList = list()

for root,dirs,files in os.walk('../discursos'):
	for id in files:
		if id != 'ids':
			if not id in idList:
				idList.append(id)

jsonList = [None] * len(idList)

for root,dirs,files in os.walk('../discursos'):
	for id in files:
		if id != 'ids':
			index = idList.index(id)

			with open(os.path.join(root,id), 'r', encoding='utf-8') as f:
				print(f)
				j = json.load(f)

				if jsonList[index] == None:
					jsonList[index] = j
				else:
					for it in j:
						jsonList[index].append(it)

for id in idList:
	with open(''.join(('../discursos_json/', id, '.json')), 'w', encoding='utf-8') as fp:
		print(fp)
		json.dump(jsonList[idList.index(id)], fp, ensure_ascii=False)
