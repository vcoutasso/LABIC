#!/usr/bin/env python

import json
import os

for root,dirs,files in os.walk('../discursos_json'):
	for name in files:
		with open(os.path.join(root,name), 'r', encoding='utf-8') as f:
			print(f)
			j = json.load(f)

			for n in range(len(j)):
				j[n]['ideCadastro'] = name[:name.index('.')]

		with open(os.path.join(root,name), 'w', encoding='utf-8') as f:
			print(f)
			json.dump(j, f, ensure_ascii=False)
