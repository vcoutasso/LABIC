#!/bin/bash

while read id; do
	./../mongodb/bin/mongoimport --db CamaraDosDeputados --collection Discursos --file ../discursos_json/$id.json --jsonArray
done < ids
