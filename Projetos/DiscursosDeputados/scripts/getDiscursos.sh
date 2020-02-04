#!/bin/bash

pg=27

while read id; do
	curl -X GET "https://dadosabertos.camara.leg.br/api/v2/deputados/$id/discursos?dataInicio=2010-01-01&dataFim=2020-01-01&ordenarPor=dataHoraInicio&ordem=ASC&itens=1000&pagina=$pg" -H  "accept: application/json" > ../discursos/pg$pg/$id
done < ids
