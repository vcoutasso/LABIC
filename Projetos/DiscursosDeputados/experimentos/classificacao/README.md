# Classificação

Este experimento consiste em preparar um pequeno dataset devidamente anotado de forma que um discurso qualquer de um deputado esteja relacionado à seu voto final em plenário. O objetivo é treinar um classificador robusto o suficiente para ser utilizado no restante dos discursos a fim de encontrar o posicionamento presente em cada discurso.

Nota 1: Os discursos foram obtidos através do script de busca:
./prettyFind.py keywords "PEC 6/2019" -s ../experimentos/classificacao/discursos_previdencia -f 2019-01-01 -u 2019-08-06

Nota 2: Como a reforma da previdência foi à votação de segundo turno, provavelmente será necessário considerar a mudança de votos de alguns deputados.
