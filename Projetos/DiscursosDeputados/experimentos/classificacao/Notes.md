# Notas sobre o experimento

Alguns pontos importantes para se ter em mente a respeito do experimento:

* As classes dizem respeito apenas ao voto do deputado que fez o discurso. Portanto, como a natureza do trabalho é justamente encontrar divergências entre o posicionamento no discurso e o voto do deputado, assume-se que há instâncias anotadas incorretamente.
* Assumindo-se que haja uma quantidade razoável dessas instâncias, o classificador não deveria ser capaz de atingir resultados excepcionais. Isso se dá devido ao fato de que essas instâncias seriam "outliers" e não seriam representativas da classe nas quais estão inseridas.


## Experimentos iniciais

O primeiro experimento realizado com os dados fez uso do classificador MultinomialNB, com n-gramas de 1 até 4, e sem remoção de stopwords, apesar de que, no processo de criação de bag of words, as palavras que são muito frequentes entre os documentos são removidas. Tokenização não foi utilizada por motivos de brevidade, mas deve ser fortemente considerada nos próximos experimentos.

Apesar de um experimento simples, a acurácia foi de 0.88 com um f1-score de 0.90 para a classe da maioria e 0.85 para a classe da minoria. Isso pode ser um indicador de que, como esperado, os discursos de mesmo posicionamento são realmente similares entre si, fazendo uso de mesmos argumentos, por exemplo.

As instâncias que foram classificadas incorretamente pelo classificador **podem** representar divergências discurso/voto.

O próximo passo para concluir este experimento é analisar os discursos incorretamente classificados em busca de divergências com a classe do texto.

### Fine-tuning

Em um segundo momento, fez se uso de _stemming_ com o algoritmo Snowball, e remoção de stop words. Validação cruzada com k=10 foi adotada a fim de obter resultados mais representativos do conjunto de dados.
