# TODO

- [x] Coletar dados da [base de dados](https://dadosabertos.camara.leg.br/) da Câmara dos Deputados
- [x] Filtrar discursos de interesse (reforma da previdência)
- [x] Fazer limpeza nos dados, removendo ruído
- [x] Treinar um classificador capaz de identificar discursos pelo seu posicionamento
- [ ] Fazer um benchmark de classificadores conforme [este link](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) demonstra*
- [ ] Escolher o classificador mais promissor e fazer um fine-tuning*
- [ ] Fazer clusterização dos discursos
- [ ] Comparar _outliers_ dos clusters com a classificação do experimento anterior
- [ ] Identificar instâncias classificadas erroneamente em busca de divergências discurso/voto

\* Talvez não seja necessário investir tanto tempo para encontrar um classificador melhor. Se o problema não for particularmente difícil, mesmo um classificador simples pode ser robusto o suficiente para a proposta do trabalho. 
