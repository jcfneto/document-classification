# O que é preciso ser feito

O objetivo deste trabalho é implementar classificadores de texto usando algoritmos de aprendizagem de máquina.
Siga as seguintes instruções para efetuar o trabalho:

- Faça o download da coleção Reuters-21578, 90 categories, do site http://disi.unitn.it/moschitti/corpora.htm.
A coleção está organizada em documentos para treino e teste, armazenados em diretórios de acordo com
a classe de cada um;

- Faça o preprocessamento dos dados, de forma a tratar codificações de caracteres, converter todo o texto
para letras minúsculas, eliminar pontuações, símbolos desnecessários, tags (no caso de HTML, XML etc.)
e stopwords, tokenizar cada palavra do texto etc. Prepare os documentos para serem usados como
entrada para os algoritmos;

- Escolha dois classificadores e compare sua eficácia para classificação nessa coleção. Para cada
classificador, experimente as seguintes duas estratégias para vetorização dos documentos de entrdada:
(1) use as palavras (tokens) como atributos (features) e o esquema de pesos TF-IDF como valor para os
atributos e (2) use um esquema de word embedding para as palavras e, para a entrada de cada
documento, use a média dos embeddings de suas palavras ou algum esquema de embedding de
documento, como o “Sentence Bert”;

- Ajuste os parâmetros dos classificadores fazendo uma busca em grid usando a técnica de validação
cruzada em 10 partes (10-fold cross validation). Use os dados de treino para isso. Depois de escolhidos os
melhores parâmetros, treine os classificadores com toda a coleção de treino e avalie-os usando a coleção
de teste;

- Avalie a qualidade dos resultados usando as seguintes métricas: precision, recall e F1 por classe, e
Macro-F1 e Micro-F1 (acurária) para o conjunto das classes. Apresente os resultados em forma de tabela,
comparando os classificadores. Faça um teste estatístico usando, por exemplo, a medida ANOVA, para
verificar se um classificador é realmente melhor do que o outro.

# Organização do repo

O repositório está organizado da seguinte forma:

- `output`: ...