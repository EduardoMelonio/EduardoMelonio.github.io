---
title: "[1/10] Jornada dos Dados – Entendendo e Preparando o Dataset"
date: 2025-04-21 00:00:00 +0800
categories: [Data Analyst, Machine Learning]
tags: [machine learning, análise de dados, preparação de dados, churn]
image: /assets/images/jornada-parte1-preview.png
---



# Jornada dos Dados com Machine Learning - Parte 1

## Objetivo do Projeto

Este projeto tem como finalidade responder a três perguntas centrais utilizando técnicas de Machine Learning:

1. Quais usuários apresentam maior propensão ao cancelamento?
2. Quais são os principais motivos que levam ao abandono do serviço?
3. Em que momento ocorre o abandono do serviço?

Para isso, utilizaremos um conjunto de dados proveniente da plataforma Kaggle, que contempla informações sobre o uso de um aplicativo de navegação.

---

## Carregamento dos Dados

Faremos uso da biblioteca `kagglehub` para realizar o download dos dados. É necessário que essa biblioteca esteja previamente instalada:

```bash
pip install kagglehub
```

Em seguida, o código abaixo permite o download automático do dataset:

```python
import kagglehub

# Download da versão mais recente do dataset
path = kagglehub.dataset_download("juliasuzuki/waze-dataset-to-predict-user-churn")

print("Caminho para os arquivos do dataset:", path)
```

---

## Análise Inicial do Dataset

Vamos iniciar examinando as primeiras linhas do conjunto de dados, a fim de compreender sua estrutura:

```python
import pandas as pd

# Carregamento do dataset
df = pd.read_csv(f"{path}/waze_dataset.csv")

# Visualização inicial
df.head()
```
---
![Visualização inicial do dataset](https://i.imgur.com/5nqeZRm.jpg)

## Compreensão das Variáveis

Cada registro do dataset representa um usuário, com diversas métricas relacionadas ao seu comportamento dentro do aplicativo. Dentre as variáveis mais relevantes, destacam-se:

- `label`: informa se o usuário foi retido ou cancelou o serviço;
- `sessions`, `drives`, `driven_km_drives`: indicadores de uso do aplicativo;
- `n_days_after_onboarding`: número de dias desde o cadastro;
- `device`: tipo de dispositivo utilizado (Android ou iPhone);
- `activity_days`, `driving_days`: frequência de uso do aplicativo.

Compreender essas variáveis é essencial para embasar decisões relacionadas à limpeza dos dados, transformação e seleção de atributos.

---

## Tratamento de Dados Ausentes

Procederemos agora à verificação de dados faltantes e à análise da distribuição das variáveis:

```python
# Informações gerais sobre o dataset
df.info()

# Verificação de valores nulos
df.isnull().sum()
```

Valores ausentes podem impactar negativamente a performance do modelo. Com base nessa análise, definiremos a estratégia mais adequada para tratá-los (remoção, imputação ou transformação).

---

## Análise de Balanceamento da Variável Alvo

Antes de construir os modelos, é importante avaliar se existe desbalanceamento entre as classes da variável alvo (`label`):

```python
df['label'].value_counts(normalize=True)
```

Desequilíbrios entre classes podem levar o modelo a privilegiar previsões da classe majoritária. Caso identificado, podemos aplicar técnicas como oversampling, undersampling ou uso de algoritmos específicos como SMOTE.

---

## Próximos Passos

Na próxima etapa da série, serão abordadas as seguintes atividades:

- Criação de novas variáveis (engenharia de atributos);
- Seleção de algoritmos adequados ao problema;
- Treinamento do modelo e avaliação dos resultados;
- Início da interpretação das respostas com base nos dados e nos insights gerados pelos modelos.

Recomenda-se a continuidade da leitura para compreender como é possível extrair histórias reais e insights significativos a partir dos dados.




