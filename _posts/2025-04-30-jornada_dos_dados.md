---
title: "[1/10] Jornada dos Dados – Exploração e Preparação de Dados para Machine Learning"
date: 2025-04-21 00:00:00 +0800
categories: [Data Analyst, Machine Learning]
tags: [machine learning, análise de dados, preparação de dados, churn]
image: /assets/images/jornada-parte1-preview.png
---



# Jornada dos Dados com Machine Learning - Parte 2

## Transformando Dados em Decisões com Machine Learning

Na primeira parte desta jornada, analisamos a estrutura do conjunto de dados, realizamos a limpeza e discutimos os fundamentos de cada etapa. Agora, iniciaremos a aplicação de técnicas de Machine Learning para responder às seguintes questões:

1. Quais usuários apresentam maior propensão ao cancelamento?
2. Quais são os principais motivos que levam ao abandono do serviço?
3. Em que momento ocorre o abandono do serviço?

---

## Engenharia de Atributos

Antes da modelagem, é fundamental realizar a engenharia de atributos, criando variáveis que potencialmente aumentam a capacidade preditiva do modelo.

```python
# Exemplo de atributo: média de sessões por dia ativo
df['avg_sessions_per_day'] = df['sessions'] / df['activity_days'].replace(0, 1)

# Proporção entre dias com direção e dias ativos
df['driving_ratio'] = df['driving_days'] / df['activity_days'].replace(0, 1)
```

Criar novas variáveis permite evidenciar relações relevantes entre os dados, facilitando a identificação de padrões ocultos.

---

## Divisão dos Dados

Os dados serão divididos em conjuntos de treino e teste, a fim de avaliar o desempenho do modelo de maneira justa.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['label', 'ID'])  # Remoção da variável alvo e ID
y = df['label'].apply(lambda x: 1 if x == 'churned' else 0)  # Conversão para formato binário

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Essa divisão permite verificar a capacidade do modelo de generalizar para novos dados.

---

## Treinamento do Modelo

Utilizaremos o algoritmo Random Forest, que se destaca pela robustez, boa performance e interpretabilidade.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

A escolha desse algoritmo se justifica por sua capacidade de lidar com dados heterogêneos e de indicar a importância das variáveis utilizadas.

---

## Avaliação do Modelo

Serão utilizadas métricas de classificação para aferir a qualidade das previsões.

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

Esse relatório fornece uma visão abrangente sobre a acurácia, precisão, revocação e F1-score do modelo.

---

## Interpretação dos Resultados

A análise da importância das variáveis permite compreender os fatores mais determinantes para a previsão de cancelamento.

```python
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

# Seleção das 10 variáveis mais relevantes
indices = importances.argsort()[-10:][::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices])
plt.yticks(range(10), features[indices])
plt.xlabel("Importância")
plt.title("Top 10 Variáveis mais Relevantes")
plt.show()
```

Entender os fatores determinantes possibilita intervenções direcionadas, como ajustes no produto ou ações de retenção.

---

## Respostas Geradas pelo Modelo

1. **Quem são os usuários mais propensos ao cancelamento?**  
   - Usuários com baixa frequência de uso, poucos dias de direção e média reduzida de sessões diárias.

2. **Por que os usuários abandonam o serviço?**  
   - Baixo engajamento, uso limitado das funcionalidades e abandono precoce após o onboarding.

3. **Quando os usuários abandonam o serviço?**  
   - O modelo pode ser ajustado para estimar o período (semana ou mês) de maior risco de cancelamento, tema a ser explorado nas próximas etapas.

---

## Conclusão

Com base nas técnicas aplicadas, foi possível identificar padrões comportamentais associados ao cancelamento do serviço. Esses insights abrem caminho para ações preditivas, como campanhas de reengajamento, melhorias no aplicativo e estratégias focadas no onboarding.

---

Nas próximas publicações, abordaremos:
- Interpretações com SHAP e LIME;
- Previsão temporal do abandono;
- Estratégias de retenção personalizadas com base nos perfis identificados.

