import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')

print("=============================================================================")
print("Módulo Corrigido: Batalha de Modelos Supervisionados sobre os Itens (Q1-Q38)")
print("=============================================================================")

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent
caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_resultados = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_FASE6_ML_AVANCADO'

try:
    df = pd.read_csv(caminho_microdados, sep=';', encoding='utf-8', low_memory=False)
    df = df[df['TP_PR_GER'] > 0].copy()
except Exception as e:
    print(f"Erro ao ler microdados: {e}")
    exit()

colunas_q = [f'Q{i}' for i in range(1, 39)]
for col in colunas_q:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).apply(lambda x: 1 if x == 1 else 0)

# Criando classe binária de proficiência macro baseada na mediana da prova objetiva
df['SOMA_OBJ'] = df[colunas_q].sum(axis=1)
corte = df['SOMA_OBJ'].median()
df['CLASSE'] = df['SOMA_OBJ'].apply(lambda x: 1 if x >= corte else 0)

X = df[colunas_q].values[:10000] # Subamostragem rápida e estável para benchmark
y = df['CLASSE'].values[:10000]

X_escalado = StandardScaler().fit_transform(X)

# 1. ElasticNet para controle de variância contra Overfitting
en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
en.fit(X_escalado, y)

# 2. Avaliação de Algoritmos via K-Fold Cross-Validation (5 Folds)
modelos = {
    "Regressão Logística": LogisticRegression(),
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
    "SVM (Linear)": LinearSVC(dual=False),
    "k-NN (5 Neighbors)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados_batalha = []

for nome, mod in modelos.items():
    scores = cross_val_score(mod, X_escalado, y, cv=cv, scoring='accuracy', n_jobs=-1)
    resultados_batalha.append({"Modelo": nome, "Acurácia Média CV": np.mean(scores), "Desvio Padrão": np.std(scores)})
    print(f"-> {nome}: {np.mean(scores):.4%}")

pd.DataFrame(resultados_batalha).to_csv(pasta_resultados / 'batalha_classificadores_classicos.csv', index=False, sep=';')