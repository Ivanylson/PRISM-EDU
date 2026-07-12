import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

print("=============================================================================")
print("Inicializando Módulo de Fronteira 2026: Graph Knowledge Tracing (GKT) Estático")
print("=============================================================================")

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_resultados_ml = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_FASE6_ML_AVANCADO_PSICOMETRIA'
pasta_resultados_ml.mkdir(parents=True, exist_ok=True)

# 1. Carregamento e Filtro de Anomalias
try:
    df = pd.read_csv(caminho_microdados, sep=';', encoding='utf-8', low_memory=False)
    if 'TP_PR_GER' in df.columns:
        df = df[df['TP_PR_GER'] > 0].copy()
except Exception as e:
    print(f"Erro ao ler microdados: {e}")
    exit()

df.columns = [str(c).strip().upper() for c in df.columns]
colunas_q = [f'Q{i}' for i in range(1, 39)]

for col in colunas_q:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).apply(lambda x: 1 if x == 1 else 0)

X_bruto = df[colunas_q].values

print("[IF] Isolando ruídos de proficiência (Chutes)...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_bruto)
df_limpo = df[outliers == 1].copy()
X_limpo = df_limpo[colunas_q].values

# 2. Construção da Matriz de Adjacência do Grafo Cognitivo (Relação entre Itens)
print("[GKT] Construindo Grafo de Dependência de Competências (Item-to-Item Graph)...")
# Calculamos a covariância phi-coefficient entre as questões para criar os pesos das arestas do grafo
matriz_corr_grafo = np.corrcoef(X_limpo.T)
matriz_corr_grafo = np.nan_to_num(matriz_corr_grafo)

# Filtrar arestas fracas para manter a topologia do grafo limpa (Sparsificação)
limiar_aresta = 0.15
matriz_adjacencia = np.where(matriz_corr_grafo > limiar_aresta, matriz_corr_grafo, 0)
np.fill_diagonal(matriz_adjacencia, 0) # Remove self-loops

# 3. Extração de Features baseadas no Grafo (Propagação de Sinais)
# Multiplicar as respostas do aluno pela matriz de adjacência do grafo para capturar a influência da vizinhança
print("[GKT] Executando Message Passing (Propagação de Mensagens) no Grafo...")
X_grafo_features = np.dot(X_limpo, matriz_adjacencia)

# 4. Modelagem Preditiva de Fronteira (Grafo + Rede Neural)
print("[GKT] Treinando Rede Neural Preditiva sobre os Embeddings do Grafo...")
coluna_reg_alvo = 'NT_DIS_CE' if 'NT_DIS_CE' in df_limpo.columns else ('NT_CE_D1' if 'NT_CE_D1' in df_limpo.columns else 'SOMA_OBJETIVA')
df_limpo[coluna_reg_alvo] = pd.to_numeric(df_limpo[coluna_reg_alvo].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
y = df_limpo[coluna_reg_alvo].values

# Treinar preditor
gkt_predictor = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)
gkt_predictor.fit(X_grafo_features, y)
preds = gkt_predictor.predict(X_grafo_features)

rmse_gkt = np.sqrt(mean_squared_error(y, preds))
r2_gkt = r2_score(y, preds)
print(f"GKT Concluído -> R² Score obtido via Estrutura de Grafo: {r2_gkt:.4f}")

# 5. Exportação da Centralidade de Conhecimento das Questões no Grafo
centralidade_grau = np.sum(matriz_adjacencia, axis=1)
df_grafo_resultados = pd.DataFrame({
    "Questao": colunas_q,
    "Centralidade_Grafo": centralidade_grau
}).sort_values(by="Centralidade_Grafo", ascending=False)

df_grafo_resultados.to_csv(pasta_resultados_ml / 'resultado_graph_knowledge.csv', index=False, sep=';')

# Salva métricas de performance para o painel consolidar
df_perf_gkt = pd.DataFrame([{"Modelo": "Graph Knowledge Tracing (GKT)", "RMSE": rmse_gkt, "R2 Score": r2_gkt}])
df_perf_gkt.to_csv(pasta_resultados_ml / 'performance_gkt.csv', index=False, sep=';')

print("Resultados de Redes em Grafos salvos com sucesso!")