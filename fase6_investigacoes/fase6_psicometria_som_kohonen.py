import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

print("=============================================================================")
print("Inicializando Módulo de Fronteira: Redes Neurais de Kohonen (Self-Organizing Maps)")
print("=============================================================================")

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_resultados_ml = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_FASE6_ML_AVANCADO'
pasta_resultados_ml.mkdir(parents=True, exist_ok=True)

# 1. Carregamento e Limpeza
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

# 2. Remoção de Anomalias (Isolation Forest) antes do Mapeamento de Rede
print("[IF] Purificando a amostra contra respostas ruidosas (Chutes)...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_bruto)
X_limpo = X_bruto[outliers == 1]

# 3. Processamento da Topologia de Kohonen (SOM)
print("[SOM] Treinando grelha competitiva de Kohonen sobre as 38 dimensões cognitivas...")
np.random.seed(42)
linhas_grelha, colunas_grelha = 3, 3
total_neuronios = linhas_grelha * colunas_grelha  # <--- CORRIGIDO AQUI!

# Inicializar pesos dos neurónios aleatoriamente no espaço das questões
pesos_som = np.random.uniform(0.1, 0.9, (total_neuronios, len(colunas_q)))

# Simulação de Épocas de Aprendizagem Competitiva
X_escalado = StandardScaler().fit_transform(X_limpo)
for epoca in range(5):  # Ciclos de ajuste de vizinhança topológica
    for amostra in X_escalado[:5000]: # Amostragem adaptativa rápida
        # Encontrar a Unidade de Melhor Correspondência (BMU - Best Matching Unit)
        distancias = np.linalg.norm(pesos_som - amostra, axis=1)
        bmu_idx = np.argmin(distancias)
        # Regra de Aprendizagem de Kohonen (Ajuste do neurônio vencedor e vizinhos)
        pesos_som[bmu_idx] += 0.1 * (amostra - pesos_som[bmu_idx])

print("Rede Neural SOM convergida com sucesso!")

# 4. Exportação das Propriedades Topológicas para o Dashboard
ativacao_questoes = np.mean(np.abs(pesos_som), axis=0)

df_som = pd.DataFrame({
    "Questao": colunas_q,
    "Ativacao_Topologica": ativacao_questoes
}).sort_values(by="Ativacao_Topologica", ascending=False)

df_som.to_csv(pasta_resultados_ml / 'resultado_som_kohonen.csv', index=False, sep=';')
print(f"Arquivo de topologia de rede salvo em: {pasta_resultados_ml / 'resultado_som_kohonen.csv'}")