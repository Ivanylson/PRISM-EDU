import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARREGAMENTO DOS DADOS (Amostra para teste de performance)
# =============================================================================
DIRETORIO_ATUAL = Path(__file__).resolve().parent
caminho_microdados = DIRETORIO_ATUAL / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'

print("Carregando base de dados para validação científica...")
# Usamos uma amostra de 3000 alunos para o teste não travar a memória, 
# já que algoritmos hierárquicos são pesados (O(N^2)).
df = pd.read_csv(caminho_microdados, sep=';', dtype=str).dropna(subset=['CO_GRUPO']).sample(3000, random_state=42)

cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df.columns]
for col in cols_questoes:
    df[col] = np.where(df[col] == '1', 1, 0)

X = df[cols_questoes].values

# =============================================================================
# 2. COMPETIÇÃO DE ALGORITMOS (O "MOTOR" DE VALIDAÇÃO)
# =============================================================================
print("Iniciando a Batalha de Algoritmos (K-Means vs GMM vs Hierárquico)...\n")

resultados = []
intervalo_k = range(2, 7) # Testando de 2 a 6 grupos

for k in intervalo_k:
    print(f"--> Avaliando divisão em {k} grupos...")
    
    # 1. K-MEANS
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    sil_kmeans = silhouette_score(X, labels_kmeans)
    db_kmeans = davies_bouldin_score(X, labels_kmeans)
    
    # 2. GAUSSIAN MIXTURE MODELS (GMM)
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels_gmm = gmm.fit_predict(X)
    sil_gmm = silhouette_score(X, labels_gmm)
    db_gmm = davies_bouldin_score(X, labels_gmm)
    
    # 3. HIERÁRQUICO (Agglomerative)
    hierarquico = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels_hier = hierarquico.fit_predict(X)
    sil_hier = silhouette_score(X, labels_hier)
    db_hier = davies_bouldin_score(X, labels_hier)
    
    # Salvando resultados
    resultados.append({
        'K (Grupos)': k,
        'K-Means (Silhueta)': sil_kmeans, 'K-Means (Davies)': db_kmeans,
        'GMM (Silhueta)': sil_gmm, 'GMM (Davies)': db_gmm,
        'Hierárquico (Silhueta)': sil_hier, 'Hierárquico (Davies)': db_hier
    })

df_resultados = pd.DataFrame(resultados)

# =============================================================================
# 3. PLOTANDO O DASHBOARD DE VALIDAÇÃO
# =============================================================================
plt.figure(figsize=(14, 6))

# Gráfico 1: Silhouette Score (Maior é melhor)
plt.subplot(1, 2, 1)
plt.plot(df_resultados['K (Grupos)'], df_resultados['K-Means (Silhueta)'], marker='o', label='K-Means', linewidth=2)
plt.plot(df_resultados['K (Grupos)'], df_resultados['GMM (Silhueta)'], marker='s', label='GMM', linewidth=2)
plt.plot(df_resultados['K (Grupos)'], df_resultados['Hierárquico (Silhueta)'], marker='^', label='Hierárquico', linewidth=2)
plt.title('Silhouette Score (Quanto MAIOR, melhor a separação)')
plt.xlabel('Número de Grupos (K)')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Gráfico 2: Davies-Bouldin Score (Menor é melhor)
plt.subplot(1, 2, 2)
plt.plot(df_resultados['K (Grupos)'], df_resultados['K-Means (Davies)'], marker='o', label='K-Means', linewidth=2)
plt.plot(df_resultados['K (Grupos)'], df_resultados['GMM (Davies)'], marker='s', label='GMM', linewidth=2)
plt.plot(df_resultados['K (Grupos)'], df_resultados['Hierárquico (Davies)'], marker='^', label='Hierárquico', linewidth=2)
plt.title('Davies-Bouldin Score (Quanto MENOR, melhor a coesão)')
plt.xlabel('Número de Grupos (K)')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

print("\nValidação concluída! Observe os gráficos gerados para a tomada de decisão.")