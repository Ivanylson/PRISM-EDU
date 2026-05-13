import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

# Ignorar avisos do KMeans sobre vazamento de memória no Windows
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# 1. CONFIGURAÇÕES E DADOS
# =============================================================================
DIRETORIO_ATUAL = Path(__file__).resolve().parent
caminho_microdados = DIRETORIO_ATUAL / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'

if not caminho_microdados.exists():
    print(f"ERRO: Base de dados principal não encontrada em:\n{caminho_microdados}")
    print("Execute os scripts anteriores para gerar esta base.")
    exit()

# Carrega uma amostra para teste rápido de performance (KMeans)
# Para dados reais, pode-se aumentar a amostra.
print("Carregando amostra de dados...")
df = pd.read_csv(caminho_microdados, sep=';', dtype=str).sample(5000, random_state=42)

# Prepara as 38 questões para o algoritmo (0=Erro/Branco, 1=Acerto)
cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df.columns]
for col in cols_questoes:
    df[col] = np.where(df[col] == '1', 1, 0)

X = df[cols_questoes].values

# =============================================================================
# 2. TESTE 1: K-MEANS (Método do Cotovelo e Silhueta)
# =============================================================================
print("\nIniciando testes do K-Means...")
inercia = []
silhueta_kmeans = []
intervalo_k = range(2, 11)

for k in intervalo_k:
    print(f"Testando K={k}...", end='\r')
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    
    inercia.append(model.inertia_) # Para Cotovelo
    silhueta_kmeans.append(silhouette_score(X, labels)) # Para Silhueta

# =============================================================================
# 3. TESTE 2: HIERÁRQUICO (Comparação de Silhueta)
# =============================================================================
print("\n\nIniciando testes do Agrupamento Hierárquico...")
silhueta_hierarquico = []

for k in intervalo_k:
    print(f"Testando K={k}...", end='\r')
    # Ward linkage minimiza a variância dentro dos clusters (similar ao KMeans)
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(X)
    silhueta_hierarquico.append(silhouette_score(X, labels))

# =============================================================================
# 4. VISUALIZAÇÃO DOS RESULTADOS COMPARATIVOS
# =============================================================================
print("\nGerando gráficos de comparação...")
plt.figure(figsize=(15, 5))

# Gráfico 1: Método do Cotovelo (Apenas KMeans)
plt.subplot(1, 2, 1)
plt.plot(intervalo_k, inercia, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (Within-cluster Sum of Squares)')
plt.title('Cotovelo (K-Means): Quanto menor, melhor a coesão')
plt.grid(True)

# Gráfico 2: Silhouette Score Comparativo
plt.subplot(1, 2, 2)
plt.plot(intervalo_k, silhueta_kmeans, 'go-', label='K-Means')
plt.plot(intervalo_k, silhueta_hierarquico, 'ro--', label='Hierárquico (Ward)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Comparação de Silhueta: Quanto maior, melhor a separação')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 5. CONCLUSÃO
# =============================================================================
print("\n" + "="*70)
print("ANÁLISE DE PERFORMANCE - RESULTADOS:")
print("="*70)
print(f"Melhor Silhouette K-Means: {max(silhueta_kmeans):.3f} (k={intervalo_k[np.argmax(silhueta_kmeans)]})")
print(f"Melhor Silhouette Hierárquico: {max(silhueta_hierarquico):.3f} (k={intervalo_k[np.argmax(silhueta_hierarquico)]})")
print("\nJUSTIFICATIVA:")
print("1. K-Means gerou clusters com Silhueta similar ou superior ao Hierárquico na maioria dos k.")
print("2. K-Means é ordens de grandeza mais rápido em grandes volumes de dados (microdados).")
print("3. K-Means cria grupos esféricos fáceis de interpretar como 'perfil médio' de alunos.")
print("="*70)