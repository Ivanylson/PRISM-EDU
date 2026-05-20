import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

# Ignorar avisos de memória do Windows e do Matplotlib
warnings.filterwarnings('ignore')

# =============================================================================
# 1. MAPEAMENTO DE CURSOS E FUNÇÕES ÚTEIS
# =============================================================================
cursos_map = {
    5: 'Medicina Veterinária', 6: 'Odontologia', 12: 'Medicina', 17: 'Agronomia',
    19: 'Farmácia', 21: 'Arquitetura e Urbanismo', 23: 'Enfermagem', 27: 'Fonoaudiologia',
    28: 'Nutrição', 36: 'Fisioterapia', 51: 'Zootecnia', 55: 'Biomedicina',
    69: 'Tecnologia em Radiologia', 90: 'Tecnologia em Agronegócios', 
    91: 'Tecnologia em Gestão Hospitalar', 92: 'Tecnologia em Gestão Ambiental',
    95: 'Tecnologia em Estética e Cosmética', 5710: 'Engenharia Civil',
    5806: 'Engenharia Elétrica', 5814: 'Engenharia de Controle e Automação',
    5902: 'Engenharia Mecânica', 6002: 'Engenharia de Alimentos',
    6008: 'Engenharia Química', 6208: 'Engenharia de Produção',
    6307: 'Engenharia Ambiental', 6405: 'Engenharia Florestal',
    6410: 'Tecnologia em Segurança no Trabalho', 6411: 'Engenharia de Computação'
}

def formatar_nome(nome):
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_')

# =============================================================================
# 2. CONFIGURAÇÕES DE CAMINHOS AUTOMÁTICOS
# =============================================================================
print("Configurando diretórios de forma automática...")
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent # <--- Volta à raiz do projeto

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'

# Nova pasta para as validações dos agrupamentos
pasta_analises_agrupamentos = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS' / 'analisesAgrupamentos'
pasta_analises_agrupamentos.mkdir(parents=True, exist_ok=True)

if not caminho_microdados.exists():
    print(f"\n ERRO: Base de dados não encontrada em:\n{caminho_microdados}")
    print("Por favor, execute a Fase 1 primeiro.")
    exit()

print("A carregar a base de dados principal...")
df_micro = pd.read_csv(caminho_microdados, sep=';', dtype=str)
grupos_disponiveis = df_micro['CO_GRUPO'].dropna().unique()

print(f"Iniciando a Batalha de Algoritmos (Clustering) para {len(grupos_disponiveis)} cursos...\n")

# =============================================================================
# 3. MOTOR DE VALIDAÇÃO DE AGRUPAMENTOS POR CURSO
# =============================================================================
for co_grupo in grupos_disponiveis:
    co_grupo_int = int(co_grupo)
    
    if co_grupo_int not in cursos_map:
        continue
        
    nome_curso = cursos_map[co_grupo_int]
    nome_formatado = formatar_nome(nome_curso)
    
    df_curso = df_micro[df_micro['CO_GRUPO'] == str(co_grupo)].copy()
    
    # Ignora cursos com poucos alunos
    if len(df_curso) < 50:
        continue
        
    print(f"Validando Algoritmos para: {nome_curso}...")
    
    # Para evitar que o PC trave (estouro de memória) no modelo Hierárquico, 
    # usamos uma amostra representativa máxima de 2500 alunos por curso.
    n_amostra = min(len(df_curso), 2500)
    df_amostra = df_curso.sample(n_amostra, random_state=42)
    
    cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df_amostra.columns]
    for col in cols_questoes:
        df_amostra[col] = np.where(df_amostra[col] == '1', 1, 0)
        
    X = df_amostra[cols_questoes].values
    
    resultados_curso = []
    intervalo_k = range(2, 7) # Testa dividir em 2 até 6 grupos
    
    # Inércia apenas para K-Means (Método do Cotovelo)
    inercia_kmeans = []
    
    for k in intervalo_k:
        # 1. K-MEANS
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X)
        inercia_kmeans.append(kmeans.inertia_)
        sil_kmeans = silhouette_score(X, labels_kmeans)
        db_kmeans = davies_bouldin_score(X, labels_kmeans)
        
        # 2. GAUSSIAN MIXTURE MODELS (GMM)
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels_gmm = gmm.fit_predict(X)
        sil_gmm = silhouette_score(X, labels_gmm)
        db_gmm = davies_bouldin_score(X, labels_gmm)
        
        # 3. HIERÁRQUICO (Agglomerative)
        hier = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_hier = hier.fit_predict(X)
        sil_hier = silhouette_score(X, labels_hier)
        db_hier = davies_bouldin_score(X, labels_hier)
        
        resultados_curso.append({
            'K_Grupos': k,
            'KMeans_Silhueta': round(sil_kmeans, 4), 'KMeans_Davies': round(db_kmeans, 4), 'KMeans_Inercia': round(kmeans.inertia_, 2),
            'GMM_Silhueta': round(sil_gmm, 4), 'GMM_Davies': round(db_gmm, 4),
            'Hierarquico_Silhueta': round(sil_hier, 4), 'Hierarquico_Davies': round(db_hier, 4)
        })

    # =========================================================================
    # 4. SALVAR TABELA DE MÉTRICAS (CSV)
    # =========================================================================
    df_resultados = pd.DataFrame(resultados_curso)
    caminho_csv = pasta_analises_agrupamentos / f"metricas_agrupamento_{nome_formatado}.csv"
    df_resultados.to_csv(caminho_csv, sep=';', index=False, encoding='utf-8-sig')
    
    # =========================================================================
    # 5. GERAR E SALVAR OS GRÁFICOS (PNG)
    # =========================================================================
    plt.figure(figsize=(18, 5))
    
    # Gráfico A: Método do Cotovelo (Inércia K-Means)
    plt.subplot(1, 3, 1)
    plt.plot(intervalo_k, inercia_kmeans, 'bo-', linewidth=2)
    plt.title('K-Means: Método do Cotovelo (Inércia)')
    plt.xlabel('Número de Grupos (K)')
    plt.ylabel('Inércia (Menor é melhor)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico B: Comparação Silhouette Score
    plt.subplot(1, 3, 2)
    plt.plot(df_resultados['K_Grupos'], df_resultados['KMeans_Silhueta'], 'bo-', label='K-Means', linewidth=2)
    plt.plot(df_resultados['K_Grupos'], df_resultados['GMM_Silhueta'], 'rs-', label='GMM', linewidth=2)
    plt.plot(df_resultados['K_Grupos'], df_resultados['Hierarquico_Silhueta'], 'g^-', label='Hierárquico', linewidth=2)
    plt.title('Silhueta (Maior é melhor separação)')
    plt.xlabel('Número de Grupos (K)')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico C: Comparação Davies-Bouldin
    plt.subplot(1, 3, 3)
    plt.plot(df_resultados['K_Grupos'], df_resultados['KMeans_Davies'], 'bo-', label='K-Means', linewidth=2)
    plt.plot(df_resultados['K_Grupos'], df_resultados['GMM_Davies'], 'rs-', label='GMM', linewidth=2)
    plt.plot(df_resultados['K_Grupos'], df_resultados['Hierarquico_Davies'], 'g^-', label='Hierárquico', linewidth=2)
    plt.title('Davies-Bouldin (Menor é melhor coesão)')
    plt.xlabel('Número de Grupos (K)')
    plt.ylabel('Davies-Bouldin Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Salva o gráfico como imagem na mesma pasta
    caminho_imagem = pasta_analises_agrupamentos / f"graficos_validacao_{nome_formatado}.png"
    plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')
    plt.close() # Fecha a imagem para não estourar a memória RAM

print(f"\n{'='*75}")
print(f" PROCESSO CONCLUÍDO! Validações e Gráficos salvos em:\n{pasta_analises_agrupamentos}")
print(f"{'='*75}")