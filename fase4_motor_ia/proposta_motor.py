import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES DE AMBIENTE
# =============================================================================
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent 

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
caminho_sintese = DIRETORIO_RAIZ / 'arquivosgerados' / 'resultadofinal_relatoriosintese' / 'csv' / 'engenharia_de_computacao.csv'
pasta_resultados = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_COMPUTACAO'
pasta_resultados.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TRATAMENTO DOS DADOS
# =============================================================================
def carregar_mapa(caminho):
    try:
        df = pd.read_csv(caminho, sep=',', quotechar='"', on_bad_lines='skip', engine='python')
        df.columns = [c.strip().upper() for c in df.columns]
        df['POSIÇÃO'] = pd.to_numeric(df['POSIÇÃO'], errors='coerce')
        return df.dropna(subset=['POSIÇÃO'])
    except: return None

# =============================================================================
#  MOTOR DE INTELIGÊNCIA ARTIFICIAL
# =============================================================================
print("Lendo microdados e aplicando filtros...")
df_micro = pd.read_csv(caminho_microdados, sep=';', dtype=str)
df_comp = df_micro[df_micro['CO_GRUPO'] == '6411'].copy() #ENGENHARIA DA COMPUTAÇÃO 
df_mapa = carregar_mapa(caminho_sintese)

# Limpeza de respostas (Converte caracteres do ENADE em números 0 ou 1)
cols_q = [f'Q{i}' for i in range(1, 39)]
for col in cols_q:
    if col in df_comp.columns:
        df_comp[col] = pd.to_numeric(df_comp[col].replace({'.': '0', '*': '0'}), errors='coerce').fillna(0).astype(int)

if df_mapa is not None:
    relatorio_final = []
    
    for ies in df_comp['CO_IES'].unique():
        df_ies = df_comp[df_comp['CO_IES'] == ies].copy()
        if len(df_ies) < 6: continue # Garantia estatística mínima

        # --- FASE 1: CONSTRUÇÃO DO ESPAÇO VETORIAL (EIXOS) ---
        X_cluster = pd.DataFrame(index=df_ies.index)
        eixos_analise = ['OC1', 'OC2', 'COMPETÊNCIAS']
        
        for eixo in eixos_analise:
            if eixo in df_mapa.columns:
                questoes_eixo = df_mapa[df_mapa[eixo].notna() & (df_mapa[eixo] != "")]['POSIÇÃO'].astype(int).unique()
                c_eixo = [f'Q{q}' for q in questoes_eixo if f'Q{q}' in df_ies.columns]
                if c_eixo:
                    # Média de acerto no eixo (Valor de 0.0 a 1.0)
                    X_cluster[eixo] = df_ies[c_eixo].mean(axis=1)

        if X_cluster.empty: continue

        # --- FASE 2: NORMALIZAÇÃO E CLUSTERIZAÇÃO ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        melhor_k, melhor_sil = 2, -1
        # Testamos de 2 a 4 grupos por IES
        for k in range(2, min(5, len(df_ies))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, lbls)
            if sil > melhor_sil: melhor_sil, melhor_k = sil, k

        # Aplicação do modelo definitivo
        km_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
        X_cluster['GRUPO_IA'] = km_final.fit_predict(X_scaled)
        
        # --- FASE 3: RANKING PEDAGÓGICO ---
        # Ordenamos os grupos pela média de todas as OCs
        colunas_oc = [c for c in eixos_analise if c in X_cluster.columns]
        rank = X_cluster.groupby('GRUPO_IA')[colunas_oc].mean().mean(axis=1).sort_values(ascending=False).index
        
        nomes_perfis = ["Alto Desempenho", "Intermediário", "Atenção", "Risco Crítico"]
        
        for i, p_id in enumerate(rank):
            grupo_X = X_cluster[X_cluster['GRUPO_IA'] == p_id]
            # Mapeamos de volta para o df_ies original para pegar as questões Q1-Q38
            grupo_original = df_ies.loc[grupo_X.index]
            
            nome_p = nomes_perfis[i] if i < len(nomes_perfis) else f"Perfil {i+1}"
            
            # Identificação das 5 questões com menor acerto no grupo
            media_questoes = grupo_original[cols_q].mean().sort_values()
            piores_q = media_questoes.head(5).index.tolist()
            
            # Cálculo de Desempenho Geral (Média das OCs)
            desempenho_final = grupo_X[colunas_oc].mean().mean() * 100

            relatorio_final.append({
                'IES': ies,
                'SILHUETA': round(melhor_sil, 4),
                'PERFIL': nome_p,
                'QTD_ALUNOS': len(grupo_X),
                'DESEMPENHO_GERAL_%': f"{round(desempenho_final, 1)}%",
                'QUESTOES_CRITICAS': ", ".join(piores_q)
            })

    # Salvamento
    df_salvar = pd.DataFrame(relatorio_final)
    df_salvar.to_csv(pasta_resultados / 'relatorio_final_computacao.csv', sep=';', index=False, encoding='utf-8-sig')
    print(f" Relatório gerado com sucesso em: {pasta_resultados}")