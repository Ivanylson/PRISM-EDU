import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. MAPEAMENTO E CONFIGURAÇÕES DE CAMINHOS AUTOMÁTICOS
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

print("Configurando diretórios de forma automática...")
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent 

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_resultados = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS'

pasta_resultados.mkdir(parents=True, exist_ok=True)

if not caminho_microdados.exists():
    print(f"\n❌ ERRO: Ficheiro de microdados não encontrado em:\n{caminho_microdados}")
    print("Por favor, execute a Fase 1 (Pré-processamento) primeiro.")
    exit()

print("A carregar a base de dados principal...")
df_micro = pd.read_csv(caminho_microdados, sep=';', dtype=str)
grupos_disponiveis = df_micro['CO_GRUPO'].dropna().unique()

# =============================================================================
# 2. MOTOR DE AGRUPAMENTO TRIPLO (Livre vs 10 vs 6)
# =============================================================================
dados_completos_grupos = []

for co_grupo in grupos_disponiveis:
    if int(co_grupo) not in cursos_map:
        continue
        
    nome_curso = cursos_map[int(co_grupo)]
    df_curso = df_micro[df_micro['CO_GRUPO'] == str(co_grupo)].copy()
    ies_disponiveis = df_curso['CO_IES'].dropna().unique()
    
    print(f"A processar K Triplo para: {nome_curso}...")
    
    for ies in ies_disponiveis:
        df_ies = df_curso[df_curso['CO_IES'] == ies].copy()
        
        if len(df_ies) < 15:
            continue
            
        cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df_ies.columns]
        for col in cols_questoes:
            df_ies[col] = np.where(df_ies[col] == '1', 1, 0)
            
        X = df_ies[cols_questoes]
        media_geral_ies = X.mean()
        
        # O CÁLCULO DOS 3 MUNDOS
        teto_livre = min(21, max(4, len(df_ies) // 2))
        
        melhor_k_livre = 2; silhueta_livre = -1
        melhor_k_10 = 2; silhueta_10 = -1
        melhor_k_6 = 2; silhueta_6 = -1
        
        for k in range(2, teto_livre):
            kmeans_teste = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_teste = kmeans_teste.fit_predict(X)
            sil = silhouette_score(X, labels_teste)
            
            if sil > silhueta_livre:
                silhueta_livre = sil
                melhor_k_livre = k
            if k <= 10 and sil > silhueta_10:
                silhueta_10 = sil
                melhor_k_10 = k
            if k <= 6 and sil > silhueta_6:
                silhueta_6 = sil
                melhor_k_6 = k

        # APLICAÇÃO PRÁTICA
        kmeans_final = KMeans(n_clusters=melhor_k_6, random_state=42, n_init=10)
        df_ies['PERFIL_IA'] = kmeans_final.fit_predict(X)
        
        perfis_temp = []
        for perfil in range(melhor_k_6):
            alunos_grupo = df_ies[df_ies['PERFIL_IA'] == perfil]
            if len(alunos_grupo) == 0: continue
            
            nota_media = alunos_grupo[cols_questoes].sum(axis=1).mean()
            perfis_temp.append({'id': perfil, 'nota': nota_media, 'alunos': alunos_grupo})
            
        perfis_temp.sort(key=lambda x: x['nota'], reverse=True)
        
        nomes_dinamicos = []
        if melhor_k_6 == 2: nomes_dinamicos = ["Alto Desempenho", "Risco Crítico"]
        elif melhor_k_6 == 3: nomes_dinamicos = ["Alto Desempenho", "Intermediário", "Risco Crítico"]
        elif melhor_k_6 == 4: nomes_dinamicos = ["Excelência", "Intermediário Superior", "Intermediário Inferior", "Risco Crítico"]
        elif melhor_k_6 == 5: nomes_dinamicos = ["Excelência", "Alto Desempenho", "Intermediário", "Atenção", "Risco Crítico"]
        elif melhor_k_6 == 6: nomes_dinamicos = ["Excelência", "Alto Desempenho", "Intermediário Superior", "Intermediário Inferior", "Atenção", "Risco Crítico"]
        
        # EXTRAÇÃO DE DADOS
        for rank, dados_grupo in enumerate(perfis_temp):
            alunos_grupo = dados_grupo['alunos']
            nome_perfil = nomes_dinamicos[rank]
            
            medias_questoes_grupo = alunos_grupo[cols_questoes].mean()
            gap = medias_questoes_grupo - media_geral_ies
            
            # --- NOVA LÓGICA: TODAS AS QUESTÕES COM PROBLEMA ---
            # Filtra apenas as questões onde o grupo foi pior que a média da IES (gap negativo)
            questoes_impactantes = gap[gap < 0].sort_values().index.tolist()
            
            # Se for um grupo de excelência e não tiver nenhuma nota abaixo da média, pegamos as 2 piores deles
            if len(questoes_impactantes) == 0:
                questoes_impactantes = gap.nsmallest(2).index.tolist()
            
            linha = {
                'CURSO': nome_curso,
                'IES': ies,
                'K_MATEMATICO_LIVRE': melhor_k_livre,
                'SILHUETA_LIVRE': round(silhueta_livre, 4),
                'K_LIMITE_10': melhor_k_10,
                'SILHUETA_LIMITE_10': round(silhueta_10, 4),
                'K_APLICADO_PEDAGOGICO (Max 6)': melhor_k_6,
                'NOME_DO_GRUPO': nome_perfil,
                'QTD_ALUNOS_GRUPO': len(alunos_grupo),
                'NOTA_MEDIA_GERAL_GRUPO': round(dados_grupo['nota'], 2),
                'QUESTOES_MAIS_IMPACTANTES': ", ".join(questoes_impactantes) # <- Agora traz todas!
            }
            
            for q in cols_questoes:
                linha[f'MEDIA_{q}_%'] = round(medias_questoes_grupo[q] * 100, 1)
                
            dados_completos_grupos.append(linha)

# =============================================================================
# 3. GUARDAR O RELATÓRIO
# =============================================================================
df_relatorio_final = pd.DataFrame(dados_completos_grupos)
caminho_csv = pasta_resultados / 'relatorio_triplo_silhueta_grupos.csv'

df_relatorio_final.to_csv(caminho_csv, sep=';', index=False, encoding='utf-8-sig', decimal=',')

print(f"\n✅ ANÁLISE TRIPLA (K-MEANS) CONCLUÍDA COM SUCESSO!")
print(f"📁 Relatório guardado e corrigido para o Excel em:\n{caminho_csv}")