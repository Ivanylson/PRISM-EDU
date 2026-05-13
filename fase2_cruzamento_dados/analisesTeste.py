import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURAÇÃO DE CAMINHOS (Arquitetura Modular)
# =============================================================================
print("Configurando diretórios de forma automática...")
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent  # <--- Volta um nível para a raiz do projeto

# Arquivos de entrada (Gerados na Fase 1 e Fase 2.1)
caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
caminho_base_ies = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS' / 'analise_por_ies_curso_enade.csv' 

# Pasta de saída dos novos relatórios de análise
pasta_resultados = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS'

# Cria a pasta RESULTADOS se ela não existir (por garantia)
pasta_resultados.mkdir(parents=True, exist_ok=True)

if not caminho_microdados.exists() or not caminho_base_ies.exists():
    print("\n❌ ERRO: Um dos ficheiros base não foi encontrado.")
    print(f"Verifique se os ficheiros existem nos caminhos:\n1. {caminho_microdados}\n2. {caminho_base_ies}")
    print("Dica: Certifique-se de executar as Fases 1 e 2.1 primeiro!")
    exit()

print("Carregando bases de dados (isso pode levar alguns segundos)...")
df_microdados = pd.read_csv(caminho_microdados, sep=';', dtype=str)
df_base = pd.read_csv(caminho_base_ies, sep=';')

# =============================================================================
# ANÁLISE 1: FADIGA DE PROVA (EFEITO "CHUTE" / DESISTÊNCIA)
# =============================================================================
print("Gerando Análise 1: Fadiga de Prova...")
fadiga_data = []
total_alunos_br = len(df_microdados)

for i in range(1, 39):
    q_col = f'Q{i}'
    if q_col in df_microdados.columns:
        # Conta respostas em branco ('9'), nulas/duplas ('*') ou anuladas ('.')
        qtd_brancos_nulos = df_microdados[q_col].isin(['9', '*', '.']).sum()
        taxa_abandono = (qtd_brancos_nulos / total_alunos_br) * 100
        
        fadiga_data.append({
            'QUESTAO': q_col,
            'BLOCO': 'Formação Geral (1 a 10)' if i <= 10 else 'Componente Específico (11 a 38)',
            'QTD_ABANDONO_BRANCO': qtd_brancos_nulos,
            'TAXA_ABANDONO_%': round(taxa_abandono, 2)
        })

df_fadiga = pd.DataFrame(fadiga_data)
df_fadiga.to_csv(pasta_resultados / 'analise_1_fadiga_prova.csv', sep=';', index=False, encoding='utf-8-sig')

# =============================================================================
# ANÁLISE 2: FORMAÇÃO GERAL (FG) VS COMPONENTE ESPECÍFICO (CE)
# =============================================================================
print("Gerando Análise 2: Formação Geral vs Componente Específico...")
# Extrai o número da questão para classificar
df_base['NUM_QUESTAO'] = df_base['QUESTAO'].str.replace('Q', '').astype(int)
df_base['BLOCO_PROVA'] = np.where(df_base['NUM_QUESTAO'] <= 10, 'FG (Geral)', 'CE (Específica)')

# Agrupa pela IES e pelo Bloco para ver a média de deficiência
df_fg_ce = df_base.groupby(['CO_IES', 'NOME_CURSO', 'BLOCO_PROVA'])['TAXA_DEFICIENCIA_%'].mean().reset_index()
df_fg_ce.rename(columns={'TAXA_DEFICIENCIA_%': 'MEDIA_DEFICIENCIA_%'}, inplace=True)
df_fg_ce['MEDIA_DEFICIENCIA_%'] = df_fg_ce['MEDIA_DEFICIENCIA_%'].round(2)

# Pivotar para ficar uma coluna para FG e outra para CE lado a lado
df_fg_ce_pivot = df_fg_ce.pivot(index=['CO_IES', 'NOME_CURSO'], columns='BLOCO_PROVA', values='MEDIA_DEFICIENCIA_%').reset_index()
df_fg_ce_pivot['DIFERENCA_CE_FG'] = df_fg_ce_pivot['CE (Específica)'] - df_fg_ce_pivot['FG (Geral)']

df_fg_ce_pivot.to_csv(pasta_resultados / 'analise_2_fg_vs_ce.csv', sep=';', index=False, encoding='utf-8-sig')

# =============================================================================
# ANÁLISE 3: OBJETIVAS VS DISCURSIVAS (O GARGALO DA ESCRITA)
# =============================================================================
print("Gerando Análise 3: Objetivas vs Discursivas...")
# Precisamos converter as notas discursivas para float
cols_notas = ['NT_FG_D1', 'NT_CE_D1']
for col in cols_notas:
    if col in df_microdados.columns:
        df_microdados[col] = pd.to_numeric(df_microdados[col].str.replace(',', '.'), errors='coerce')

# Calcular acertos nas objetivas (soma de '1')
cols_q = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df_microdados.columns]
df_microdados['TOTAL_ACERTOS_OBJ'] = (df_microdados[cols_q] == '1').sum(axis=1)

# Agrupar por IES e Curso
df_escrita = df_microdados.groupby(['CO_IES', 'CO_GRUPO']).agg({
    'TOTAL_ACERTOS_OBJ': 'mean',
    'NT_FG_D1': 'mean', 
    'NT_CE_D1': 'mean'  
}).reset_index()

df_escrita = df_escrita.round(2)
df_escrita.rename(columns={
    'TOTAL_ACERTOS_OBJ': 'MEDIA_ACERTOS_OBJETIVAS_MAX_38',
    'NT_FG_D1': 'MEDIA_DISCURSIVA_GERAL_MAX_100',
    'NT_CE_D1': 'MEDIA_DISCURSIVA_ESPECIFICA_MAX_100'
}, inplace=True)

df_escrita.to_csv(pasta_resultados / 'analise_3_objetivas_vs_discursivas.csv', sep=';', index=False, encoding='utf-8-sig')

# =============================================================================
# ANÁLISE 4: RANKING NACIONAL DE LACUNAS POR OBJETO DE CONHECIMENTO (OC1)
# =============================================================================
print("Gerando Análise 4: Ranking Nacional por Objeto de Conhecimento...")
if 'OC1' in df_base.columns:
    df_oc = df_base.groupby(['NOME_CURSO', 'OC1'])['TAXA_DEFICIENCIA_%'].mean().reset_index()
    df_oc.rename(columns={'TAXA_DEFICIENCIA_%': 'MEDIA_NACIONAL_DEFICIENCIA_%'}, inplace=True)
    df_oc = df_oc.sort_values(by=['NOME_CURSO', 'MEDIA_NACIONAL_DEFICIENCIA_%'], ascending=[True, False])
    df_oc['MEDIA_NACIONAL_DEFICIENCIA_%'] = df_oc['MEDIA_NACIONAL_DEFICIENCIA_%'].round(2)
    
    df_oc.to_csv(pasta_resultados / 'analise_4_ranking_nacional_oc1.csv', sep=';', index=False, encoding='utf-8-sig')

# =============================================================================
# ANÁLISE 5: BENCHMARK REGIONAL (IES vs ESTADO)
# =============================================================================
print("Gerando Análise 5: Benchmark Regional (IES vs Estado)...")
# Calcula a média geral de deficiência por UF e por Curso
df_estado = df_base.groupby(['CO_UF_CURSO', 'NOME_CURSO'])['TAXA_DEFICIENCIA_%'].mean().reset_index()
df_estado.rename(columns={'TAXA_DEFICIENCIA_%': 'MEDIA_ESTADUAL_%'}, inplace=True)

# Calcula a média geral da IES no Curso
df_ies_media = df_base.groupby(['CO_IES', 'CO_UF_CURSO', 'NOME_CURSO'])['TAXA_DEFICIENCIA_%'].mean().reset_index()
df_ies_media.rename(columns={'TAXA_DEFICIENCIA_%': 'MEDIA_IES_%'}, inplace=True)

# Cruza a IES com a média do Estado
df_benchmark = pd.merge(df_ies_media, df_estado, on=['CO_UF_CURSO', 'NOME_CURSO'], how='left')

# Se a diferença for Negativa, a IES está MELHOR (erra menos) que o Estado.
df_benchmark['DIFERENCA_PARA_O_ESTADO'] = (df_benchmark['MEDIA_IES_%'] - df_benchmark['MEDIA_ESTADUAL_%']).round(2)
df_benchmark['MEDIA_IES_%'] = df_benchmark['MEDIA_IES_%'].round(2)
df_benchmark['MEDIA_ESTADUAL_%'] = df_benchmark['MEDIA_ESTADUAL_%'].round(2)

# Ordena para destacar quem está muito pior que o estado no topo
df_benchmark = df_benchmark.sort_values(by=['NOME_CURSO', 'DIFERENCA_PARA_O_ESTADO'], ascending=[True, False])

df_benchmark.to_csv(pasta_resultados / 'analise_5_benchmark_regional.csv', sep=';', index=False, encoding='utf-8-sig')

print(f"\n{'='*60}")
print(f"✅ SUCESSO ABSOLUTO! Os 5 relatórios foram salvos na pasta:\n{pasta_resultados}")
print(f"{'='*60}")