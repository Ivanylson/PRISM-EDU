import pandas as pd
import unicodedata
from pathlib import Path

# =============================================================================
# 1. MAPEAMENTO E FUNÇÕES DE FORMATAÇÃO
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

def formatar_nome_arquivo(nome):
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_') + '.csv'

# =============================================================================
# 2. CAMINHOS DINÂMICOS (Estrutura de Pastas Modular)
# =============================================================================
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent  # <--- Volta um nível para a raiz do projeto

caminho_relatorio_final = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_sintese = DIRETORIO_RAIZ / 'arquivosgerados' / 'resultadofinal_relatoriosintese' / 'csv'
caminho_saida = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS'/ 'analise_por_ies_curso_enade.csv'

# Garante que a pasta RESULTADOS existe antes de tentar salvar
caminho_saida.parent.mkdir(parents=True, exist_ok=True)

if not caminho_relatorio_final.exists():
    print(f"\n ERRO: O ficheiro de microdados não foi encontrado em:\n{caminho_relatorio_final}")
    print("Por favor, execute a Fase 1 (preprocessamentoMicrodados.py) primeiro.")
    exit()

# =============================================================================
# 3. LEITURA E PROCESSAMENTO DAS RESPOSTAS DOS ALUNOS
# =============================================================================
print("A carregar o relatório final do ENADE (Microdados)...")
# Lendo como string para evitar perda de zeros à esquerda nos códigos
df_enade = pd.read_csv(caminho_relatorio_final, sep=';', dtype=str)

# Colunas que definem um curso ÚNICO na IES
colunas_agrupamento = ['NU_ANO', 'CO_CURSO', 'CO_IES', 'CO_GRUPO', 'CO_UF_CURSO', 'CO_REGIAO_CURSO']

# Verifica se todas as colunas necessárias existem no CSV
for col in colunas_agrupamento:
    if col not in df_enade.columns:
        print(f" ERRO: A coluna {col} não foi encontrada no relatório final.")
        exit()

print("A calcular totais de alunos e deficiências por IES/Curso e por Questão...")

# Conta o total de alunos que fizeram a prova para aquele curso específico
df_total_alunos = df_enade.groupby(colunas_agrupamento).size().reset_index(name='TOTAL_ALUNOS')

resultados_questoes = []

# Analisa as questões de 1 a 38
for i in range(1, 39):
    q_col = f'Q{i}'
    if q_col in df_enade.columns:
        # Cria um dataframe temporário para calcular os erros dessa questão
        df_temp = df_enade[colunas_agrupamento].copy()
        
        # 1 = Acerto | 0 = Erro | 9, *, . = Brancos/Nulos
        df_temp['QTD_ERROS'] = (df_enade[q_col] == '0').astype(int)
        df_temp['QTD_BRANCOS_NULOS'] = df_enade[q_col].isin(['9', '*', '.']).astype(int)
        
        # Agrupa pelos dados do curso e soma os erros dos alunos
        df_agrupado_q = df_temp.groupby(colunas_agrupamento, as_index=False).sum()
        df_agrupado_q['QUESTAO'] = str(i) # Adiciona o número da questão para cruzamento
        
        resultados_questoes.append(df_agrupado_q)

# Empilha os resultados de todas as questões
df_todas_questoes = pd.concat(resultados_questoes, ignore_index=True)

# Mescla com a contagem total de alunos
df_analise = pd.merge(df_todas_questoes, df_total_alunos, on=colunas_agrupamento, how='left')

# Calcula a taxa de deficiência em percentagem
df_analise['TAXA_DEFICIENCIA_%'] = ((df_analise['QTD_ERROS'] + df_analise['QTD_BRANCOS_NULOS']) / df_analise['TOTAL_ALUNOS'] * 100).round(2)

# =============================================================================
# 4. CARREGAR MATRIZES E CRUZAR DADOS
# =============================================================================
print("A carregar relatórios de síntese (Competências e Objetos de Conhecimento)...")
df_sinteses_todas = pd.DataFrame()

# Carrega e empilha todos os ficheiros de síntese disponíveis
for co_grupo, nome_curso in cursos_map.items():
    nome_arquivo = formatar_nome_arquivo(nome_curso)
    caminho_sintese = pasta_sintese / nome_arquivo
    
    if caminho_sintese.exists():
        # CORREÇÃO: Alterado de sep=',' para sep=';'
        df_s = pd.read_csv(caminho_sintese, sep=',')
        
        # Renomeamos a coluna 'POSIÇÃO' para 'QUESTAO' para uniformizar e cruzar os dados
        if 'POSIÇÃO' in df_s.columns:
            df_s.rename(columns={'POSIÇÃO': 'QUESTAO'}, inplace=True)
            
        df_s['QUESTAO'] = df_s['QUESTAO'].astype(str).str.strip()
        df_s['CO_GRUPO'] = str(co_grupo)
        df_s['NOME_CURSO'] = nome_curso
        df_sinteses_todas = pd.concat([df_sinteses_todas, df_s], ignore_index=True)

print("A cruzar os dados de desempenho matemático com a pedagogia (Competências)...")

# Garante que os tipos batem para o merge
df_analise['CO_GRUPO'] = df_analise['CO_GRUPO'].astype(str)

# Cruzamento Final usando a coluna QUESTAO
df_final = pd.merge(df_analise, df_sinteses_todas, on=['CO_GRUPO', 'QUESTAO'], how='inner')

# Adiciona um "Q" antes do número da questão para ficar bem explícito (ex: Q1, Q15, Q38)
df_final['QUESTAO'] = 'Q' + df_final['QUESTAO']

# Reordena as colunas para o relatório ficar legível e perfeitamente estruturado
cols_order = [
    'NU_ANO', 'CO_IES', 'CO_CURSO', 'NOME_CURSO', 'CO_GRUPO', 'CO_UF_CURSO', 'CO_REGIAO_CURSO',
    'QUESTAO', 'TOTAL_ALUNOS', 'QTD_ERROS', 'QTD_BRANCOS_NULOS', 'TAXA_DEFICIENCIA_%',
    'PERFIL', 'COMPETÊNCIAS', 'OC1', 'OC2', 'OC3', 'OC_unificado'
]

# Mantém apenas as colunas que realmente existem (adicionamos as OCs extras de segurança)
cols_order = [c for c in cols_order if c in df_final.columns]
df_final = df_final[cols_order]

# Ordena por IES, depois Curso, e então pelas QUESTÕES com maior taxa de deficiência no topo
df_final = df_final.sort_values(by=['CO_IES', 'CO_CURSO', 'TAXA_DEFICIENCIA_%'], ascending=[True, True, False])

# =============================================================================
# 5. SALVAR BASE CONSOLIDADA
# =============================================================================
if not df_final.empty:
    df_final.to_csv(caminho_saida, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n{'='*60}")
    print(f" SUCESSO! Banco de Dados Consolidado gerado em:\n{caminho_saida}")
    print(f"{'='*60}")
else:
    print("\n AVISO: Nenhum dado foi cruzado. Verifique os códigos dos grupos e ficheiros de síntese.")