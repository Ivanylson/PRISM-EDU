import pandas as pd
from pathlib import Path
import unicodedata

# =============================================================================
# 1. FUNÇÕES E CAMINHOS AUTOMÁTICOS
# =============================================================================
def formatar_nome(nome):
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_')

print("Configurando diretórios de forma automática...")
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent # <--- Retorna à raiz do projeto

pasta_resultados = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS'
caminho_relatorio = pasta_resultados / 'relatorio_triplo_silhueta_grupos.csv'
pasta_sintese = DIRETORIO_RAIZ / 'arquivosgerados' / 'resultadofinal_relatoriosintese' / 'csv'

if not caminho_relatorio.exists():
    print(f"\n ERRO: Ficheiro de agrupamento não encontrado em:\n{caminho_relatorio}")
    print("Por favor, execute o 'motor_agrupamento_triplo.py' primeiro.")
    exit()

print("A ler o relatório de agrupamentos da IA...")
# Lendo com decimal=',' para respeitar os formatos que salvámos antes
df_grupos = pd.read_csv(caminho_relatorio, sep=';', decimal=',', dtype=str)

lista_deficiencias_oc = []
lista_deficiencias_comp = []
cache_sinteses = {}

print("A traduzir as questões para Objetos de Conhecimento e Competências...")

# =============================================================================
# 2. CRUZAMENTO DE DADOS (Tradutor)
# =============================================================================
for index, row in df_grupos.iterrows():
    nome_curso = row['CURSO']
    nome_formatado = formatar_nome(nome_curso)
    questoes_impactantes = str(row['QUESTOES_MAIS_IMPACTANTES']).split(', ')
    
    # Carrega a matriz do curso (com cache para não ler o ficheiro várias vezes)
    if nome_formatado not in cache_sinteses:
        caminho_sintese = pasta_sintese / f"{nome_formatado}.csv"
        if caminho_sintese.exists():
            # CORREÇÃO: Alterado sep=',' para sep=';' para ler corretamente os nossos CSVs limpos
            try:
             df_sint = pd.read_csv(caminho_sintese, sep=';', encoding='utf-8-sig')
             if len(df_sint.columns) ==1:
                 df_sint = pd.read_csv(caminho_sintese, sep=',', encoding='utf-8-sig')
            except pd.errors.ParserError:
                df_sint = pd.read_csv(caminho_sintese, sep=',', encoding='utf-8-sig', on_bad_lines='skip')
        
            if 'POSIÇÃO' in df_sint.columns:
                df_sint.rename(columns={'POSIÇÃO': 'QUESTAO'}, inplace=True)
            df_sint['QUESTAO'] = 'Q' + df_sint['QUESTAO'].astype(str).str.strip()
            cache_sinteses[nome_formatado] = df_sint
        else:
            cache_sinteses[nome_formatado] = None
            
    df_matriz = cache_sinteses[nome_formatado]
    texto_oc_final = []
    texto_comp_final = []
    
    if df_matriz is not None and questoes_impactantes[0] != 'nan':
        for q in questoes_impactantes:
            q = q.strip()
            linha_questao = df_matriz[df_matriz['QUESTAO'] == q]
            
            if not linha_questao.empty:
                linha = linha_questao.iloc[0]
                
                # Objetos de Conhecimento (Matérias)
                colunas_oc = [col for col in df_matriz.columns if str(col).startswith('OC')]
                ocs = [str(linha[col]).strip() for col in colunas_oc if pd.notna(linha[col]) and str(linha[col]).strip() != ""]
                str_ocs = " + ".join(ocs) if ocs else "Assunto Geral"
                texto_oc_final.append(f"**[{q}]** {str_ocs}")
                
                # Competências
                comp = "Não especificada"
                if 'COMPETÊNCIAS' in df_matriz.columns and pd.notna(linha['COMPETÊNCIAS']):
                    comp = str(linha['COMPETÊNCIAS']).strip()
                elif 'COMPETENCIA' in df_matriz.columns and pd.notna(linha['COMPETENCIA']):
                    comp = str(linha['COMPETENCIA']).strip()
                texto_comp_final.append(f"**[{q}]** {comp}")
            else:
                texto_oc_final.append(f"**[{q}]** Matéria não mapeada na matriz")
                texto_comp_final.append(f"**[{q}]** Competência não mapeada na matriz")
    else:
        texto_oc_final.append("Matriz do curso não encontrada")
        texto_comp_final.append("Matriz do curso não encontrada")
        
    lista_deficiencias_oc.append(" | ".join(texto_oc_final))
    lista_deficiencias_comp.append(" | ".join(texto_comp_final))

# =============================================================================
# 3. GUARDAR O ARQUIVO FINAL PARA O DASHBOARD
# =============================================================================
# PROTEÇÃO: Se a coluna já existir do processamento anterior, apaga antes de inserir
if 'MATERIAS_DEFICIENTES' in df_grupos.columns:
    df_grupos.drop(columns=['MATERIAS_DEFICIENTES'], inplace=True)
if 'COMPETENCIAS_A_DESENVOLVER' in df_grupos.columns:
    df_grupos.drop(columns=['COMPETENCIAS_A_DESENVOLVER'], inplace=True)

indice_coluna = df_grupos.columns.get_loc('QUESTOES_MAIS_IMPACTANTES') + 1

df_grupos.insert(indice_coluna, 'MATERIAS_DEFICIENTES', lista_deficiencias_oc)
df_grupos.insert(indice_coluna + 1, 'COMPETENCIAS_A_DESENVOLVER', lista_deficiencias_comp)

caminho_final = pasta_resultados / 'relatorio_diagnostico_pedagogico.csv'
df_grupos.to_csv(caminho_final, sep=';', index=False, encoding='utf-8-sig', decimal=',')

print(f"\n SUCESSO! Diagnóstico Pedagógico finalizado com sucesso!")
print(f" Ficheiro guardado em: {caminho_final}")