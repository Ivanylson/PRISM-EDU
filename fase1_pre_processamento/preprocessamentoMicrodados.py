import pandas as pd
import os
import gc
import requests
import zipfile
from pathlib import Path

def extrair_38_questoes(vetor_fg, vetor_ce):
    """Extrai as respostas num formato padronizado de 38 caracteres."""
    v_fg = str(vetor_fg).strip() if pd.notna(vetor_fg) and str(vetor_fg).lower() != 'nan' else ""
    v_ce = str(vetor_ce).strip() if pd.notna(vetor_ce) and str(vetor_ce).lower() != 'nan' else ""
    tudo = (v_fg + v_ce).ljust(38, '*')[:38]
    return list(tudo)

# =============================================================================
# 1. CONFIGURAÇÃO AUTOMÁTICA DA ESTRUTURA DE PASTAS (Sem Pop-ups)
# =============================================================================
print("Configurando diretórios de forma automática...")
# Descobre onde este script está e volta uma pasta para trás (Raiz do Projeto)
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent 

base_dir = DIRETORIO_RAIZ
pre_process_dir = base_dir / "preprocessamento"
gerados_dir = base_dir / "arquivosgerados"

# Cria as pastas caso ainda não existam
pre_process_dir.mkdir(parents=True, exist_ok=True)
gerados_dir.mkdir(parents=True, exist_ok=True)

url_zip = "https://download.inep.gov.br/microdados/microdados_enade_2023.zip"
caminho_zip = pre_process_dir / "microdados_enade_2023.zip"
arquivo_saida = gerados_dir / "relatorio_final_enade_2023.csv"

# =============================================================================
# 2. INÍCIO DO PROCESSO (Download, Extração e Tratamento)
# =============================================================================
try:
    # Passo 1: Download Automático
    if not caminho_zip.exists():
        print(f"Baixando microdados (INEP) para: {pre_process_dir}")
        response = requests.get(url_zip, stream=True)
        with open(caminho_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download finalizado!")
    else:
        print("Ficheiro ZIP já encontrado. Saltando download...")

    # Passo 2: Extração
    print("Descompactando arquivos (isto pode demorar um pouco)...")
    with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
        zip_ref.extractall(pre_process_dir)
    
    # Passo 3: Localização dos arquivos TXT extraídos
    caminhos_txt = list(pre_process_dir.rglob("*.txt"))
    file_arq1 = next(f for f in caminhos_txt if "microdados2023_arq1.txt" in f.name)
    file_arq3 = next(f for f in caminhos_txt if "microdados2023_arq3.txt" in f.name)

    # --- PROCESSAMENTO DOS DADOS ---
    print("Mapeando cursos (Arquivo 1)...")
    cols_arq1 = ['NU_ANO', 'CO_CURSO', 'CO_IES', 'CO_GRUPO', 'CO_UF_CURSO', 'CO_REGIAO_CURSO']
    df_ref = pd.read_csv(file_arq1, sep=';', usecols=cols_arq1, encoding='latin-1', on_bad_lines='skip')
    df_ref['CO_CURSO'] = pd.to_numeric(df_ref['CO_CURSO'], errors='coerce')
    mapa_cursos = df_ref.dropna(subset=['CO_CURSO']).drop_duplicates(subset=['CO_CURSO']).set_index('CO_CURSO').to_dict('index')
    del df_ref
    gc.collect()

    print("Lendo e filtrando respostas (Arquivo 3)...")
    # Mantidas as colunas extras que você adicionou
    cols_arq3 = [
        'NU_ANO', 'CO_CURSO', 'DS_VT_ACE_OFG', 'DS_VT_ACE_OCE', 
        'TP_PRES', 'TP_PR_GER', 'TP_PR_OB_FG', 'TP_PR_DI_FG', 
        'TP_PR_OB_CE', 'TP_PR_DI_CE', 'NT_FG_D1_PT', 'NT_FG_D1_CT',
        'NT_FG_D1', 'NT_DIS_CE', 'NT_CE_D1'
    ]
    
    lista_chunks = []
    # Lê em chunks para não estourar a memória RAM
    for chunk in pd.read_csv(file_arq3, sep=';', usecols=cols_arq3, encoding='latin-1', 
                             chunksize=100000, on_bad_lines='skip', low_memory=False):
        filt = chunk[chunk['TP_PR_GER'].isin([333, 555])].copy()
        if not filt.empty:
            lista_chunks.append(filt)
    
    df_total = pd.concat(lista_chunks, ignore_index=True)
    del lista_chunks
    gc.collect()

    print("Ordenando por CO_CURSO e gerando identificadores...")
    df_total['CO_CURSO'] = pd.to_numeric(df_total['CO_CURSO'], errors='coerce')
    df_total = df_total.dropna(subset=['CO_CURSO']).sort_values(by='CO_CURSO').reset_index(drop=True)
    
    df_total = df_total[df_total['CO_CURSO'].isin(mapa_cursos.keys())].copy()
    df_extras = pd.DataFrame(df_total['CO_CURSO'].map(mapa_cursos).tolist(), index=df_total.index)
    df_final = pd.concat([df_total.drop(columns=['NU_ANO']), df_extras], axis=1)

    # Contador reiniciável por curso
    contadores = {}
    ids_formatados = []
    for _, row in df_final.iterrows():
        c = int(row['CO_CURSO'])
        contadores[c] = contadores.get(c, 0) + 1
        ids_formatados.append(f"aluno{contadores[c]}_{int(row['NU_ANO'])}_{c}_{int(row['CO_IES'])}")
    
    df_final['ALUNO'] = ids_formatados

    # Formatação de Inteiros
    cols_int = ['TP_PRES', 'TP_PR_GER', 'TP_PR_OB_FG', 'TP_PR_DI_FG', 'TP_PR_OB_CE', 'TP_PR_DI_CE', 'CO_IES', 'CO_GRUPO', 'NU_ANO']
    for col in cols_int:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0).astype(int)

    # Ajuste de nomenclatura
    df_final = df_final.rename(columns={'NT_FG_D1_CT': 'NT_FG_D_CT'})

    print("Gerando questões Q1-Q38 (matriz matemática)...")
    res_list = df_final.apply(lambda r: extrair_38_questoes(r['DS_VT_ACE_OFG'], r['DS_VT_ACE_OCE']), axis=1)
    df_q = pd.DataFrame(res_list.tolist(), columns=[f'Q{i}' for i in range(1, 39)], index=df_final.index)
    
    output = pd.concat([df_final.drop(['DS_VT_ACE_OFG', 'DS_VT_ACE_OCE'], axis=1), df_q], axis=1)

    # Cabeçalho final atualizado
    cabecalho = [
        'ALUNO', 'NU_ANO', 'CO_CURSO', 'CO_IES', 'CO_GRUPO', 'CO_UF_CURSO', 'CO_REGIAO_CURSO',
        'TP_PRES', 'TP_PR_GER', 'TP_PR_OB_FG', 'TP_PR_DI_FG', 'TP_PR_OB_CE', 'TP_PR_DI_CE',
        'NT_FG_D1_PT', 'NT_FG_D_CT', 'NT_FG_D1', 'NT_DIS_CE', 'NT_CE_D1'
    ] + [f'Q{i}' for i in range(1, 39)]

    # Salva o arquivo CSV
    print(f"A guardar ficheiro final na pasta '{gerados_dir.name}'...")
    output[cabecalho].to_csv(arquivo_saida, index=False, sep=';', encoding='utf-8')
    
    print(f"\n PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f" Relatório guardado em: {arquivo_saida}")

except Exception as e:
    print(f"\n ERRO DURANTE A EXECUÇÃO: {e}")