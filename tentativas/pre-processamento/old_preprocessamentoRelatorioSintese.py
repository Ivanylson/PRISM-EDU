import pandas as pd
from docling.document_converter import DocumentConverter
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import unicodedata
import re

def selecionar_arquivo_pdf():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    arquivo = filedialog.askopenfilename(title="Selecione o PDF do Anexo IX", filetypes=[("Arquivos PDF", "*.pdf")])
    root.destroy()
    return arquivo

def limpeza_profunda(texto):
    """Remove quebras de linha, espaços extras e caracteres invisíveis."""
    if texto is None or pd.isna(texto): return ""
    # Substitui quebras de linha por espaço simples
    t = str(texto).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Remove espaços duplos e limpa as bordas
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def normalizar_nome_coluna(col):
    c = str(col).upper().strip()
    c = "".join(ch for ch in unicodedata.normalize('NFD', c) if unicodedata.category(ch) != 'Mn')
    c = re.sub(r'[^A-Z0-9]', '', c)
    # Dicionário de sinônimos para colunas "sujas"
    if any(x in c for x in ['POSICAO', 'QUESTAO', 'ITEM']): return 'POSIÇÃO'
    if 'PERFIL' in c: return 'PERFIL'
    if any(x in c for x in ['COMPETENCIA', 'HABILIDADE']): return 'COMPETÊNCIAS'
    if 'OC1' in c or 'OBJETO1' in c: return 'OC1'
    if 'OC2' in c or 'OBJETO2' in c: return 'OC2'
    if 'OC3' in c or 'OBJETO3' in c: return 'OC3'
    return c

def aplicar_regra_oc(row):
    """Junta OC1, OC2 e OC3 sem espaços extras."""
    ocs = [row.get(c) for c in ['OC1', 'OC2', 'OC3'] if c in row and len(str(row.get(c))) > 2]
    ocs = [str(o) for o in ocs if o]
    if not ocs: return ""
    if len(ocs) == 1: return ocs[0]
    return ", ".join(ocs[:-1]) + " e " + ocs[-1]

def tratar_linhas_grudadas(df):
    """Separa questões que o OCR leu juntas (ex: '37 38')."""
    novas_linhas = []
    for _, row in df.iterrows():
        pos = str(row.get('POSIÇÃO', '')).strip()
        matches = re.findall(r'(D1|D2|\d+)', pos)
        if len(matches) > 1:
            for m in matches:
                nova_r = row.copy()
                nova_r['POSIÇÃO'] = m
                novas_linhas.append(nova_r)
        else:
            novas_linhas.append(row)
    return pd.DataFrame(novas_linhas)

# --- EXECUÇÃO ---
caminho_pdf = selecionar_arquivo_pdf()
if not caminho_pdf: exit()

print(f"🤖 Analisando minuciosamente: {Path(caminho_pdf).name}...")

try:
    converter = DocumentConverter()
    result = converter.convert(caminho_pdf)
    
    # Extrai todas as tabelas encontradas em todas as páginas
    lista_dfs = []
    for table in result.document.tables:
        df_parcial = table.export_to_dataframe(result.document)
        # Normaliza colunas da tabela parcial
        df_parcial.columns = [normalizar_nome_coluna(c) for c in df_parcial.columns]
        lista_dfs.append(df_parcial)

    if not lista_dfs:
        print("❌ Nenhuma tabela detectada."); exit()

    # Une todas as partes da tabela
    df_bruto = pd.concat(lista_dfs, ignore_index=True, sort=False)
    
    # Remove colunas duplicadas que podem surgir na união
    df_bruto = df_bruto.loc[:, ~df_bruto.columns.duplicated()].copy()

    # Limpa o texto de TODAS as células antes de filtrar
    for col in df_bruto.columns:
        df_bruto[col] = df_bruto[col].apply(limpeza_profunda)

    # Identifica se o curso tem OC3 (pelo menos uma célula preenchida)
    tem_oc3 = 'OC3' in df_bruto.columns and df_bruto['OC3'].str.len().gt(2).any()

    # Filtra apenas o que é questão (D1, D2, 1-38)
    df_processado = tratar_linhas_grudadas(df_bruto)
    padrao = r'^(D1|D2|[1-9]|[1-2][0-9]|3[0-8])$'
    df_final = df_processado[df_processado['POSIÇÃO'].astype(str).str.match(padrao, na=False)].copy()

    # Remove duplicatas de questões
    df_final = df_final.drop_duplicates(subset=['POSIÇÃO'], keep='first')

    # Unifica OCs
    df_final['OC_unificado'] = df_final.apply(aplicar_regra_oc, axis=1)

    # Define se o arquivo final terá 6 ou 7 colunas
    cols_base = ['POSIÇÃO', 'PERFIL', 'COMPETÊNCIAS', 'OC1', 'OC2']
    if tem_oc3: cols_base.append('OC3')
    cols_base.append('OC_unificado')

    # Garante que as colunas existam no DF final
    for c in cols_base:
        if c not in df_final.columns: df_final[c] = ""

    # Salva o resultado
    output_path = Path(caminho_pdf).parent / (Path(caminho_pdf).stem + "_LIMPO.csv")
    df_final[cols_base].to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')

    print(f"\n✅ SUCESSO NA LIMPEZA!")
    print(f"📊 Total de Questões: {len(df_final)} de 40")
    print(f"📐 Estrutura: {len(cols_base)} colunas (OC3 detectado: {'Sim' if tem_oc3 else 'Não'})")
    print(f"📂 Arquivo gerado: {output_path.name}")

except Exception as e:
    print(f"❌ Erro: {e}")