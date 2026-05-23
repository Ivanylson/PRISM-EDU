import pandas as pd
import requests
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import os
import unicodedata
import re

def selecionar_pasta():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    caminho = filedialog.askdirectory(title="Selecione a pasta 'projetoENADE'")
    root.destroy()
    return caminho

def limpeza_radical(texto):
    """Remove quebras de linha, espaços duplos e pontos e vírgulas residuais."""
    # Se for uma lista (erro comum em OCR), pega o primeiro elemento
    if isinstance(texto, list):
        texto = texto[0] if len(texto) > 0 else ""
    
    if texto is None or pd.isna(texto): return ""
    
    t = str(texto).replace('\n', ' ').replace('\r', ' ')
    t = t.replace(';', ' ')
    return " ".join(t.split()).strip()

def normalizar_nome_coluna(col):
    """Padroniza nomes de colunas removendo ruídos e acentos."""
    c = str(col).upper().strip()
    c = "".join(ch for ch in unicodedata.normalize('NFD', c) if unicodedata.category(ch) != 'Mn')
    c = re.sub(r'[^A-Z0-9]', '', c)
    return c

def aplicar_regra_oc(row):
    """Unifica OC1 e OC2 evitando erros de ambiguidade do Pandas."""
    oc1 = limpeza_radical(row.get('OC1', ''))
    oc2 = limpeza_radical(row.get('OC2', ''))
    
    # Verificação explícita para evitar 'Ambiguous Series'
    tem_oc1 = len(oc1) > 0 and oc1.lower() != 'nan'
    tem_oc2 = len(oc2) > 0 and oc2.lower() != 'nan'

    if not tem_oc1 and not tem_oc2: 
        return ""
    
    if tem_oc1 and tem_oc2: 
        return f"{oc1} e {oc2}"
    
    res = oc1 if tem_oc1 else oc2
    return res if res.endswith('.') else f"{res}."

# --- INÍCIO DO FLUXO ---
caminho_base = selecionar_pasta()
if not caminho_base: exit()

base_dir = Path(caminho_base)
csv_input = base_dir / "levantamento_relatorio_sintese" / "relatorio_sintese_site.csv"
pasta_pdfs_originais = base_dir / "preprocessamento" / "relatoriosintese"
pasta_pdfs_simplificados = base_dir / "preprocessamento" / "relatoriosintesesimplificado"
pasta_csv_final = base_dir / "arquivosgerados" / "resultadofinal_relatoriosintese"

for p in [pasta_pdfs_originais, pasta_pdfs_simplificados, pasta_csv_final]:
    p.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()
df_lista = pd.read_csv(csv_input)

print(f" Processando {len(df_lista)} cursos...")

for index, row in df_lista.iterrows():
    if pd.isna(row['curso']) or str(row['paginaInicial_anexoIX']) == '-': continue
    
    nome_curso_limpo = str(row['curso']).replace(" ", "_").lower()
    nome_arquivo = "".join(c for c in unicodedata.normalize('NFD', nome_curso_limpo) if unicodedata.category(c) != 'Mn')
    
    pdf_ori = pasta_pdfs_originais / f"{nome_arquivo}_completo.pdf"
    pdf_simp = pasta_pdfs_simplificados / f"{nome_arquivo}_anexoIX.pdf"
    csv_final = pasta_csv_final / f"{nome_arquivo}_conteudo.csv"

    # 1. Download e 2. Recorte (Mantidos conforme lógica original)
    try:
        if not pdf_ori.exists():
            r = requests.get(row['Link do PDF'], timeout=60)
            with open(pdf_ori, 'wb') as f: f.write(r.content)
            
        if not pdf_simp.exists():
            p_ini, p_fim = int(row['paginaInicial_anexoIX']), int(row['paginaFinal_anexoIX'])
            doc_ori = fitz.open(str(pdf_ori))
            doc_simp = fitz.open()
            doc_simp.insert_pdf(doc_ori, from_page=p_ini-1, to_page=p_fim-1)
            doc_simp.save(str(pdf_simp))
            doc_ori.close(); doc_simp.close()
    except Exception as e:
        print(f" Erro nos arquivos de {row['curso']}: {e}"); continue

    # 3. Extração com Docling (Ajustada para robustez)
    if not csv_final.exists():
        print(f" Extraindo: {row['curso']}...")
        try:
            result = converter.convert(str(pdf_simp.absolute()))
            df_list = [table.export_to_dataframe(result.document) for table in result.document.tables]
            
            if not df_list: continue

            df_temp = pd.concat(df_list, ignore_index=True)

            # Normalização de Colunas
            mapa_colunas = {}
            for col in df_temp.columns:
                norm = normalizar_nome_coluna(col)
                if 'POSICAO' in norm: mapa_colunas[col] = 'POSIÇÃO'
                elif 'PERFIL' in norm: mapa_colunas[col] = 'PERFIL'
                elif 'COMPETENCIA' in norm: mapa_colunas[col] = 'COMPETÊNCIAS'
                elif 'OC1' in norm: mapa_colunas[col] = 'OC1'
                elif 'OC2' in norm: mapa_colunas[col] = 'OC2'
            
            df_temp = df_temp.rename(columns=mapa_colunas)

            # Garante colunas alvo
            colunas_alvo = ['POSIÇÃO', 'PERFIL', 'COMPETÊNCIAS', 'OC1', 'OC2']
            for c in colunas_alvo:
                if c not in df_temp.columns: df_temp[c] = ""

            df_temp = df_temp[colunas_alvo].copy()

            # Limpeza de cada célula (Tratando Series ambíguas)
            for col in df_temp.columns:
                df_temp[col] = df_temp[col].apply(limpeza_radical)

            # Remove linhas de cabeçalho fantasma
            df_temp = df_temp[~df_temp['POSIÇÃO'].str.contains('POSIÇÃO', case=False, na=False)]
            
            # Unificação OC
            df_temp['OC_unificado'] = df_temp.apply(aplicar_regra_oc, axis=1)

            df_temp.to_csv(csv_final, index=False, sep=';', encoding='utf-8-sig')
            print(f" Sucesso: {nome_arquivo}")

        except Exception as e:
            print(f" Erro Crítico em {row['curso']}: {e}")

print("\n FIM DO PROCESSAMENTO!")