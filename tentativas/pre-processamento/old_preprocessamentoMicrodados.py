import pandas as pd
import requests
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import unicodedata
import re
import csv

def selecionar_pasta():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    caminho = filedialog.askdirectory(title="Selecione a pasta 'projetoENADE'")
    root.destroy()
    return caminho

# --- ALGORITMO LÉXICO E DE LIMPEZA ---
def lexer_blindado(valor):
    """
    Analisa e limpa o conteúdo para garantir integridade no CSV:
    1. Converte listas em strings.
    2. Remove quebras de linha.
    3. Substitui ';' por '.' para evitar quebra de colunas.
    4. Remove aspas duplas internas para evitar erro de encapsulamento.
    """
    if pd.isna(valor) or valor is None: return ""
    if isinstance(valor, list): valor = " ".join(map(str, valor))
    
    t = str(valor).replace('\n', ' ').replace('\r', ' ')
    t = t.replace(';', '.')  # Troca separador interno por ponto
    t = t.replace('"', "'")  # Troca aspas duplas por simples
    
    return " ".join(t.split()).strip()

def extrair_tokens_posicao(texto):
    """Extrai IDs de posição (D1, D2 ou números isolados)."""
    t = lexer_blindado(texto).upper()
    if 'D1' in t: return ['D1']
    if 'D2' in t: return ['D2']
    return re.findall(r'\b\d+\b', t)

def normalizar_cabecalho(col):
    """Normaliza nomes de colunas via análise de tokens."""
    c = "".join(ch for ch in unicodedata.normalize('NFD', str(col).upper()) if unicodedata.category(ch) != 'Mn')
    c = re.sub(r'[^A-Z0-9]', '', c)
    if 'POSICAO' in c: return 'POSIÇÃO'
    if 'PERFIL' in c: return 'PERFIL'
    if 'COMPETENCIA' in c: return 'COMPETÊNCIAS'
    if 'OC1' in c: return 'OC1'
    if 'OC2' in c: return 'OC2'
    return c

# --- CONFIGURAÇÃO ---
caminho_base = selecionar_pasta()
if not caminho_base: exit()

base_dir = Path(caminho_base)
csv_input = base_dir / "levantamento_relatorio_sintese" / "relatorio_sintese_site.csv"
pasta_pdfs_originais = base_dir / "preprocessamento" / "relatoriosintese"
pasta_pdfs_simplificados = base_dir / "preprocessamento" / "relatoriosintesesimplificado"
pasta_csv_final = base_dir / "arquivosgerados" / "resultadofinal_relatoriosintese"

for p in [pasta_pdfs_originais, pasta_pdfs_simplificados, pasta_csv_final]:
    p.mkdir(parents=True, exist_ok=True)

# Molde fixo de 40 posições (D1, 1-9, D2, 10-38)
GABARITO_FIXO = ['D1'] + [str(i) for i in range(1, 10)] + ['D2'] + [str(i) for i in range(10, 39)]

converter = DocumentConverter()
df_lista = pd.read_csv(csv_input)

print(f"Iniciando processamento de {len(df_lista)} cursos...")

for index, row in df_lista.iterrows():
    if pd.isna(row['curso']) or str(row['paginaInicial_anexoIX']) == '-': continue
    
    nome_curso_limpo = str(row['curso']).replace(" ", "_").lower()
    nome_arquivo = "".join(c for c in unicodedata.normalize('NFD', nome_curso_limpo) if unicodedata.category(c) != 'Mn')
    
    pdf_ori = pasta_pdfs_originais / f"{nome_arquivo}_completo.pdf"
    pdf_simp = pasta_pdfs_simplificados / f"{nome_arquivo}_anexoIX.pdf"
    csv_final = pasta_csv_final / f"{nome_arquivo}_conteudo.csv"

    # 1. Download e 2. Simplificação (Mantidos do seu molde)
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
    except: continue

    # 3. EXTRAÇÃO E MAPEAMENTO LÉXICO
    if not csv_final.exists():
        print(f"extraindo: {row['curso']}...")
        try:
            result = converter.convert(str(pdf_simp.absolute()))
            
            # Criar esqueleto vazio para as 40 posições
            df_res = pd.DataFrame(index=GABARITO_FIXO, columns=['PERFIL', 'COMPETÊNCIAS', 'OC1', 'OC2'])
            
            # Processar tabelas encontradas pelo Docling
            for table in result.document.tables:
                df_tab = table.export_to_dataframe(result.document)
                df_tab.columns = [normalizar_cabecalho(c) for c in df_tab.columns]
                
                for _, l_pdf in df_tab.iterrows():
                    tokens = extrair_tokens_posicao(l_pdf.get('POSIÇÃO', ''))
                    for t in tokens:
                        if t in df_res.index:
                            df_res.at[t, 'PERFIL'] = lexer_blindado(l_pdf.get('PERFIL', ''))
                            df_res.at[t, 'COMPETÊNCIAS'] = lexer_blindado(l_pdf.get('COMPETÊNCIAS', ''))
                            df_res.at[t, 'OC1'] = lexer_blindado(l_pdf.get('OC1', ''))
                            df_res.at[t, 'OC2'] = lexer_blindado(l_pdf.get('OC2', ''))

            # Algoritmo de Unificação OC (Lexa)
            def aplicar_regra_oc(r):
                oc1, oc2 = str(r['OC1']), str(r['OC2'])
                v1 = oc1 if oc1 and oc1.lower() != 'nan' else ""
                v2 = oc2 if oc2 and oc2.lower() != 'nan' else ""
                if v1 and v2: return f"{v1} e {v2}"
                if v1: return v1 if v1.endswith('.') else f"{v1}."
                if v2: return v2 if v2.endswith('.') else f"{v2}."
                return ""

            df_res['OC_unificado'] = df_res.apply(aplicar_regra_oc, axis=1)
            df_res.index.name = 'POSIÇÃO'
            df_res = df_res.reset_index()

            # Salvar com Blindagem (Aspas Duplas e QUOTE_ALL)
            df_res.to_csv(
                csv_final, 
                index=False, 
                sep=';', 
                encoding='utf-8-sig',
                quoting=csv.QUOTE_ALL
            )
            print(f"Gerado: {nome_arquivo}_conteudo.csv")

        except Exception as e:
            print(f"Erro Crítico em {row['curso']}: {e}")

print("\n🚀 PROCESSO FINALIZADO COM SUCESSO!")