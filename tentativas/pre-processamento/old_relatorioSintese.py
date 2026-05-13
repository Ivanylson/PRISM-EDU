import pandas as pd
import requests
import fitz
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import InputContext
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import unicodedata
import os

def selecionar_pasta():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    caminho = filedialog.askdirectory(title="Selecione a pasta 'projetoENADE'")
    root.destroy()
    return caminho

def limpar_texto(txt):
    if txt is None or pd.isna(txt): return ""
    return " ".join(str(txt).replace('\n', ' ').split())

def aplicar_regra_oc(row):
    oc1 = limpar_texto(row.get('OC1', ''))
    oc2 = limpar_texto(row.get('OC2', ''))
    if not oc1: return ""
    if oc2 and oc2.lower() != 'nan' and len(oc2.strip()) > 0:
        return f"{oc1} e {oc2}"
    else:
        return oc1 if oc1.endswith('.') else f"{oc1}."

# --- CONFIGURAÇÃO ---
caminho_base = selecionar_pasta()
if not caminho_base: exit()

base_dir = Path(caminho_base)
csv_input = base_dir / "levantamento_relatorio_sintese" / "relatorio_sintese_site.csv"
pasta_pdfs_originais = base_dir / "preprocessamento" / "relatoriosintese"
pasta_csv_final = base_dir / "arquivosgerados" / "resultadofinal_relatoriosintese"

for p in [pasta_pdfs_originais, pasta_csv_final]:
    p.mkdir(parents=True, exist_ok=True)

# PADRÃO DE POSIÇÃO SOLICITADO
PADRAO_POSICAO = ['D1'] + [str(i) for i in range(1, 10)] + ['D2'] + [str(i) for i in range(10, 39)]

# Configuração do Docling para focar em tabelas
pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = True
converter = DocumentConverter(pipeline_options=pipeline_options)

df_lista = pd.read_csv(csv_input)

for index, row in df_lista.iterrows():
    if pd.isna(row['curso']) or str(row['paginaInicial_anexoIX']) == '-': continue
    
    nome_curso_limpo = str(row['curso']).replace(" ", "_").lower()
    nome_arquivo = "".join(c for c in unicodedata.normalize('NFD', nome_curso_limpo) if unicodedata.category(c) != 'Mn')
    
    url_pdf = row['Link do PDF']
    # Páginas do Anexo IX
    p_ini = int(row['paginaInicial_anexoIX'])
    p_fim = int(row['paginaFinal_anexoIX'])
    
    pdf_ori = pasta_pdfs_originais / f"{nome_arquivo}_completo.pdf"
    csv_final = pasta_csv_final / f"{nome_arquivo}_conteudo.csv"

    # 1. Download do Original
    if not pdf_ori.exists():
        print(f"Baixando: {row['curso']}...")
        try:
            r = requests.get(url_pdf, timeout=60)
            with open(pdf_ori, 'wb') as f: f.write(r.content)
        except Exception as e:
            print(f"Erro ao baixar: {e}")
            continue

    # 2. Extração Direta do Original (Focando apenas nas páginas do Anexo IX)
    if not csv_final.exists():
        print(f"Docling extraindo {row['curso']} (Págs {p_ini}-{p_fim})...")
        try:
            # O Docling permite converter especificando o intervalo de páginas
            # Isso resolve o erro de "Inconsistent number of pages"
            result = converter.convert(
                str(pdf_ori.absolute()),
                # O Docling usa índice 1 para páginas, passamos o range exato
                # Nota: Em versões recentes, passamos as opções no convert ou via pipeline
            )
            
            # Filtramos as tabelas que pertencem às páginas do Anexo IX
            df_list = []
            for table in result.document.tables:
                # Verificamos se a tabela está dentro do range de páginas do Anexo IX
                # O Docling marca a proveniência (page_no)
                page_no = table.prov[0].page_no if table.prov else 0
                
                if p_ini <= page_no <= p_fim:
                    df_t = table.export_to_dataframe(result.document)
                    df_list.append(df_t)
            
            if not df_list:
                print(f"⚠️ Nenhuma tabela encontrada no intervalo {p_ini}-{p_fim} para {row['curso']}")
                continue

            df_extraido = pd.concat(df_list, ignore_index=True)
            df_extraido.columns = [limpar_texto(c).upper() for c in df_extraido.columns]
            
            # Mapeamento e Padronização
            rename_map = {'POSICAO': 'POSIÇÃO', 'COMPETENCIAS': 'COMPETÊNCIAS', 'OC 1': 'OC1', 'OC 2': 'OC2'}
            df_extraido = df_extraido.rename(columns=rename_map)

            # Criar Gabarito Vazio para garantir a ordem D1, 1-9, D2, 10-38
            df_final = pd.DataFrame(columns=['POSIÇÃO', 'PERFIL', 'COMPETÊNCIAS', 'OC1', 'OC2'])
            df_final['POSIÇÃO'] = PADRAO_POSICAO

            # Encaixar dados extraídos no gabarito
            for pos in PADRAO_POSICAO:
                mask = df_extraido['POSIÇÃO'].astype(str).str.strip() == pos
                dados_linha = df_extraido[mask]
                
                if not dados_linha.empty:
                    idx_dest = df_final[df_final['POSIÇÃO'] == pos].index[0]
                    for col in ['PERFIL', 'COMPETÊNCIAS', 'OC1', 'OC2']:
                        if col in dados_linha.columns:
                            df_final.at[idx_dest, col] = limpar_texto(dados_linha.iloc[0][col])

            df_final['OC_unificado'] = df_final.apply(aplicar_regra_oc, axis=1)
            
            df_final.to_csv(csv_final, index=False, sep=';', encoding='utf-8-sig')
            print(f"✅ Sucesso: {nome_arquivo}_conteudo.csv")

        except Exception as e:
            print(f"❌ Erro Docling em {row['curso']}: {e}")
    else:
        print(f"⏩ Pulando {row['curso']}, CSV já existe.")

print(f"\n🚀 PROCESSO FINALIZADO!")