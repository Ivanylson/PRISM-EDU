import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import unicodedata
import os
import subprocess
import sys
import json
import tempfile
import requests
import socket
import time

# =============================================================================
# 1. CONFIGURAÇÕES E FUNÇÕES ÚTEIS
# =============================================================================
st.set_page_config(page_title="Dashboard IA Educacional + LCA + OpenCode", layout="wide", page_icon="🎓")

def formatar_nome(nome):
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_')

def limitar_lista_texto(texto, separador, limite):
    if pd.isna(texto) or str(texto).strip() == "" or str(texto).lower() == "nan":
        return "Nenhum dado ou não aplicável"
    itens = [i.strip() for i in str(texto).split(separador) if i.strip()]
    return f" {separador} ".join(itens[:limite])

def is_port_in_use(port):
    """Verifica se a porta do servidor web já está sendo usada."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def iniciar_servidor_opencode_background():
    """Inicia o OpenCode Web automaticamente nos bastidores, se não estiver rodando."""
    if not is_port_in_use(4096):
        try:
            # Tenta rodar de forma invisível
            if os.name == 'nt':  # Windows
                subprocess.Popen(["opencode", "web", "--port", "4096"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else: # Linux/Mac
                subprocess.Popen(["opencode", "web", "--port", "4096"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3) # Aguarda 3 segundos para o servidor subir
            return True
        except Exception as e:
            st.error(f"Não foi possível iniciar o OpenCode automaticamente: {e}")
            return False
    return True

# Caminhos
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent
PASTA_RESULTADOS = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS'
PASTA_LCA = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_LCA'
PASTA_LCA_FIGURAS = DIRETORIO_RAIZ / 'arquivosgerados' / 'FIGURAS_LCA'

caminho_base_consolidada = PASTA_RESULTADOS / 'analise_por_ies_curso_enade.csv'
pasta_preditivos = PASTA_RESULTADOS / 'analisesPredetivos'
pasta_agrupamentos = PASTA_RESULTADOS / 'analisesAgrupamentos'
caminho_ia = PASTA_RESULTADOS / 'relatorio_diagnostico_pedagogico.csv'

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

cursos_reverso = {v: k for k, v in cursos_map.items()}

@st.cache_data
def carregar_dados_resumidos():
    if not caminho_base_consolidada.exists():
        return None
    return pd.read_csv(caminho_base_consolidada, sep=';')

@st.cache_data
def carregar_lca_geral():
    path = PASTA_LCA / '01_relatorio_geral_por_ies.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=';', decimal=',', dtype={'ies': str})
    df['ies'] = df['ies'].str.strip()
    return df

@st.cache_data
def carregar_lca_classes():
    path = PASTA_LCA / '02_caracterizacao_das_classes.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=';', decimal=',', dtype={'ies': str})
    df['ies'] = df['ies'].str.strip()
    return df

@st.cache_data
def carregar_lca_criterios():
    path = PASTA_LCA / '03_criterios_de_selecao_k.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=';', decimal=',', dtype={'ies': str})
    df['ies'] = df['ies'].str.strip()
    return df

df_base = carregar_dados_resumidos()
df_lca_geral = carregar_lca_geral()
df_lca_classes = carregar_lca_classes()
df_lca_crit = carregar_lca_criterios()

st.title("🎓 Painel de IA Educacional - ENADE 2023")
st.markdown("**K-Means + LCA + OpenCode** — Análise de Perfis de Aprendizagem com Explicação em Linguagem Natural")
st.markdown("---")

# =============================================================================
# 2. SIDEBAR - FILTROS GLOBAIS (sempre visíveis)
# =============================================================================
ies_selecionada = None
grupo_selecionado = None
nome_curso_final = "Nenhum"
nome_curso_arquivo = ""
df_final = pd.DataFrame()
df_curso_nacional = pd.DataFrame()

with st.sidebar:
    st.header("Filtros de Pesquisa")

    km_disponivel = df_base is not None and not df_base.empty
    
    if km_disponivel:
        ies_disponiveis = df_base['CO_IES'].dropna().unique()
        ies_disponiveis.sort()
        ies_selecionada = st.selectbox("IES (código):", ies_disponiveis, key="ies_km")
        
        df_ies = df_base[df_base['CO_IES'] == ies_selecionada]
        grupos_ies = df_ies['CO_GRUPO'].dropna().unique()
        grupos_labels = {g: cursos_map.get(g, f"Curso {g}") for g in grupos_ies}
        
        grupo_selecionado = st.selectbox("Curso:", grupos_ies,
            format_func=lambda x: grupos_labels[x], key="curso_km")
            
        nome_curso_final = cursos_map.get(grupo_selecionado, f"Curso {grupo_selecionado}")
        nome_curso_arquivo = formatar_nome(nome_curso_final)
        df_final = df_ies[df_ies['CO_GRUPO'] == grupo_selecionado].copy()
        df_curso_nacional = df_base[df_base['CO_GRUPO'] == grupo_selecionado].copy()

    st.markdown("---")
    if st.button("Sair e Fechar", use_container_width=True):
        st.success("A encerrar... Pode fechar a janela.")
        os._exit(0)

# =============================================================================
# 3. NAVEGAÇÃO PRINCIPAL (radio horizontal substitui st.tabs)
# =============================================================================
TABS = [
    "Diagnóstico e Prescrição",
    "Fatores de Sucesso (Preditivo)",
    "Validação + Clustering",
    "Plano de Ação por Perfil",
    "LCA - Classes Latentes",
    "OpenCode + IA Explicativa"
]

tab_default = 0
if "--tab" in sys.argv:
    idx = sys.argv.index("--tab")
    if idx + 1 < len(sys.argv):
        arg = sys.argv[idx + 1].lower()
        if arg == "opencode":
            tab_default = 5

tab_selector = st.radio("Navegação:", TABS, index=tab_default, horizontal=True, label_visibility="collapsed")

# =============================================================================
# ABA 1: DIAGNÓSTICO
# =============================================================================
if tab_selector == TABS[0]:
    if df_base is None or df_base.empty:
        st.warning("Base de dados K-Means não encontrada. Execute a Fase 1 e 2 primeiro.")
    elif df_final.empty:
        st.warning("Selecione uma IES e Curso no filtro lateral.")
    else:
        col1, col2, col3 = st.columns(3)
        media_deficiencia_ies = df_final['TAXA_DEFICIENCIA_%'].mean() if 'TAXA_DEFICIENCIA_%' in df_final.columns else 0
        media_deficiencia_nacional = df_curso_nacional['TAXA_DEFICIENCIA_%'].mean() if 'TAXA_DEFICIENCIA_%' in df_curso_nacional.columns else 0
        with col1: st.metric("Curso", nome_curso_final)
        with col2: st.metric(f"Erro IES {ies_selecionada}", f"{media_deficiencia_ies:.1f}%")
        with col3: st.metric("Erro Nacional", f"{media_deficiencia_nacional:.1f}%",
            delta=f"{(media_deficiencia_ies - media_deficiencia_nacional):.1f}% vs Brasil", delta_color="inverse")
        st.markdown("---")

        colunas_oc = [col for col in df_final.columns if str(col).startswith('OC')]
        def unir_ocs(linha):
            lista_ocs = [str(linha.get(col)).strip() for col in colunas_oc
                if pd.notna(linha.get(col)) and str(linha.get(col)).strip() != "" and str(linha.get(col)).lower() != "nan"]
            return " + ".join(lista_ocs) if lista_ocs else "Não especificado"

        st.subheader(f" Top 5 Questões Críticas - IES {ies_selecionada}")
        if 'QUESTAO' in df_final.columns and 'TAXA_DEFICIENCIA_%' in df_final.columns:
            agg_dict_ies = {'TAXA_DEFICIENCIA_%': 'mean'}
            for col in colunas_oc: agg_dict_ies[col] = 'first'
            if 'COMPETÊNCIAS' in df_final.columns: agg_dict_ies['COMPETÊNCIAS'] = 'first'
            df_ies_agrup = df_final.groupby('QUESTAO').agg(agg_dict_ies).reset_index()
            df_top_questoes = df_ies_agrup.nlargest(5, 'TAXA_DEFICIENCIA_%').copy()

            c_graf, c_tab = st.columns([1, 1.8])
            with c_graf:
                fig_q = px.bar(df_top_questoes, x='QUESTAO', y='TAXA_DEFICIENCIA_%',
                    text='TAXA_DEFICIENCIA_%', color='TAXA_DEFICIENCIA_%',
                    color_continuous_scale='Reds')
                fig_q.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_q.update_layout(yaxis_range=[0, 115])
                st.plotly_chart(fig_q, use_container_width=True)
            with c_tab:
                df_top_questoes['MATÉRIAS (OCs)'] = df_top_questoes.apply(unir_ocs, axis=1)
                cols_t = ['QUESTAO', 'TAXA_DEFICIENCIA_%', 'MATÉRIAS (OCs)']
                if 'COMPETÊNCIAS' in df_final.columns: cols_t.append('COMPETÊNCIAS')
                st.dataframe(df_top_questoes[cols_t].rename(
                    columns={'TAXA_DEFICIENCIA_%': 'ERRO (%)'}).reset_index(drop=True),
                    use_container_width=True)

        st.subheader("🇧🇷 Panorama Nacional")
        if not df_curso_nacional.empty and 'QUESTAO' in df_curso_nacional.columns:
            agg_dict = {'TAXA_DEFICIENCIA_%': 'mean'}
            for col in colunas_oc: agg_dict[col] = 'first'
            if 'COMPETÊNCIAS' in df_curso_nacional.columns: agg_dict['COMPETÊNCIAS'] = 'first'
            df_nac_agrup = df_curso_nacional.groupby('QUESTAO').agg(agg_dict).reset_index()
            df_top_nac = df_nac_agrup.nlargest(5, 'TAXA_DEFICIENCIA_%').copy()
            c_graf_nac, c_tab_nac = st.columns([1, 1.8])
            with c_graf_nac:
                fig_nac = px.bar(df_top_nac, x='QUESTAO', y='TAXA_DEFICIENCIA_%',
                    text='TAXA_DEFICIENCIA_%', color='TAXA_DEFICIENCIA_%', color_continuous_scale='Oranges')
                fig_nac.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_nac.update_layout(yaxis_range=[0, 115])
                st.plotly_chart(fig_nac, use_container_width=True)
            with c_tab_nac:
                df_top_nac['MATÉRIAS (OCs)'] = df_top_nac.apply(unir_ocs, axis=1)
                cols_tn = ['QUESTAO', 'TAXA_DEFICIENCIA_%', 'MATÉRIAS (OCs)']
                if 'COMPETÊNCIAS' in df_top_nac.columns: cols_tn.append('COMPETÊNCIAS')
                st.dataframe(df_top_nac[cols_tn].rename(
                    columns={'TAXA_DEFICIENCIA_%': 'ERRO NACIONAL (%)'}).reset_index(drop=True),
                    use_container_width=True)

# =============================================================================
# ABA 2: PREDITIVO
# =============================================================================
elif tab_selector == TABS[1]:
    if df_base is None or df_base.empty:
        st.warning("Base não encontrada.")
    elif nome_curso_arquivo == "":
        st.warning("Selecione IES/Curso no filtro lateral.")
    else:
        caminho_pesos = pasta_preditivos / f"importancia_variaveis_{nome_curso_arquivo}.csv"
        caminho_metricas_pred = pasta_preditivos / f"metricas_modelos_{nome_curso_arquivo}.csv"
        if caminho_pesos.exists() and caminho_metricas_pred.exists():
            df_pesos = pd.read_csv(caminho_pesos, sep=';')
            df_metricas_pred = pd.read_csv(caminho_metricas_pred, sep=';')
            col_m1, col_m2 = st.columns([1, 2.5])
            with col_m1:
                st.markdown("#### Qualidade da Predição")
                st.dataframe(df_metricas_pred[['Modelo', 'Acuracia', 'F1_Score']].set_index('Modelo'), use_container_width=True)
            with col_m2:
                st.markdown("#### Top 10 Matérias que mais impactam")
                df_top_pesos = df_pesos.head(10).copy()
                df_top_pesos['OC_CURTO'] = df_top_pesos['TODOS_OS_OCs'].str.wrap(50)
                fig_pesos = px.bar(df_top_pesos, x='PESO_IMPORTANCIA_%', y='OC_CURTO',
                    orientation='h', color='PESO_IMPORTANCIA_%', color_continuous_scale='Greens')
                fig_pesos.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
                st.plotly_chart(fig_pesos, use_container_width=True)
        else:
            st.warning("Arquivos Preditivos não encontrados.")

# =============================================================================
# ABA 3: VALIDAÇÃO CIENTÍFICA
# =============================================================================
elif tab_selector == TABS[2]:
    if df_base is None or df_base.empty:
        st.warning("Base não encontrada.")
    elif nome_curso_arquivo == "":
        st.warning("Selecione IES/Curso no filtro lateral.")
    else:
        st.header("Validação dos Agrupamentos (K-Means)")
        caminho_img_val = pasta_agrupamentos / f"graficos_validacao_{nome_curso_arquivo}.png"
        caminho_metricas_agrup = pasta_agrupamentos / f"metricas_agrupamento_{nome_curso_arquivo}.csv"
        if caminho_img_val.exists():
            st.image(str(caminho_img_val), use_container_width=True)
        if caminho_metricas_agrup.exists():
            df_metricas_agrup = pd.read_csv(caminho_metricas_agrup, sep=';')
            st.dataframe(df_metricas_agrup.set_index('K_Grupos'), use_container_width=True)
        if not caminho_img_val.exists() and not caminho_metricas_agrup.exists():
            st.warning("Arquivos de validação não encontrados.")

# =============================================================================
# ABA 4: CLUSTERING + PLANO DE AÇÃO
# =============================================================================
elif tab_selector == TABS[3]:
    if df_base is None or df_base.empty:
        st.warning("Base não encontrada.")
    elif nome_curso_final == "Nenhum":
        st.warning("Selecione IES/Curso no filtro lateral.")
    else:
        col_titulo, col_param = st.columns([2, 1])
        with col_titulo:
            st.header(f"Plano de Ação: {nome_curso_final}")
        with col_param:
            limite_top_n = st.slider("Top N problemas:", 1, 38, 5, key="top_n_tab4")

        with st.expander("Explicação dos Algoritmos e Metodologia"):
            st.markdown("""
            ### K-Means + Gap Analysis + Interseção de Falhas

            O motor de IA usa **K-Means** para agrupar alunos por similaridade de respostas.
            A métrica de **GAP** compara cada grupo contra a média da IES. A **Interseção**
            identifica falhas institucionais que afetam todos os perfis simultaneamente.
            """)

        if caminho_ia.exists():
            df_ia = pd.read_csv(caminho_ia, sep=';', decimal=',', dtype={'IES': str})
            ies_filtro = str(int(ies_selecionada)) if isinstance(ies_selecionada, float) else str(ies_selecionada)
            df_ia_filtrado = df_ia[(df_ia['CURSO'] == nome_curso_final) & (df_ia['IES'] == ies_filtro)]
            if not df_ia_filtrado.empty:
                k_livre = df_ia_filtrado.get('K_MATEMATICO_LIVRE', pd.Series(['N/A'])).iloc[0]
                silhueta = df_ia_filtrado.get('SILHUETA_LIVRE', pd.Series(['N/A'])).iloc[0]
                k_aplicado = df_ia_filtrado.get('K_APLICADO_PEDAGOGICO (Max 6)', pd.Series([0])).iloc[0]
                st.info(f"K-Livre: {k_livre} (Silhueta: {silhueta}) | K-Aplicado: {k_aplicado}")

                if 'FALHA_SISTEMICA_IES' in df_ia_filtrado.columns:
                    falha = df_ia_filtrado['FALHA_SISTEMICA_IES'].iloc[0]
                    if pd.notna(falha) and str(falha).strip() not in ["Nenhuma", "nan", ""]:
                        st.error(f"Falha Institucional (interseção em todos os perfis): {falha}")

                for _, row in df_ia_filtrado.iterrows():
                    nome_grupo = str(row.get('NOME_DO_GRUPO', 'Grupo'))
                    qtd = row.get('QTD_ALUNOS_GRUPO', 0)
                    media = row.get('NOTA_MEDIA_GERAL_GRUPO', 0)
                    if "Excelência" in nome_grupo or "Alto" in nome_grupo:
                        cor = "🟢"
                    elif "Risco" in nome_grupo:
                        cor = "🔴"
                    else:
                        cor = "🟡"
                    with st.expander(f"{cor} {nome_grupo} — {qtd} alunos | Média: {media}", expanded=True):
                        qs = limitar_lista_texto(row.get('QUESTOES_MAIS_IMPACTANTES', ''), ',', limite_top_n)
                        mats = limitar_lista_texto(row.get('MATERIAS_DEFICIENTES', ''), '|', limite_top_n)
                        comps = limitar_lista_texto(row.get('COMPETENCIAS_A_DESENVOLVER', ''), '|', limite_top_n)
                        st.markdown(f"**Questões:** {qs}")
                        c1, c2 = st.columns(2)
                        c1.warning(f"**Disciplinas:**\n" + mats.replace(" | ", "\n"))
                        c2.info(f"**Competências:**\n" + comps.replace(" | ", "\n"))
            else:
                st.warning("Sem dados de cluster para esta IES/Curso.")
        else:
            st.warning("Arquivo de diagnóstico não encontrado.")

# =============================================================================
# ABA 5: LCA - CLASSES LATENTES
# =============================================================================
elif tab_selector == TABS[4]:
    st.header("Análise de Classes Latentes (LCA)")
    st.markdown("Identificação de perfis de aprendizagem via **Latent Class Analysis** — abordagem estatística robusta para dados categóricos binários.")

    if df_lca_geral is None:
        st.error(f"Arquivos LCA não encontrados em: {PASTA_LCA}")
        st.info("Execute o pipeline: `python 0_pipeline_principal_LCA.py`")
    else:
        cursos_lca = sorted(df_lca_geral['curso'].dropna().unique())
        if not cursos_lca:
            st.warning("Nenhum curso encontrado nos dados LCA.")
        else:
            curso_default = cursos_lca[0]
            ies_default = None
            if km_disponivel and grupo_selecionado is not None:
                nome_curso_sidebar = cursos_map.get(grupo_selecionado)
                if nome_curso_sidebar and nome_curso_sidebar in cursos_lca:
                    curso_default = nome_curso_sidebar
                    ies_sidebar = str(int(float(ies_selecionada)))
                    if ies_sidebar in df_lca_geral[df_lca_geral['curso'] == curso_default]['ies'].values:
                        ies_default = ies_sidebar

            col1, col2 = st.columns(2)
            with col1:
                curso_idx = cursos_lca.index(curso_default)
                curso_lca_sel = st.selectbox("Curso (LCA):", cursos_lca, index=curso_idx, key="lca_curso")
            with col2:
                df_lca_curso_filtro = df_lca_geral[df_lca_geral['curso'] == curso_lca_sel]
                ies_lca_list = sorted(df_lca_curso_filtro['ies'].dropna().unique())
                ies_idx = ies_lca_list.index(ies_default) if ies_default and ies_default in ies_lca_list else 0
                ies_lca_sel = st.selectbox("IES (código):", ies_lca_list, index=ies_idx, key="lca_ies")

            if ies_default:
                st.caption(f"Sincronizado com o filtro global: {nome_curso_final} / IES {ies_selecionada}")

            st.markdown("---")

            df_lca_ies = df_lca_curso_filtro[df_lca_curso_filtro['ies'] == ies_lca_sel]
            df_cls_ies = df_lca_classes[(df_lca_classes['curso'] == curso_lca_sel) &
                                        (df_lca_classes['ies'] == ies_lca_sel)].copy()

            if not df_lca_ies.empty:
                row = df_lca_ies.iloc[0]
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                col_m1.metric("k (classes)", str(int(row['k_escolhido'])))
                col_m2.metric("Entropia", f"{row['entropia_normalizada']:.3f}")
                col_m3.metric("Estabilidade (Jaccard)", f"{row['jaccard_medio_bootstrap']:.3f}")
                col_m4.metric("ARI (LCA vs K-Means)", f"{row['ari_lca_vs_kmeans_mca']:.3f}")
                col_m5.metric("N alunos", str(int(row['n_alunos'])))

                with st.expander("Justificativa da seleção de k"):
                    st.info(row['justificativa_k'])

                st.markdown("---")

                if not df_cls_ies.empty:
                    st.subheader("Perfis Identificados pela LCA")

                    prob_cols = [f'PROB_Q{i}' for i in range(1, 39)]
                    gap_cols = [f'GAP_Q{i}' for i in range(1, 39)]
                    q_labels = [f'Q{i}' for i in range(1, 39)]

                    fig = go.Figure()
                    for _, cls_row in df_cls_ies.sort_values('taxa_acerto_media', ascending=False).iterrows():
                        probs = [cls_row.get(c, 0) for c in prob_cols]
                        classe_nome = f"Classe {int(cls_row['classe_id'])} ({int(cls_row['n_alunos'])} alunos)"
                        fig.add_trace(go.Scatter(x=q_labels, y=probs, mode='lines+markers', name=classe_nome))

                    fig.update_layout(title="Probabilidade de Acerto por Questão", yaxis_range=[0, 1.05], height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Detalhamento das Classes")
                    cols_show = ['classe_id', 'n_alunos', 'taxa_acerto_media', 'itens_diferencialmente_fortes', 'itens_diferencialmente_fracos']
                    cols_show = [c for c in cols_show if c in df_cls_ies.columns]
                    st.dataframe(df_cls_ies[cols_show], use_container_width=True)

                st.markdown("---")
                st.subheader("Figuras da Análise (Artigo SBIE)")
                imagens = list(PASTA_LCA_FIGURAS.glob("fig*.png")) + list(PASTA_LCA_FIGURAS.glob("*.png"))
                if imagens:
                    cols_img = st.columns(3)
                    for i, img_p in enumerate(imagens[:6]):
                        with cols_img[i % 3]:
                            st.image(str(img_p), use_container_width=True, caption=img_p.stem)
            else:
                st.warning("Nenhum resultado LCA encontrado para esta IES/Curso.")

# =============================================================================
# ABA 6: OPENCODE + IA EXPLICATIVA (AUTOMÁTICA E SEM TERMINAL)
# =============================================================================
elif tab_selector == TABS[5]:
    st.header("OpenCode + IA Explicativa (Integração Web Automática)")
    st.markdown("""
    O painel se conectará **automaticamente** ao OpenCode Web para analisar os dados e produzir
    uma explicação em linguagem natural sem a necessidade de comandos manuais no terminal.
    """)

    st.subheader("1. Selecione o CSV para análise")

    csvs_disponiveis = []
    if PASTA_RESULTADOS.exists():
        csvs_disponiveis.extend(list(PASTA_RESULTADOS.rglob("*.csv")))
    if PASTA_LCA.exists():
        csvs_disponiveis.extend(list(PASTA_LCA.glob("*.csv")))

    csvs_validos = [p for p in csvs_disponiveis if p.stat().st_size <= 50 * 1024 * 1024]

    if not csvs_validos:
        st.error("Nenhum arquivo CSV encontrado nas pastas de resultados.")
    else:
        csv_labels = {str(p.relative_to(p.parents[2]) if len(p.parents) > 2 else p.name): p for p in sorted(set(csvs_validos))}
        labels_ordenados = sorted(csv_labels.keys())

        csv_default_idx = 0
        if km_disponivel and nome_curso_arquivo != "":
            curso_csv = nome_curso_arquivo.lower().replace('_', ' ')
            ies_str = str(int(float(ies_selecionada)))
            for i, label in enumerate(labels_ordenados):
                if ies_str in label and curso_csv in label.lower():
                    csv_default_idx = i
                    break

        csv_selecionado = st.selectbox("Arquivo CSV:", labels_ordenados, index=csv_default_idx)
        caminho_csv = csv_labels[csv_selecionado]

        st.subheader("2. Prévia dos Dados")
        try:
            df_csv_preview = pd.read_csv(caminho_csv, sep=';', nrows=10, encoding='utf-8-sig', on_bad_lines='skip')
            st.dataframe(df_csv_preview, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")
            st.stop()

        st.subheader("3. Personalize a Análise")
        col_t, col_f = st.columns([1, 2])
        with col_t:
            tom = st.selectbox("Tom da explicação:", ["Didático", "Técnico", "Resumo executivo", "Crítico"])
        with col_f:
            foco = st.text_input("Foco adicional (opcional):", placeholder="Ex: 'Destaque as principais deficiências'")

        st.subheader("4. Gerar Explicação")
        if st.button("Gerar Explicação com IA", type="primary", use_container_width=True):
            
            with st.status("Iniciando IA Explicativa...", expanded=True) as status:
                try:
                    # Preparando dados resumidos para a IA
                    df_full = pd.read_csv(caminho_csv, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
                    colunas_num = [c for c in df_full.columns if pd.api.types.is_numeric_dtype(df_full[c])]
                    
                    resumo_estatistico = {}
                    for c in colunas_num[:10]:
                        resumo_estatistico[c] = {"media": round(float(df_full[c].mean()), 2), "max": round(float(df_full[c].max()), 2)}

                    prompt = f"""
                    Analise os dados educacionais do arquivo {caminho_csv.name}.
                    TOM: {tom}. FOCO: {foco}.
                    ESTATÍSTICAS: {json.dumps(resumo_estatistico, ensure_ascii=False)}
                    AMOSTRA (10 linhas): {df_full.head(10).to_string()}
                    Dê recomendações e formate em Markdown.
                    """

                    # Passo automático: Verificar/Ligar o servidor
                    status.update(label="Verificando servidor OpenCode (Automático)...", state="running")
                    iniciar_servidor_opencode_background()
                    
                    URL_OPENCODE = "http://127.0.0.1:4096"
                    
                    # Conectar à API Web
                    status.update(label="Criando sessão na IA...", state="running")
                    res_sess = requests.post(f"{URL_OPENCODE}/session", json={}, timeout=15)
                    res_sess.raise_for_status()
                    session_id = res_sess.json().get("id") or res_sess.json().get("data", {}).get("id")
                    
                    status.update(label="Analisando os dados (Isso pode levar até 2 minutos)...", state="running")
                    payload = {"parts": [{"type": "text", "text": prompt}]}
                    res_msg = requests.post(f"{URL_OPENCODE}/session/{session_id}/message", json=payload, timeout=200)
                    res_msg.raise_for_status()
                    
                    # Extraindo resposta
                    partes = res_msg.json().get("parts", [])
                    resposta_final = "".join([p.get("text", "") for p in partes if p.get("type") == "text"])
                    if not resposta_final:
                        resposta_final = str(res_msg.json())

                    # Exibindo resultado
                    status.update(label="Explicação gerada com sucesso!", state="complete")
                    st.success("Análise Finalizada!")
                    st.markdown("### Parecer da Inteligência Artificial")
                    st.markdown("---")
                    st.write(resposta_final)

                except requests.exceptions.ConnectionError:
                    status.update(label="Erro de Conexão", state="error")
                    st.error("Não foi possível conectar ao OpenCode automaticamente.")
                    st.warning("Verifique se o OpenCode está instalado globalmente (comando `opencode`).")
                except Exception as e:
                    status.update(label="Erro no Processamento", state="error")
                    st.error(f"Ocorreu um erro: {e}")
                    
                    st.markdown("### Análise Estatística Automática (Fallback)")
                    for col, stats in resumo_estatistico.items():
                        st.markdown(f"- **{col}**: média={stats['media']}, max={stats['max']}")