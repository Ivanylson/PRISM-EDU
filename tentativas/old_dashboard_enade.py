import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =============================================================================
# 1. CONFIGURAÇÕES DA PÁGINA E DADOS
# =============================================================================
st.set_page_config(page_title="Dashboard ENADE IES", layout="wide", page_icon="📊")

# Caminhos de dados
DIRETORIO_ATUAL = Path(__file__).resolve().parent
# Usamos o arquivo RESULTADO consolidado como "Banco de Dados"
caminho_base_consolidada = DIRETORIO_ATUAL / 'arquivosgerados' / 'RESULTADOS' / 'analise_por_ies_curso_enade.csv'

@st.cache_data # Cache para carregar o arquivo uma única vez e economizar memória
def carregar_dados_resumidos():
    if not caminho_base_consolidada.exists():
        return None
    # Lemos apenas as colunas necessárias se o arquivo for muito grande
    df = pd.read_csv(caminho_base_consolidada, sep=';')
    return df

df_base = carregar_dados_resumidos()

# Títulos da Interface
st.title("📊 Painel Executivo de Acompanhamento ENADE")
st.markdown("---")

if df_base is None:
    st.error(f"ERRO: Base de dados consolidada não encontrada em: {caminho_base_consolidada}")
    st.markdown("Execute o script de análise principal primeiro para gerar o arquivo CSV.")
    st.stop()

# Mapeamento útil (pode ser expandido dinamicamente)
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

# =============================================================================
# 2. FILTROS INTERATIVOS (Barra Lateral)
# =============================================================================
with st.sidebar:
    st.header("Filtros de Pesquisa")
    
    # Filtro 1: IES (Código)
    ies_disponiveis = df_base['CO_IES'].dropna().unique()
    ies_disponiveis.sort()
    ies_selecionada = st.selectbox("Selecione a IES (Código):", ies_disponiveis)
    
    # Filtra dados apenas da IES selecionada para otimizar os filtros seguintes
    df_ies = df_base[df_base['CO_IES'] == ies_selecionada]
    
    # Filtro 2: Curso
    grupos_ies = df_ies['CO_GRUPO'].dropna().unique()
    
    # Mostra o nome do curso se estiver no mapa, senão mostra o código
    grupos_labels = {g: cursos_map.get(g, f"Curso {g}") for g in grupos_ies}
    
    grupo_selecionado = st.selectbox(
        "Selecione o Curso:", 
        grupos_ies, 
        format_func=lambda x: grupos_labels[x]
    )

    # DADOS DA IES: Apenas da faculdade selecionada
    df_final = df_ies[df_ies['CO_GRUPO'] == grupo_selecionado].copy()
    
    # DADOS NACIONAIS: Ignora a IES, pega todas as faculdades do Brasil para este curso
    df_curso_nacional = df_base[df_base['CO_GRUPO'] == grupo_selecionado].copy()

# =============================================================================
# 3. ÁREA PRINCIPAL DO DASHBOARD
# =============================================================================
nome_curso_final = cursos_map.get(grupo_selecionado, f"Curso {grupo_selecionado}")

if df_final.empty:
    st.warning(f"Não foram encontrados dados para a IES {ies_selecionada} no curso {nome_curso_final}.")
    st.stop()

# KPIs no topo
col1, col2, col3 = st.columns(3)
qtd_questoes = df_final['QUESTAO'].nunique() if 'QUESTAO' in df_final.columns else 0
media_deficiencia_ies = df_final['TAXA_DEFICIENCIA_%'].mean() if 'TAXA_DEFICIENCIA_%' in df_final.columns else 0
media_deficiencia_nacional = df_curso_nacional['TAXA_DEFICIENCIA_%'].mean() if 'TAXA_DEFICIENCIA_%' in df_curso_nacional.columns else 0

with col1:
    st.metric(label="Curso Analisado", value=nome_curso_final)
with col2:
    st.metric(
        label=f"Erro Geral - IES {ies_selecionada}", 
        value=f"{media_deficiencia_ies:.1f}%"
    )
with col3:
    st.metric(
        label="Erro Geral - Média Nacional", 
        value=f"{media_deficiencia_nacional:.1f}%",
        delta=f"{(media_deficiencia_ies - media_deficiencia_nacional):.1f}% vs Brasil",
        delta_color="inverse" # Se a IES erra mais (positivo), a cor fica vermelha
    )

st.markdown("---")

# -------------------------------------------------------------------------
# FUNÇÃO AUXILIAR: UNIR TODAS AS MATÉRIAS (OCs)
# -------------------------------------------------------------------------
colunas_oc = [col for col in df_final.columns if str(col).startswith('OC')]

def unir_ocs(linha):
    lista_ocs = []
    for col in colunas_oc:
        val = linha.get(col)
        if pd.notna(val) and str(val).strip() != "" and str(val).lower() != "nan":
            lista_ocs.append(str(val).strip())
    return " + ".join(lista_ocs) if lista_ocs else "Não especificado"

# -------------------------------------------------------------------------
# GRÁFICO 1: RANKING OC DA IES
# -------------------------------------------------------------------------
st.header("1. Ranking de Gargalos por Objeto de Conhecimento Principal (IES)")
st.markdown(f"Principais matérias onde a **IES {ies_selecionada}** apresenta deficiência consolidada.")

if 'OC1' in df_final.columns and 'TAXA_DEFICIENCIA_%' in df_final.columns:
    df_ranking_oc = df_final.groupby('OC1')['TAXA_DEFICIENCIA_%'].mean().reset_index()
    df_ranking_oc = df_ranking_oc.nlargest(10, 'TAXA_DEFICIENCIA_%')

    df_ranking_oc['OC1_curto'] = df_ranking_oc['OC1'].str.wrap(40)

    fig_oc = px.bar(
        df_ranking_oc,
        x='TAXA_DEFICIENCIA_%',
        y='OC1_curto',
        orientation='h',
        labels={'TAXA_DEFICIENCIA_%': 'Taxa de Erro (%)', 'OC1_curto': 'Objeto de Conhecimento'},
        color='TAXA_DEFICIENCIA_%',
        color_continuous_scale='Reds',
        title=f"Top 10 Gargalos - IES {ies_selecionada}"
    )
    fig_oc.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(fig_oc, use_container_width=True)

# -------------------------------------------------------------------------
# GRÁFICO 2 & TABELA: TOP QUESTÕES CRÍTICAS DA IES
# -------------------------------------------------------------------------
st.markdown("---")
st.header("2. Plano de Ação Local: Top 5 Questões Críticas da IES")
st.markdown(f"As 5 questões exatas da prova em que a **IES {ies_selecionada}** mais falhou.")

if 'QUESTAO' in df_final.columns and 'TAXA_DEFICIENCIA_%' in df_final.columns:
    df_top_questoes = df_final.nlargest(5, 'TAXA_DEFICIENCIA_%').copy()
    
    col_grafico, col_tabela = st.columns([1, 1.8])
    
    with col_grafico:
        fig_questoes = px.bar(
            df_top_questoes,
            x='QUESTAO',
            y='TAXA_DEFICIENCIA_%',
            text='TAXA_DEFICIENCIA_%',
            color='TAXA_DEFICIENCIA_%',
            color_continuous_scale='Reds',
            labels={'QUESTAO': 'Questão', 'TAXA_DEFICIENCIA_%': 'Erro (%)'}
        )
        fig_questoes.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_questoes.update_layout(yaxis_range=[0, 115], margin=dict(t=30, b=0, l=0, r=0)) 
        st.plotly_chart(fig_questoes, use_container_width=True)
        
    with col_tabela:
        df_top_questoes['TODAS AS MATÉRIAS (OCs)'] = df_top_questoes.apply(unir_ocs, axis=1)
        
        cols_tabela = ['QUESTAO', 'TAXA_DEFICIENCIA_%', 'TODAS AS MATÉRIAS (OCs)']
        if 'COMPETÊNCIAS' in df_final.columns:
            cols_tabela.append('COMPETÊNCIAS')
        elif 'COMPETENCIA' in df_final.columns:
            cols_tabela.append('COMPETENCIA')
            
        df_mostrar = df_top_questoes[cols_tabela].rename(columns={'TAXA_DEFICIENCIA_%': 'ERRO (%)'}).reset_index(drop=True)
        st.dataframe(df_mostrar, use_container_width=True)

# -------------------------------------------------------------------------
# NOVO: TÓPICO 3 - PANORAMA NACIONAL (TODAS AS IES JUNTAS)
# -------------------------------------------------------------------------
st.markdown("---")
st.header("3. Panorama Nacional: As 5 Maiores Deficiências do Curso no Brasil")
st.markdown(f"Agrupamento de **todas as instituições do país** que realizaram a prova de **{nome_curso_final}**. Serve para identificar se a deficiência da IES é um problema local ou um gargalo nacional do ensino.")

if not df_curso_nacional.empty and 'QUESTAO' in df_curso_nacional.columns:
    # Agrupa por questão calculando a média nacional de erro
    agg_dict = {'TAXA_DEFICIENCIA_%': 'mean'}
    
    # Mantém a primeira ocorrência das competências e OCs (pois são iguais para a mesma questão)
    for col in colunas_oc:
        agg_dict[col] = 'first'
    if 'COMPETÊNCIAS' in df_curso_nacional.columns:
        agg_dict['COMPETÊNCIAS'] = 'first'
    elif 'COMPETENCIA' in df_curso_nacional.columns:
        agg_dict['COMPETENCIA'] = 'first'
        
    df_nacional_agrupado = df_curso_nacional.groupby('QUESTAO').agg(agg_dict).reset_index()
    
    # Pega as 5 piores do Brasil
    df_top_nacional = df_nacional_agrupado.nlargest(5, 'TAXA_DEFICIENCIA_%').copy()
    
    col_graf_nac, col_tab_nac = st.columns([1, 1.8])
    
    with col_graf_nac:
        fig_nac = px.bar(
            df_top_nacional,
            x='QUESTAO',
            y='TAXA_DEFICIENCIA_%',
            text='TAXA_DEFICIENCIA_%',
            color='TAXA_DEFICIENCIA_%',
            color_continuous_scale='Oranges', # Usando laranja para diferenciar do gráfico da IES
            labels={'QUESTAO': 'Questão', 'TAXA_DEFICIENCIA_%': 'Erro Médio Brasil (%)'}
        )
        fig_nac.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_nac.update_layout(yaxis_range=[0, 115], margin=dict(t=30, b=0, l=0, r=0)) 
        st.plotly_chart(fig_nac, use_container_width=True)
        
    with col_tab_nac:
        df_top_nacional['TODAS AS MATÉRIAS (OCs)'] = df_top_nacional.apply(unir_ocs, axis=1)
        
        cols_tab_nac = ['QUESTAO', 'TAXA_DEFICIENCIA_%', 'TODAS AS MATÉRIAS (OCs)']
        if 'COMPETÊNCIAS' in df_top_nacional.columns:
            cols_tab_nac.append('COMPETÊNCIAS')
        elif 'COMPETENCIA' in df_top_nacional.columns:
            cols_tab_nac.append('COMPETENCIA')
            
        df_mostrar_nac = df_top_nacional[cols_tab_nac].rename(columns={'TAXA_DEFICIENCIA_%': 'ERRO NACIONAL (%)'}).reset_index(drop=True)
        st.dataframe(df_mostrar_nac, use_container_width=True)

st.markdown("---")
st.caption(f"Dados gerados automaticamente. Motor de IA em: {DIRETORIO_ATUAL}")

#streamlit run dashboard_enade.py 