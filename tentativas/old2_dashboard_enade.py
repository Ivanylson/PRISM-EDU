import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import unicodedata
import os

# =============================================================================
# 1. CONFIGURAÇÕES E FUNÇÕES ÚTEIS
# =============================================================================
st.set_page_config(page_title="Dashboard IA Educacional", layout="wide", page_icon="🎓")

def formatar_nome(nome):
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_')

# Caminhos de dados dinâmicos
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent 

PASTA_RESULTADOS = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS'
caminho_base_consolidada = PASTA_RESULTADOS / 'analise_por_ies_curso_enade.csv'
pasta_preditivos = PASTA_RESULTADOS / 'analisesPredetivos'
pasta_agrupamentos = PASTA_RESULTADOS / 'analisesAgrupamentos'
caminho_ia = PASTA_RESULTADOS / 'relatorio_diagnostico_pedagogico.csv' 

@st.cache_data
def carregar_dados_resumidos():
    if not caminho_base_consolidada.exists():
        return None
    return pd.read_csv(caminho_base_consolidada, sep=';')

df_base = carregar_dados_resumidos()

st.title("🎓 Painel Executivo de IA Educacional - ENADE")
st.markdown("---")

if df_base is None:
    st.error(f"❌ ERRO: Base de dados consolidada não encontrada em: {caminho_base_consolidada}")
    st.warning("Execute as Fases 1 e 2 do pipeline antes de abrir o Dashboard.")
    st.stop()

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
    ies_disponiveis = df_base['CO_IES'].dropna().unique()
    ies_disponiveis.sort()
    ies_selecionada = st.selectbox("Selecione a IES (Código):", ies_disponiveis)
    
    df_ies = df_base[df_base['CO_IES'] == ies_selecionada]
    grupos_ies = df_ies['CO_GRUPO'].dropna().unique()
    grupos_labels = {g: cursos_map.get(g, f"Curso {g}") for g in grupos_ies}
    
    grupo_selecionado = st.selectbox(
        "Selecione o Curso:", 
        grupos_ies, 
        format_func=lambda x: grupos_labels[x]
    )

    df_final = df_ies[df_ies['CO_GRUPO'] == grupo_selecionado].copy()
    df_curso_nacional = df_base[df_base['CO_GRUPO'] == grupo_selecionado].copy()

    # --- NOVO: BOTÃO DE SAIR ---
    st.markdown("---")
    if st.button("🚪 Sair e Fechar Dashboard", use_container_width=True):
        st.success("A encerrar o sistema... Pode fechar esta janela do navegador e voltar ao seu terminal.")
        os._exit(0) # Mata o processo do Streamlit, voltando ao menu_principal.py

nome_curso_final = cursos_map.get(grupo_selecionado, f"Curso {grupo_selecionado}")
nome_curso_arquivo = formatar_nome(nome_curso_final)

if df_final.empty:
    st.warning(f"Não foram encontrados dados para a IES {ies_selecionada} no curso {nome_curso_final}.")
    st.stop()

# =============================================================================
# 3. CRIAÇÃO DAS ABAS (TABS)
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Diagnóstico e Prescrição", 
    "🎯 Fatores de Sucesso (Preditivo)", 
    "🧪 Validação Científica (Algoritmos)",
    "🤖 Clustering Pedagógico e Plano de Ação"
])

# -----------------------------------------------------------------------------
# ABA 1: DIAGNÓSTICO
# -----------------------------------------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    media_deficiencia_ies = df_final['TAXA_DEFICIENCIA_%'].mean() if 'TAXA_DEFICIENCIA_%' in df_final.columns else 0
    media_deficiencia_nacional = df_curso_nacional['TAXA_DEFICIENCIA_%'].mean() if 'TAXA_DEFICIENCIA_%' in df_curso_nacional.columns else 0

    with col1: st.metric(label="Curso Analisado", value=nome_curso_final)
    with col2: st.metric(label=f"Erro Geral - IES {ies_selecionada}", value=f"{media_deficiencia_ies:.1f}%")
    with col3: st.metric(label="Erro Geral - Média Nacional", value=f"{media_deficiencia_nacional:.1f}%", delta=f"{(media_deficiencia_ies - media_deficiencia_nacional):.1f}% vs Brasil", delta_color="inverse")

    st.markdown("---")
    
    colunas_oc = [col for col in df_final.columns if str(col).startswith('OC')]
    def unir_ocs(linha):
        lista_ocs = [str(linha.get(col)).strip() for col in colunas_oc if pd.notna(linha.get(col)) and str(linha.get(col)).strip() != "" and str(linha.get(col)).lower() != "nan"]
        return " + ".join(lista_ocs) if lista_ocs else "Não especificado"

    st.subheader(f"📍 Plano de Ação Local: Top 5 Questões Críticas da IES {ies_selecionada}")
    if 'QUESTAO' in df_final.columns and 'TAXA_DEFICIENCIA_%' in df_final.columns:
        df_top_questoes = df_final.nlargest(5, 'TAXA_DEFICIENCIA_%').copy()
        c_graf, c_tab = st.columns([1, 1.8])
        with c_graf:
            fig_q = px.bar(df_top_questoes, x='QUESTAO', y='TAXA_DEFICIENCIA_%', text='TAXA_DEFICIENCIA_%', color='TAXA_DEFICIENCIA_%', color_continuous_scale='Reds', labels={'QUESTAO': 'Questão', 'TAXA_DEFICIENCIA_%': 'Erro (%)'})
            fig_q.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_q.update_layout(yaxis_range=[0, 115], margin=dict(t=10, b=0, l=0, r=0)) 
            st.plotly_chart(fig_q, use_container_width=True)
        with c_tab:
            df_top_questoes['MATÉRIAS (OCs)'] = df_top_questoes.apply(unir_ocs, axis=1)
            cols_t = ['QUESTAO', 'TAXA_DEFICIENCIA_%', 'MATÉRIAS (OCs)']
            if 'COMPETÊNCIAS' in df_final.columns: cols_t.append('COMPETÊNCIAS')
            df_mostrar = df_top_questoes[cols_t].rename(columns={'TAXA_DEFICIENCIA_%': 'ERRO (%)'}).reset_index(drop=True)
            st.dataframe(df_mostrar, use_container_width=True)

    st.markdown("---")
    st.subheader("🇧🇷 Panorama Nacional: Maiores Gargalos do Curso no Brasil")
    if not df_curso_nacional.empty and 'QUESTAO' in df_curso_nacional.columns:
        agg_dict = {'TAXA_DEFICIENCIA_%': 'mean'}
        for col in colunas_oc: agg_dict[col] = 'first'
        if 'COMPETÊNCIAS' in df_curso_nacional.columns: agg_dict['COMPETÊNCIAS'] = 'first'
        df_nac_agrup = df_curso_nacional.groupby('QUESTAO').agg(agg_dict).reset_index()
        df_top_nac = df_nac_agrup.nlargest(5, 'TAXA_DEFICIENCIA_%').copy()
        
        c_graf_nac, c_tab_nac = st.columns([1, 1.8])
        with c_graf_nac:
            fig_nac = px.bar(df_top_nac, x='QUESTAO', y='TAXA_DEFICIENCIA_%', text='TAXA_DEFICIENCIA_%', color='TAXA_DEFICIENCIA_%', color_continuous_scale='Oranges', labels={'QUESTAO': 'Questão', 'TAXA_DEFICIENCIA_%': 'Erro Médio Brasil (%)'})
            fig_nac.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_nac.update_layout(yaxis_range=[0, 115], margin=dict(t=10, b=0, l=0, r=0)) 
            st.plotly_chart(fig_nac, use_container_width=True)
        with c_tab_nac:
            df_top_nac['MATÉRIAS (OCs)'] = df_top_nac.apply(unir_ocs, axis=1)
            cols_tn = ['QUESTAO', 'TAXA_DEFICIENCIA_%', 'MATÉRIAS (OCs)']
            if 'COMPETÊNCIAS' in df_top_nac.columns: cols_tn.append('COMPETÊNCIAS')
            df_mostrar_nac = df_top_nac[cols_tn].rename(columns={'TAXA_DEFICIENCIA_%': 'ERRO NACIONAL (%)'}).reset_index(drop=True)
            st.dataframe(df_mostrar_nac, use_container_width=True)

# -----------------------------------------------------------------------------
# ABA 2: INTELIGÊNCIA PREDITIVA
# -----------------------------------------------------------------------------
with tab2:
    st.header(f"🧠 Fatores Determinantes de Sucesso em {nome_curso_final}")
    st.markdown("Esta seção utiliza Machine Learning (**XGBoost** e **Random Forest**) para descobrir quais competências determinam se um aluno será classificado como 'Alto Desempenho' a nível nacional.")
    
    caminho_pesos = pasta_preditivos / f"importancia_variaveis_{nome_curso_arquivo}.csv"
    caminho_metricas_pred = pasta_preditivos / f"metricas_modelos_{nome_curso_arquivo}.csv"
    
    if caminho_pesos.exists() and caminho_metricas_pred.exists():
        df_pesos = pd.read_csv(caminho_pesos, sep=';')
        df_metricas_pred = pd.read_csv(caminho_metricas_pred, sep=';')
        
        col_m1, col_m2 = st.columns([1, 2.5])
        
        with col_m1:
            st.markdown("#### 🏆 Batalha de Modelos")
            st.markdown("Qualidade da predição da IA:")
            st.dataframe(df_metricas_pred[['Modelo', 'Acuracia', 'F1_Score']].set_index('Modelo'), use_container_width=True)
            st.info("Quanto mais próximo de 1.0, mais exata é a IA em prever a nota final do aluno baseada nas matérias que ele domina.")
            
        with col_m2:
            st.markdown("#### 🚀 Top 10 Matérias que mais impactam a Nota (Feature Importance)")
            df_top_pesos = df_pesos.head(10).copy()
            df_top_pesos['OC_CURTO'] = df_top_pesos['TODOS_OS_OCs'].str.wrap(50)
            
            fig_pesos = px.bar(
                df_top_pesos, x='PESO_IMPORTANCIA_%', y='OC_CURTO', orientation='h',
                color='PESO_IMPORTANCIA_%', color_continuous_scale='Greens',
                labels={'PESO_IMPORTANCIA_%': 'Peso na Aprovação (%)', 'OC_CURTO': 'Matérias (OCs)'}
            )
            fig_pesos.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=0, b=0), height=350)
            st.plotly_chart(fig_pesos, use_container_width=True)
            
        st.markdown("#### 📋 Detalhamento das Competências-Chave")
        st.dataframe(df_pesos.head(10), use_container_width=True)
    else:
        st.warning("⚠️ Os arquivos de Inteligência Preditiva ainda não foram gerados para este curso. Execute a Fase 3 do pipeline.")

# -----------------------------------------------------------------------------
# ABA 3: VALIDAÇÃO CIENTÍFICA
# -----------------------------------------------------------------------------
with tab3:
    st.header(f"🔬 Rigor Científico dos Agrupamentos: {nome_curso_final}")
    st.markdown("Justificativa matemática da escolha do algoritmo (K-Means) e do número ideal de grupos de alunos.")
    
    caminho_img_val = pasta_agrupamentos / f"graficos_validacao_{nome_curso_arquivo}.png"
    caminho_metricas_agrup = pasta_agrupamentos / f"metricas_agrupamento_{nome_curso_arquivo}.csv"
    
    if caminho_img_val.exists() and caminho_metricas_agrup.exists():
        st.image(str(caminho_img_val), caption="Batalha de Algoritmos Não-Supervisionados (K-Means vs GMM vs Hierárquico)", use_column_width=True)
        
        st.markdown("#### 📊 Tabela de Métricas Internas")
        df_metricas_agrup = pd.read_csv(caminho_metricas_agrup, sep=';')
        st.dataframe(df_metricas_agrup.set_index('K_Grupos'), use_container_width=True)
        
        st.success("**Conclusão Acadêmica:** O modelo **K-Means** foi selecionado para os painéis principais por garantir alta coerência (Davies-Bouldin baixo) e bom isolamento de grupos (Silhouette aceitável), aliado ao menor custo computacional, permitindo a identificação clara do aluno mediano.")
    else:
        st.warning("⚠️ Os arquivos de Validação Científica ainda não foram gerados para este curso. Execute a Fase 3 do pipeline.")

# -----------------------------------------------------------------------------
# ABA 4: CLUSTERING E PLANO DE AÇÃO
# -----------------------------------------------------------------------------
with tab4:
    st.header(f"🤖 Clustering Pedagógico e Plano de Ação: {nome_curso_final}")
    st.markdown("""
    Nesta secção, o nosso motor de Inteligência Artificial analisou os padrões de acertos e erros dos alunos, 
    dividiu-os em **grupos de proficiência** e cruzou as deficiências com a Matriz Curricular do MEC.
    """)
    
    if caminho_ia.exists():
        df_ia = pd.read_csv(caminho_ia, sep=';', decimal=',', dtype={'IES': str})
        ies_filtro = str(int(ies_selecionada)) if isinstance(ies_selecionada, float) else str(ies_selecionada)
        
        df_ia_filtrado = df_ia[(df_ia['CURSO'] == nome_curso_final) & (df_ia['IES'] == ies_filtro)]
        
        if not df_ia_filtrado.empty:
            k_livre = df_ia_filtrado['K_MATEMATICO_LIVRE'].iloc[0]
            silhueta_livre = df_ia_filtrado['SILHUETA_LIVRE'].iloc[0]
            k_aplicado = df_ia_filtrado['K_APLICADO_PEDAGOGICO (Max 6)'].iloc[0]
            
            st.info("📊 **Justificativa do Algoritmo:**")
            col1, col2 = st.columns(2)
            col1.metric("K Matemático (Puro)", f"{k_livre} grupos", f"Silhueta: {silhueta_livre}")
            col2.metric("K Pedagógico (Aplicado)", f"{k_aplicado} grupos", "Adaptado para a rotina docente")
            
            st.divider()
            st.subheader(f"👥 Perfis e Plano de Intervenção ({k_aplicado} grupos na IES {ies_filtro})")
            
            for _, row in df_ia_filtrado.iterrows():
                if "Risco" in row['NOME_DO_GRUPO']:
                    cor_status = "🔴" 
                elif "Excelência" in row['NOME_DO_GRUPO'] or "Alto" in row['NOME_DO_GRUPO']:
                    cor_status = "🟢"
                else:
                    cor_status = "🟡"
                
                with st.expander(f"{cor_status} {row['NOME_DO_GRUPO']} — {row['QTD_ALUNOS_GRUPO']} alunos (Média do Grupo: {row['NOTA_MEDIA_GERAL_GRUPO']})", expanded=True):
                    
                    st.error(f"**📉 Piores Questões do Grupo (GAPs Nacionais):** `{row['QUESTOES_MAIS_IMPACTANTES']}`")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.warning("**📘 Objetos de Conhecimento a Revisar:**\n\n" + str(row['MATERIAS_DEFICIENTES']).replace(" | ", "\n\n"))
                    with c2:
                        st.info("**🎯 Competências a Desenvolver:**\n\n" + str(row['COMPETENCIAS_A_DESENVOLVER']).replace(" | ", "\n\n"))
        else:
            st.warning(f"Não há dados suficientes nesta IES ({ies_filtro}) para o algoritmo de IA realizar um agrupamento seguro.")
    else:
        st.error("⚠️ Ficheiro relatorio_diagnostico_pedagogico.csv não encontrado. Execute a Fase 4 do pipeline.")

st.markdown("---")
st.caption(f"Sistema desenvolvido com base nos microdados ENADE 2023. | Diretório de Dados: {DIRETORIO_RAIZ}")