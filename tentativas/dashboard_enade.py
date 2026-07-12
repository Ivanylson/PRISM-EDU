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

def limitar_lista_texto(texto, separador, limite):
    """Fatia textos grandes (ex: Q1, Q2, Q3) consoante o Top N escolhido pelo utilizador."""
    if pd.isna(texto) or str(texto).strip() == "" or str(texto).lower() == "nan": 
        return "Nenhum dado ou não aplicável"
    itens = [i.strip() for i in str(texto).split(separador) if i.strip()]
    return f" {separador} ".join(itens[:limite])

# Caminhos de dados
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent 
PASTA_RESULTADOS = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS'

caminho_base_consolidada = PASTA_RESULTADOS / 'analise_por_ies_curso_enade.csv'
pasta_preditivos = PASTA_RESULTADOS / 'analisesPredetivos'
pasta_agrupamentos = PASTA_RESULTADOS / 'analisesAgrupamentos'
# Certifique-se de que este ficheiro contém as colunas geradas pelo Motor Triplo e cruzadas com as OCs
caminho_ia = PASTA_RESULTADOS / 'relatorio_diagnostico_pedagogico.csv' 

@st.cache_data
def carregar_dados_resumidos():
    if not caminho_base_consolidada.exists(): return None
    return pd.read_csv(caminho_base_consolidada, sep=';')

df_base = carregar_dados_resumidos()

st.title("Painel Executivo de IA Educacional - ENADE")
st.markdown("---")

if df_base is None:
    st.error(f" ERRO: Base de dados consolidada não encontrada em: {caminho_base_consolidada}")
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

    st.markdown("---")
    if st.button("Sair e Fechar Dashboard", use_container_width=True):
        st.success("A encerrar o sistema... Pode fechar esta janela e voltar ao terminal.")
        os._exit(0)

nome_curso_final = cursos_map.get(grupo_selecionado, f"Curso {grupo_selecionado}")
nome_curso_arquivo = formatar_nome(nome_curso_final)

if df_final.empty:
    st.warning("Não foram encontrados dados para este curso nesta IES.")
    st.stop()

# =============================================================================
# 3. CRIAÇÃO DAS ABAS 
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Diagnóstico e Prescrição", 
    "Fatores de Sucesso (Preditivo)", 
    "Validação Científica (Algoritmos)",
    "Clustering Pedagógico e Plano de Ação"
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
        
        # --- CORREÇÃO AQUI: Agrupar por QUESTAO para evitar duplicatas antes de pegar o Top 5 ---
        agg_dict_ies = {'TAXA_DEFICIENCIA_%': 'mean'}
        for col in colunas_oc: agg_dict_ies[col] = 'first'
        if 'COMPETÊNCIAS' in df_final.columns: agg_dict_ies['COMPETÊNCIAS'] = 'first'
        
        # Agrupando para garantir 1 linha por questão na IES
        df_ies_agrup = df_final.groupby('QUESTAO').agg(agg_dict_ies).reset_index()
        df_top_questoes = df_ies_agrup.nlargest(5, 'TAXA_DEFICIENCIA_%').copy()
        # ---------------------------------------------------------------------------------------
        
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
    st.header(f"Fatores Determinantes de Sucesso")
    caminho_pesos = pasta_preditivos / f"importancia_variaveis_{nome_curso_arquivo}.csv"
    caminho_metricas_pred = pasta_preditivos / f"metricas_modelos_{nome_curso_arquivo}.csv"
    
    if caminho_pesos.exists() and caminho_metricas_pred.exists():
        df_pesos = pd.read_csv(caminho_pesos, sep=';')
        df_metricas_pred = pd.read_csv(caminho_metricas_pred, sep=';')
        
        col_m1, col_m2 = st.columns([1, 2.5])
        with col_m1:
            st.markdown("#### 🏆 Qualidade da Predição")
            st.dataframe(df_metricas_pred[['Modelo', 'Acuracia', 'F1_Score']].set_index('Modelo'), use_container_width=True)
        with col_m2:
            st.markdown("#### 🚀 Top 10 Matérias que mais impactam a Nota")
            df_top_pesos = df_pesos.head(10).copy()
            df_top_pesos['OC_CURTO'] = df_top_pesos['TODOS_OS_OCs'].str.wrap(50)
            fig_pesos = px.bar(df_top_pesos, x='PESO_IMPORTANCIA_%', y='OC_CURTO', orientation='h', color='PESO_IMPORTANCIA_%', color_continuous_scale='Greens')
            fig_pesos.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=0, b=0), height=350)
            st.plotly_chart(fig_pesos, use_container_width=True)
    else:
        st.warning("Arquivos Preditivos não encontrados.")

# -----------------------------------------------------------------------------
# ABA 3: VALIDAÇÃO CIENTÍFICA
# -----------------------------------------------------------------------------
with tab3:
    st.header(f"🔬 Rigor Científico dos Agrupamentos")
    caminho_img_val = pasta_agrupamentos / f"graficos_validacao_{nome_curso_arquivo}.png"
    caminho_metricas_agrup = pasta_agrupamentos / f"metricas_agrupamento_{nome_curso_arquivo}.csv"
    
    if caminho_img_val.exists() and caminho_metricas_agrup.exists():
        st.image(str(caminho_img_val), use_container_width=True)
        df_metricas_agrup = pd.read_csv(caminho_metricas_agrup, sep=';')
        st.dataframe(df_metricas_agrup.set_index('K_Grupos'), use_container_width=True)
    else:
        st.warning("Arquivos de Validação não encontrados.")
# -----------------------------------------------------------------------------
# ABA 4: CLUSTERING IA, ANÁLISE DE K E PLANOS DE AÇÃO
# -----------------------------------------------------------------------------
with tab4:
    col_titulo, col_param = st.columns([2, 1])
    with col_titulo:
        st.header(f"Clustering Pedagógico e Plano de Ação: {nome_curso_final}")
    with col_param:
        limite_top_n = st.slider(
            "Quantidade de problemas a exibir (Top N):", 
            min_value=1, max_value=38, value=5
        )
    
    st.markdown("""
    Neste painel é mostrado o  motor de Inteligência Artificial que analisou os padrões de acertos e erros dos alunos, 
    dividiu-os em **grupos de proficiência** e cruzou as deficiências com os relatórios de síntese do ENADE.
    """)
    
    # --- EXPLICAÇÃO COMPLETA: ALGORITMOS, GRUPOS E INTERSECÇÃO ---

    with st.expander("Resumo Analítico: Os Algoritmos, os Perfis e a Intersecção"):
        st.markdown(r"""
        ### Entendendo o Motor por Trás da Aba 4: O Guia Definitivo e Matemático
        
      Segue abaixo toda a  engenharia, a matemática e a lógica pedagógica dos códigos que geram esses relatórios.
        
        ---
        ### 1. A Interface e o Balizamento (Top N)
         
        * **O Balizamento (Slider Top N):** Como uma disciplina pode apresentar dezenas de lacunas, a barra atua como um regulador dinâmico. Ele permite parametrizar a visualização, cortando as listas extensas de falhas e exibindo apenas os $N$ problemas mais urgentes escolhidos pelo utilizador (por exemplo, os 5 principais). Isto garante o foco no que é estritamente prioritário.   
       
        ---
        ### 2. O Motor de Agrupamento (Algoritmo K-Means) e a Matemática
        
        A IA **não divide os alunos pelas notas finais, ela agrupa os alunos**. Ela utiliza o algoritmo *K-Means*  para criar uma "impressão digital" matemática de cada aluno (uma matriz binária das questões que acertou (1) ou errou (0)) para todas as 38 questões da prova.. 
        
        O K-Means não divide os dados aleatoriamente. Ele agrupa as informações com base na **semelhança matemática** entre elas. O "correlacionado" aqui significa que um aluno pertence a um grupo porque ele está geograficamente (em um espaço de dados multidimensional) mais perto do centro daquele grupo do que de qualquer outro. Assim, os alunos são agrupados não pela sua nota final, mas sim porque partilham as mesmas dificuldades estruturais (erraram exatamente as mesmas questões).
        
        Para descobrir a qual grupo um aluno pertence, o algoritmo mede a distância $d$ entre o aluno ($p$) e o centroide do grupo ($q$) usando a **Distância Euclidiana**:
        """)
        
        st.latex(r"d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}")
        
        st.markdown(r"""
        **Legenda das Variáveis:**
        * $p$: É o vetor que representa as respostas do aluno analisado.
        * $q$: É o vetor "centroide", ou seja, o perfil de respostas ideal do centro do grupo.
        * $n$: É o número total de dimensões analisadas (neste caso, as 38 questões do exame).
        * $p_i$ e $q_i$: Representam a resposta específica na questão $i$ (0 ou 1).
        
        """)
        
        st.markdown(r"""
        **O algoritmo faz isso em ciclos (iterações):**
        1. Distribui "pontos centrais" (centroides) aleatórios.
        2. Mede a distância de todos os alunos para esses centros.
        3. Agrupa o aluno ao centro mais próximo.
        4. Recalcula a posição do centro tirando a média exata de todos os itens que caíram ali.
        5. Repete até estabilizar (Inércia mínima).
        
        A Inércia (o quão compactos e precisos são os grupos) é minimizada segundo a fórmula:
        """)
        
        st.latex(r"\text{Inércia} = \sum_{j=1}^{k} \sum_{x_i \in C_j} ||x_i - \mu_j||^2")

        st.markdown(r"""
        **Legenda das Variáveis:**
        * $k$: O número total de grupos formados.
        * $C_j$: Representa o agrupamento (cluster) específico a ser avaliado.
        * $x_i$: Um aluno individual inserido no grupo $C_j$.
        * $\mu_j$: A média do cluster $C_j$ (o ponto matemático central que define o grupo).
        * $||x_i - \mu_j||^2$: A distância quadrática de cada aluno até ao centro do seu próprio grupo.
        
        """)
        
        st.markdown(r"""
        ---
        
        ### 3. A Validação da Divisão e o Tamanho dos Grupos
        
        Para saber exatamente em quantos grupos a turma deve ser dividida (o valor de $k$, variando de 2 a 6), a máquina testa várias hipóteses e calcula o *Silhouette Score*  para cada aluno $i$:
        """)
        
        st.latex(r"s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}")
        
        st.markdown(r"""
        Onde $a(i)$ é a coesão interna do grupo e $b(i)$ é a separação do grupo vizinho. O algoritmo escolhe o $k$ com a maior pontuação, provando matematicamente que os grupos são consistentes.
        
        **Por que há uma variação de quantidade (tamanho) em cada grupo?**
        O K-Means não tem a obrigação de dividir a turma em partes iguais. Ele foca na **similaridade**:
        * **O Grupo Maior (A Massa):** Representa a média da operação. É natural que 60% a 70% da turma se comporte de maneira parecida.
        * **Os Grupos Menores (Os Extremos):** Representam nichos ou anomalias. Pode ser um pequeno grupo brilhante ou um grupo minúsculo com déficit severo. Como o comportamento deles é extremo, a IA os isola em clusters menores para não contaminar a média geral.
        
        **A Trava de Segurança:** Existe a regra `MIN_ALUNOS_POR_GRUPO = 5`. Se a IA tentar criar 4 grupos, mas um deles ficar com apenas 3 alunos, o algoritmo descarta essa divisão e tenta outra.
        
        ---
        
        ### 4. O Cálculo da Nota (Scoring) e Ordenação Matemática (Ranking)
        
        Depois da IA separar os alunos pelo "Código de Barras" (Imagine que o gabarito do aluno seja transformado numa sequência de números, onde 1 é um acerto e 0 é um erro. O exame de um aluno teria esta aparência: [1, 0, 1, 1, 0, 0, 1...]. Isso parece, literalmente, um código de barras.) das respostas, ela precisa descobrir quem é quem no nível de desempenho.
        * **A Soma Absoluta:** A prova tem 38 questões. O motor soma os acertos brutos de cada aluno (`sum(axis=1)`).
        * **A Média:** Em seguida, calcula a média de acertos de cada grupo recém-formado.
        
        O sistema executa uma função de ordenação (`sort(reverse=True)`), colocando estas médias brutas do maior para o menor valor. Por exemplo, se dividiu em 3 grupos:
        * **1º Lugar (Média 28 acertos):** Fica no topo do ranking.
        * **2º Lugar (Média 20 acertos):** Fica no meio.
        * **3º Lugar (Média 14 acertos):** Fica na base.
        
        *(Observação: A ordenação utiliza a nota absoluta em vez da porcentagem para garantir eficiência e estabilidade, evitando divisões percentuais flutuantes durante o K-Means. A porcentagem é calculada depois, apenas para visualização).*
        
        ---
        
        ### 5. Por que não usar uma percentagem fixa (ex: 80% para Excelência)?
        
        Se parametrizássemos o sistema para dizer que "Excelência é só quem acerta mais de 30 questões", poderíamos ter cursos onde **nenhum** aluno atingiria esse valor. O painel ficaria vazio.
        
        Como o cálculo é um **Ranking Relativo Posicional**, ele mostra a realidade daquela instituição:
        * Numa faculdade de elite, o grupo de "Risco Crítico" pode ter uma média de 22 acertos.
        * Numa faculdade com muitas dificuldades, o grupo de "Alto Desempenho" pode ter uma média de apenas 18 acertos.
        
        O objetivo não é dar diploma de gênio, mas dizer ao coordenador: *"Dentro da sua realidade atual, com os alunos que o senhor tem hoje, este é o seu grupo mais forte, este está na média, e este está a afundar a nota geral no ENADE."*
        
        ---
        
        ### 6. O Que Representa Cada Grupo e a Nomenclatura Dinâmica
        
        Após o K-Means fechar os grupos e o ranking ser estabelecido, o sistema atribui os nomes aos rótulos. Como a máquina decide sozinha quantos grupos (K) a turma vai ter, os rótulos adaptam-se automaticamente a esse "pódio":
        
        * **Se a IA criar 2 grupos (K=2 - Turma muito polarizada):**
        * 1º Lugar: "Alto Desempenho"
        * 2º Lugar: "Risco Crítico"
        * **Se a IA criar 3 grupos (K=3):**
        * 1º Lugar: "Alto Desempenho"
        * 2º Lugar: "Intermediário"
        * 3º Lugar: "Risco Crítico"
        * **Se a IA criar 4 grupos (K=4):**
        * 1º Lugar: "Excelência"
        * 2º Lugar: "Intermediário Superior"
        * 3º Lugar: "Intermediário Inferior"
        * 4º Lugar: "Risco Crítico"
        * **Se a IA criar 6 grupos (K=6):**
        * 1º Lugar: "Excelência"
        * 2º Lugar: "Alto Desempenho"
        * 3º Lugar: "Intermediário Superior"
        * 4º Lugar: "Intermediário Inferior"
        * 5º Lugar: "Atenção"
        * 6º Lugar: "Risco Crítico"
        
        **O Perfil de Cada Rótulo:**
        * **🟢 Excelência / Alto Desempenho:** É o grupo com a maior média. Representa alunos com forte domínio das competências do curso, permitindo identificar onde até os melhores alunos estão a falhar.
        * **🟡 Desempenho Intermediário:** Grupos que estão na média. Possuem lacunas específicas que impedem a transição para a excelência.
        * **🔴 Risco Crítico / Atenção:** É o grupo com a menor média. Representa alunos com défices severos de base que necessitam de intervenção pedagógica urgente.
        
        ---
        
        ### 7. O GAP, Intersecção (Falha Sistêmica) e o Mapeador de Deficiências
        
        Para identificar o que o grupo precisa estudar, usamos o **GAP** (Média do Grupo na Questão - Média da IES na Questão). Se o GAP for negativo, aquele cluster foi **pior que a média da própria faculdade** naquela questão.
        
        **A Falha Sistêmica (Intersecção de Problemas):**
        Se a IA encontrar uma questão com GAP negativo **simultaneamente em TODOS os grupos** (do Risco Crítico à Excelência), ela decreta uma Intersecção usando a teoria de conjuntos (`set.intersection`). Isso significa que a causa do erro não é o perfil do aluno, mas sim uma anomalia na matriz curricular da instituição, falha na didática do professor ou no próprio enunciado da prova.
        
        **O Tradutor de Mapeamento:**
        Mas como verificamos se a culpa é da instituição e não do aluno? Através do cruzamento matemático dos conjuntos.
        O código usa a teoria da intersecção: `set.intersection(*questoes_ruins_por_grupo)`. Se a questão [Q14] for o ponto de colisão e apresentar GAP negativo **simultaneamente em TODOS os grupos** (do Risco Crítico à Excelência), decreta-se a Intersecção. O facto de uma falha sobreviver ao corte dos alunos de alto desempenho prova algoritmicamente que o erro reside na lecionação, no enunciado ou na matriz do curso.
        Dizer que o aluno errou a "Q14" não ajuda na prática. O tradutor cruza essa informação com as matrizes sintéticas em CSV e extrai os **Objetos de Conhecimento (OC)** e as **Competências**, entregando o plano de ação pronto e legível no Dashboard.
       """)
    st.divider()

    if caminho_ia.exists():
        df_ia = pd.read_csv(caminho_ia, sep=';', decimal=',', dtype={'IES': str})
        ies_filtro = str(int(ies_selecionada)) if isinstance(ies_selecionada, float) else str(ies_selecionada)
        
        df_ia_filtrado = df_ia[(df_ia['CURSO'] == nome_curso_final) & (df_ia['IES'] == ies_filtro)]
        
        if not df_ia_filtrado.empty:
            
            # --- MÉTRICAS DE JUSTIFICATIVA DO ALGORITMO ---
            k_livre = df_ia_filtrado.get('K_MATEMATICO_LIVRE', pd.Series(['N/A'])).iloc[0]
            silhueta_livre = df_ia_filtrado.get('SILHUETA_LIVRE', pd.Series(['N/A'])).iloc[0]
            k_aplicado = df_ia_filtrado.get('K_APLICADO_PEDAGOGICO (Max 6)', pd.Series([len(df_ia_filtrado)])).iloc[0]
            
            st.info("**Justificativa do Algoritmo:**")
            col1, col2 = st.columns(2)
            col1.metric("K Matemático (Puro)", f"{k_livre} grupos", f"Silhueta: {silhueta_livre}")
            col2.metric("K Pedagógico (Aplicado)", f"{k_aplicado} grupos", "Adaptado para a rotina docente")
            
            st.divider()
            
            # --- ALERTA DE INTERSECÇÃO (FALHA SISTÉMICA) ---
            if 'FALHA_SISTEMICA_IES' in df_ia_filtrado.columns:
                falha = df_ia_filtrado['FALHA_SISTEMICA_IES'].iloc[0]
                # Verifica se não é nulo e não é strings vazias ou indicativas de ausência
                if pd.notna(falha) and str(falha).strip() not in ["Nenhuma", "nan", ""]:
                    falha_cortada = limitar_lista_texto(str(falha), ',', limite_top_n)
                    st.error(f"""
                    **ALERTA DE FALHA INSTITUCIONAL (INTERSECÇÃO COMPROVADA EM TODOS OS PERFIS)**
                    As questões **[{falha_cortada}]** foram detetadas em todos os grupos estruturados pela IA (desde o Risco Crítico até à Excelência) com desempenhos negativos.
                    
                    *Diagnóstico:* Esta intersecção confirma que o problema não deriva do perfil cognitivo dos alunos. Recomenda-se auditoria urgente à matriz curricular, à abordagem didática desta disciplina ou ao alinhamento com os critérios do MEC.
                    """)
            
            # --- CABEÇALHO DOS PERFIS ---
            st.subheader(f"Perfis e Plano de Intervenção ({k_aplicado} grupos na IES {ies_filtro})")
            
            # --- EXIBIÇÃO DOS GRUPOS E GAPS ---
            for _, row in df_ia_filtrado.iterrows():
                nome_grupo = str(row.get('NOME_DO_GRUPO', 'Grupo Sem Nome'))
                qtd_alunos = row.get('QTD_ALUNOS_GRUPO', 0)
                media_grupo = row.get('NOTA_MEDIA_GERAL_GRUPO', 0)
                
                # Definir as cores com base no rótulo
                if "Excelência" in nome_grupo or "Alto" in nome_grupo:
                    cor_status = "🟢"
                elif "Risco" in nome_grupo or "Baixo" in nome_grupo:
                    cor_status = "🔴"
                else:
                    cor_status = "🟡"
                
                titulo_expander = f"{cor_status} {nome_grupo} — Cluster com {qtd_alunos} alunos | Média de Acertos: {media_grupo}"
                
                with st.expander(titulo_expander, expanded=True):
                    
                    questoes_cortadas = limitar_lista_texto(row.get('QUESTOES_MAIS_IMPACTANTES', ''), ',', limite_top_n)
                    materias_cortadas = limitar_lista_texto(row.get('MATERIAS_DEFICIENTES', 'Dados pendentes'), '|', limite_top_n)
                    competencias_cortadas = limitar_lista_texto(row.get('COMPETENCIAS_A_DESENVOLVER', 'Dados pendentes'), '|', limite_top_n)

                    st.markdown(f"** Principais GAPs Negativos do Cluster (Top {limite_top_n} Questões Críticas):** `{questoes_cortadas}`")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.warning(f"**📘 Disciplinas (OCs) Recomendadas para Revisão Direcionada:**\n\n" + materias_cortadas.replace(" | ", "\n\n"))
                    with c2:
                        st.info(f"**🎯 Competências a Desenvolver nestes Alunos:**\n\n" + competencias_cortadas.replace(" | ", "\n\n"))
        else:
            st.warning(f"Sem dados suficientes na IES ({ies_filtro}) para gerar a análise de clusters para o curso selecionado.")
    else:
        st.error(f"⚠️ O ficheiro de diagnósticos da IA não foi encontrado. Execute o script do K-Means primeiro.")