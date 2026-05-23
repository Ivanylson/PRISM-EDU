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
import time
import urllib.request
import atexit
import requests
import base64
import logging
import threading

# =============================================================================
# 1. CONFIGURAÇÕES E FUNÇÕES ÚTEIS
# =============================================================================
st.set_page_config(page_title="PRISM-EDU Dashboard IA Educacional + LCA + OpenCode", layout="wide")

def formatar_nome(nome):
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_')

def limitar_lista_texto(texto, separador, limite):
    if pd.isna(texto) or str(texto).strip() == "" or str(texto).lower() == "nan":
        return "Nenhum dado ou não aplicável"
    itens = [i.strip() for i in str(texto).split(separador) if i.strip()]
    return f" {separador} ".join(itens[:limite])

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

# =============================================================================
# GERENCIAMENTO DO SERVIDOR OPENCODE
# =============================================================================

OPENCODE_SERVER_URL = "http://localhost:4096"
OPENCODE_USERNAME = "opencode"

def _get_logger():
    """Get a logger that doesn't duplicate output on Streamlit reruns."""
    logger = logging.getLogger("opencode_integration")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = _get_logger()

def procurar_opencode():
    """Procura o executável do OpenCode no PATH e locais comuns."""
    try:
        result = subprocess.run(["opencode", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return "opencode"
    except:
        pass
    for name in ["opencode-cli.exe", "opencode.exe", "opencode.cmd"]:
        try:
            result = subprocess.run(["where", name], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0].strip()
        except:
            pass
    localappdata = os.environ.get("LOCALAPPDATA", "")
    candidatos = [
        Path(localappdata) / "opencode" / "opencode-cli.exe",
        Path(localappdata) / "opencode" / "opencode.exe",
        DIRETORIO_RAIZ / "opencode.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "opencode" / "opencode.exe",
    ]
    for c in candidatos:
        if c.exists():
            return str(c)
    return None


def _senha_efetiva():
    """Retorna a senha fornecida pelo usuario na sidebar (se houver)."""
    return st.session_state.get("senha_opencode", "").strip()


def _auth_headers():
    """Retorna headers de autenticação Basic Auth, se senha foi fornecida."""
    senha = _senha_efetiva()
    if senha:
        credentials = f"{OPENCODE_USERNAME}:{senha}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {encoded}"}
    return {}


def servidor_opencode_ativo():
    """Verifica se o servidor OpenCode está rodando em http://localhost:4096.
    Tenta sem autenticação primeiro (servidor local), depois com senha se fornecida."""
    try:
        resp = requests.get(f"{OPENCODE_SERVER_URL}/global/health", timeout=3)
        if resp.status_code == 200:
            return True
    except:
        pass
    # Fallback: tenta com autenticação
    headers = _auth_headers()
    if headers:
        try:
            resp = requests.get(f"{OPENCODE_SERVER_URL}/global/health", headers=headers, timeout=3)
            return resp.status_code == 200
        except:
            pass
    return False


def iniciar_servidor_opencode():
    """Inicia opencode serve --port 4096 em background e aguarda ficar pronto."""
    # Se já estiver rodando, retorna True
    if servidor_opencode_ativo():
        logger.info("Servidor OpenCode já está ativo.")
        return True
    
    exe = procurar_opencode()
    if not exe:
        logger.warning("Executável OpenCode não encontrado.")
        return False
    try:
        # Inicia servidor SEM senha (uso local)
        cmd = [exe, "serve", "--port", "4096"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=str(DIRETORIO_RAIZ)
        )
        st.session_state.opencode_server_proc = proc
        logger.info("Iniciando servidor OpenCode...")
        for _ in range(15):
            if servidor_opencode_ativo():
                logger.info("Servidor OpenCode iniciado com sucesso.")
                return True
            time.sleep(1)
        proc.kill()
        logger.warning("Timeout ao iniciar servidor OpenCode.")
        return False
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor OpenCode: {e}")
        return False


def _parar_servidor_opencode():
    proc = st.session_state.get("opencode_server_proc")
    if proc and proc.poll() is None:
        try:
            proc.kill()
        except:
            pass

atexit.register(_parar_servidor_opencode)


def _check_zen_rate_limit():
    """Verifica se há limite de uso excedido no Zen e retorna info sobre reset."""
    try:
        headers = _auth_headers()
        status_resp = requests.get(f"{OPENCODE_SERVER_URL}/session/status", headers=headers, timeout=10)
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            for session_id, info in status_data.items():
                if isinstance(info, dict) and info.get("type") == "retry":
                    msg = info.get("message", "")
                    next_ts = info.get("next", 0)
                    if "Free usage exceeded" in msg or "rate limit" in msg.lower():
                        reset_time = None
                        if next_ts:
                            reset_time = time.strftime("%H:%M:%S", time.localtime(next_ts / 1000))
                        return {
                            "limited": True,
                            "message": msg,
                            "reset_at": reset_time,
                            "next_timestamp": next_ts
                        }
        return {"limited": False}
    except:
        return {"limited": False}


MODELOS = {
    "nemotron-3-super-free": "Nemotron 3 Super Free",
    "big-pickle": "Big Pickle",
    "deepseek-v4-flash-free": "DeepSeek V4 Flash Free",
    "minimax-m2.5-free": "MiniMax M2.5 Free",
    "qwen3.6-plus-free": "Qwen3.6 Plus Free",
}


def enviar_prompt_opencode(prompt, model_id="nemotron-3-super-free", timeout=300):
    """
    Envia prompt para o OpenCode via HTTP API com polling assíncrono.
    
    Args:
        prompt: Texto do prompt
        model_id: ID do modelo (ex: "minimax-m2.5-free", "big-pickle")
        timeout: Timeout em segundos
        
    Returns:
        dict: {"success": bool, "response": str, "error": str, "model": str, "rate_limit_info": dict}
    """
    if not servidor_opencode_ativo():
        return {"success": False, "response": "", "error": "Servidor OpenCode não está ativo.", "model": model_id}
    
    url = f"{OPENCODE_SERVER_URL}/session"
    headers = _auth_headers()
    headers["Content-Type"] = "application/json"
    
    try:
        # Step 1: Create session
        logger.info(f"Criando sessão com modelo: {model_id}")
        session_resp = requests.post(
            url,
            json={"message": "Analise de dados ENADE"},
            headers=headers,
            timeout=15
        )
        
        if session_resp.status_code != 200:
            return {
                "success": False,
                "response": "",
                "error": f"Falha ao criar sessão: HTTP {session_resp.status_code}",
                "model": model_id
            }
        
        session_data = session_resp.json()
        session_id = session_data.get("id")
        if not session_id:
            return {
                "success": False,
                "response": "",
                "error": "Sessão criada sem ID válido.",
                "model": model_id
            }
        
        logger.info(f"Sessão criada: {session_id}")
        
        # Step 2: Send message asynchronously
        msg_url = f"{OPENCODE_SERVER_URL}/session/{session_id}/prompt_async"
        msg_body = {
            "parts": [{"type": "text", "text": prompt}],
            "model": {
                "providerID": "opencode",
                "modelID": model_id
            }
        }
        
        logger.info(f"Enviando mensagem assíncrona para modelo {model_id}...")
        async_resp = requests.post(msg_url, json=msg_body, headers=headers, timeout=30)
        
        if async_resp.status_code != 204:
            return {
                "success": False,
                "response": "",
                "error": f"Falha ao enviar mensagem: HTTP {async_resp.status_code}",
                "model": model_id
            }
        
        # Step 3: Poll for response
        logger.info("Aguardando resposta via polling...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            time.sleep(3)
            
            # Check session status for rate limit errors
            status_resp = requests.get(f"{OPENCODE_SERVER_URL}/session/status", headers=headers, timeout=10)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                current_status = status_data.get(session_id, {})
                status_type = current_status.get("type", "")
                
                if status_type == "retry":
                    error_msg = current_status.get("message", "")
                    if "Free usage exceeded" in error_msg or "rate limit" in error_msg.lower():
                        next_ts = current_status.get("next", 0)
                        reset_time = time.strftime("%H:%M:%S", time.localtime(next_ts / 1000)) if next_ts else "desconhecido"
                        logger.warning(f"Rate limit detectado: {error_msg}")
                        return {
                            "success": False,
                            "response": "",
                            "error": f"Limite de uso gratuito excedido. Reset às {reset_time}. Tente outro modelo ou aguarde.",
                            "model": model_id,
                            "rate_limit_info": {"limited": True, "reset_at": reset_time}
                        }
            
            # Get messages
            msg_list_url = f"{OPENCODE_SERVER_URL}/session/{session_id}/message?limit=5"
            msg_list_resp = requests.get(msg_list_url, headers=headers, timeout=10)
            
            if msg_list_resp.status_code != 200:
                continue
            
            messages = msg_list_resp.json()
            if not isinstance(messages, list):
                continue
            
            # Look for assistant message with text content
            for msg in messages:
                info = msg.get("info", {})
                if info.get("role") == "assistant":
                    parts = msg.get("parts", [])
                    for part in parts:
                        if part.get("type") == "text":
                            text = part.get("text", "")
                            if text and len(text.strip()) > 10:
                                elapsed = int(time.time() - start_time)
                                logger.info(f"Resposta recebida em {elapsed}s: {len(text)} caracteres")
                                return {
                                    "success": True,
                                    "response": text,
                                    "error": "",
                                    "model": model_id
                                }
            
            # Timeout reached
        logger.warning(f"Timeout após {timeout}s ao aguardar resposta do OpenCode.")
        return {
            "success": False,
            "response": "",
            "error": f"Timeout após {timeout}s. O modelo '{model_id}' pode estar sobrecarregado.",
            "model": model_id
        }
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout ao comunicar com OpenCode.")
        return {
            "success": False,
            "response": "",
            "error": "Timeout na comunicação com o servidor OpenCode.",
            "model": model_id
        }
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Erro de conexão com OpenCode: {e}")
        return {
            "success": False,
            "response": "",
            "error": f"Erro de conexão: {str(e)}",
            "model": model_id
        }
    except Exception as e:
        logger.error(f"Erro inesperado ao comunicar com OpenCode: {e}")
        return {
            "success": False,
            "response": "",
            "error": f"Erro: {str(e)}",
            "model": model_id
        }

# =============================================================================

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

st.title("PRISM-EDU - Painel de IA Educacional - ENADE 2023")
st.markdown("**K-Means + LCA + OpenCode** — Análise de Perfis de Aprendizagem com Explicação em Linguagem Natural")
st.markdown("---")

# =============================================================================
# 2. SIDEBAR - FILTROS GLOBAIS (sempre visíveis)
# =============================================================================
with st.sidebar:
    st.header("Filtros de Pesquisa")

    km_disponivel = df_base is not None
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
    st.markdown("###  OpenCode")
    senha_opencode = st.text_input(
        "Senha (opcional):",
        type="password",
        key="senha_opencode",
        help="Senha para conexão com servidor OpenCode remoto (opcional). Deixe vazio para usar o servidor local sem autenticação."
    )
    if senha_opencode:
        pass  # senha usada via _auth_headers() nas chamadas HTTP
    st.markdown("---")
    if st.button("Sair e Fechar", use_container_width=True):
        st.success("A encerrar... Pode fechar a janela.")
        os._exit(0)

# =============================================================================
# 3. NAVEGAÇÃO PRINCIPAL (radio horizontal substitui st.tabs)
# =============================================================================
TABS = [
    " Diagnóstico e Prescrição",
    " Fatores de Sucesso (Preditivo)",
    " Validação + Clustering",
    " Plano de Ação por Perfil",
    " LCA - Classes Latentes",
    " OpenCode + IA Explicativa"
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
    if df_base is None:
        st.warning("Base de dados K-Means não encontrada. Execute a Fase 1 e 2 primeiro.")
    elif not km_disponivel or 'df_final' not in dir() or df_final.empty:
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

        st.subheader(f"📍 Top 5 Questões Críticas - IES {ies_selecionada}")
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
    if df_base is None:
        st.warning("Base não encontrada.")
    elif not km_disponivel or 'nome_curso_arquivo' not in dir():
        st.warning("Selecione IES/Curso no filtro lateral.")
    else:
        caminho_pesos = pasta_preditivos / f"importancia_variaveis_{nome_curso_arquivo}.csv"
        caminho_metricas_pred = pasta_preditivos / f"metricas_modelos_{nome_curso_arquivo}.csv"
        if caminho_pesos.exists() and caminho_metricas_pred.exists():
            df_pesos = pd.read_csv(caminho_pesos, sep=';')
            df_metricas_pred = pd.read_csv(caminho_metricas_pred, sep=';')
            col_m1, col_m2 = st.columns([1, 2.5])
            with col_m1:
                st.markdown("####  Qualidade da Predição")
                st.dataframe(df_metricas_pred[['Modelo', 'Acuracia', 'F1_Score']].set_index('Modelo'), use_container_width=True)
            with col_m2:
                st.markdown("####  Top 10 Matérias que mais impactam")
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
    if df_base is None:
        st.warning("Base não encontrada.")
    elif not km_disponivel or 'nome_curso_arquivo' not in dir():
        st.warning("Selecione IES/Curso no filtro lateral.")
    else:
        st.header(" Validação dos Agrupamentos (K-Means)")
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
    if df_base is None:
        st.warning("Base não encontrada.")
    elif not km_disponivel or 'nome_curso_final' not in dir():
        st.warning("Selecione IES/Curso no filtro lateral.")
    else:
        col_titulo, col_param = st.columns([2, 1])
        with col_titulo:
            st.header(f" Plano de Ação: {nome_curso_final}")
        with col_param:
            limite_top_n = st.slider("Top N problemas:", 1, 38, 5, key="top_n_tab4")

        with st.expander(" Explicação dos Algoritmos e Metodologia"):
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
                st.info(f" K-Livre: {k_livre} (Silhueta: {silhueta}) | K-Aplicado: {k_aplicado}")

                if 'FALHA_SISTEMICA_IES' in df_ia_filtrado.columns:
                    falha = df_ia_filtrado['FALHA_SISTEMICA_IES'].iloc[0]
                    if pd.notna(falha) and str(falha).strip() not in ["Nenhuma", "nan", ""]:
                        st.error(f" Falha Institucional (interseção em todos os perfis): {falha}")

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
    st.header(" Análise de Classes Latentes (LCA)")
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
            if km_disponivel and 'grupo_selecionado' in dir():
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
                st.caption(f" Sincronizado com o filtro global: {nome_curso_final} / IES {ies_selecionada}")
            else:
                st.caption(" Selecione um curso e IES no filtro lateral (sidebar) para sincronizar automaticamente.")

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

                with st.expander(" Justificativa da seleção de k"):
                    st.info(row['justificativa_k'])
                    st.caption(f"Log-verossimilhança: {row['log_lik']:.2f} | Convergiu: {row['convergiu']}")

                st.markdown("---")

                if not df_cls_ies.empty:
                    st.subheader(" Perfis Identificados pela LCA")

                    prob_cols = [f'PROB_Q{i}' for i in range(1, 39)]
                    gap_cols = [f'GAP_Q{i}' for i in range(1, 39)]
                    q_labels = [f'Q{i}' for i in range(1, 39)]

                    fig = go.Figure()
                    for _, cls_row in df_cls_ies.sort_values('taxa_acerto_media', ascending=False).iterrows():
                        probs = [cls_row.get(c, 0) for c in prob_cols]
                        classe_nome = f"Classe {int(cls_row['classe_id'])} ({int(cls_row['n_alunos'])} alunos, {cls_row['taxa_acerto_media']:.2f})"
                        fig.add_trace(go.Scatter(
                            x=q_labels, y=probs, mode='lines+markers',
                            name=classe_nome, marker=dict(size=5)))

                    fig.update_layout(
                        title="Probabilidade de Acerto por Questão - Classes LCA",
                        xaxis_title="Questão",
                        yaxis_title="P(Acerto)",
                        yaxis_range=[0, 1.05],
                        hovermode='x unified',
                        height=400)
                    fig.add_vrect(x0=-0.5, x1=7.5, fillcolor="gray", opacity=0.05,
                        layer="below", line_width=0, annotation_text="FG", annotation_position="top left")
                    st.plotly_chart(fig, use_container_width=True)

                    fig_gap = go.Figure()
                    for _, cls_row in df_cls_ies.sort_values('taxa_acerto_media', ascending=False).iterrows():
                        gaps = [cls_row.get(c, 0) for c in gap_cols]
                        fig_gap.add_trace(go.Scatter(
                            x=q_labels, y=gaps, mode='lines+markers',
                            name=f"Classe {int(cls_row['classe_id'])}", marker=dict(size=5)))
                    fig_gap.update_layout(
                        title="GAP (diferença para média da IES) por Questão",
                        xaxis_title="Questão", yaxis_title="GAP",
                        hovermode='x unified', height=350)
                    fig_gap.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_gap.add_vrect(x0=-0.5, x1=7.5, fillcolor="gray", opacity=0.05,
                        layer="below", line_width=0)
                    st.plotly_chart(fig_gap, use_container_width=True)

                    st.subheader(" Detalhamento das Classes")
                    cols_show = ['classe_id', 'n_alunos', 'pct_alunos', 'taxa_acerto_media',
                        'taxa_acerto_media_ies', 'itens_diferencialmente_fortes', 'itens_diferencialmente_fracos']
                    cols_show = [c for c in cols_show if c in df_cls_ies.columns]
                    st.dataframe(df_cls_ies[cols_show].rename(columns={
                        'classe_id': 'Classe', 'n_alunos': 'N Alunos',
                        'pct_alunos': '%', 'taxa_acerto_media': 'Média Acertos',
                        'taxa_acerto_media_ies': 'Média IES',
                        'itens_diferencialmente_fortes': 'Itens Fortes',
                        'itens_diferencialmente_fracos': 'Itens Fracos'
                    }), use_container_width=True)

                st.markdown("---")
                st.subheader(" Distribuição de k no Curso (todas as IES)")
                df_curso_k = df_lca_curso_filtro['k_escolhido'].value_counts().sort_index().reset_index()
                df_curso_k.columns = ['k', 'quantidade']
                fig_k = px.bar(df_curso_k, x='k', y='quantidade', text='quantidade',
                    color='quantidade', color_continuous_scale='Blues')
                fig_k.update_traces(textposition='outside')
                fig_k.update_layout(xaxis=dict(tickmode='linear', dtick=1), height=300)
                st.plotly_chart(fig_k, use_container_width=True)

                if df_lca_crit is not None:
                    df_crit_ies = df_lca_crit[(df_lca_crit['curso'] == curso_lca_sel) &
                                              (df_lca_crit['ies'] == ies_lca_sel)]
                    if not df_crit_ies.empty:
                        st.subheader(" Critérios de Seleção de k")
                        cols_crit = ['k', 'bic', 'aic', 'entropia', 'menor_classe_pct']
                        cols_crit = [c for c in cols_crit if c in df_crit_ies.columns]
                        st.dataframe(df_crit_ies[cols_crit].set_index('k').round(3), use_container_width=True)

                st.markdown("---")
                st.subheader(" Figuras da Análise (Artigo SBIE)")
                st.caption("Figuras referentes aos cursos de **Medicina**, **Engenharia Civil** e **Enfermagem** — artigos publicados no SBIE.")
                cursos_artigo = ['medicina', 'engenharia_civil', 'enfermagem']
                figuras_exibidas = 0
                for curso_artigo in cursos_artigo:
                    fig_paths = list(PASTA_LCA_FIGURAS.glob(f"*{curso_artigo}*.png"))
                    if fig_paths:
                        for fp in fig_paths:
                            st.image(str(fp), use_container_width=True, caption=f"{fp.stem.replace('_', ' ').title()}")
                            figuras_exibidas += 1
                if figuras_exibidas == 0:
                    imagens = list(PASTA_LCA_FIGURAS.glob("fig*.png"))
                    if imagens:
                        st.info("Nenhuma figura específica encontrada. Exibindo figuras gerais:")
                        cols_img = st.columns(3)
                        for i, img_p in enumerate(imagens[:6]):
                            with cols_img[i % 3]:
                                st.image(str(img_p), use_container_width=True, caption=img_p.stem)
                    else:
                        st.info("Figuras não encontradas em: " + str(PASTA_LCA_FIGURAS))

            else:
                st.warning("Nenhum resultado LCA encontrado para esta IES/Curso.")

# =============================================================================
# ABA 6: OPENCODE + IA EXPLICATIVA
# =============================================================================
elif tab_selector == TABS[5]:
    st.header(" OpenCode + IA Explicativa")
    st.markdown("""
    Selecione um arquivo CSV de resultados para gerar uma **explicação em linguagem natural**,
    acessível para não-especialistas. O dashboard chama o **OpenCode.ai** via API HTTP
    para analisar os dados e produzir um texto explicativo.
    """)

    # Status do servidor
    server_online = servidor_opencode_ativo()
    status_color = "🟢" if server_online else "🔴"
    status_text = "Online" if server_online else "Offline"
    
    # Check rate limit
    rate_info = _check_zen_rate_limit() if server_online else {"limited": False}
    if rate_info.get("limited"):
        reset_at = rate_info.get("reset_at", "desconhecido")
        st.error(f" **Limite de uso gratuito Zen excedido** — Reset previsto para **{reset_at}**. [Adicionar créditos](https://opencode.ai/zen)")
        status_text += " (limite excedido)"
    
    st.caption(f"{status_color} Servidor OpenCode: {status_text} — {OPENCODE_SERVER_URL}")

    st.subheader("1. Selecione o CSV para análise")

    csvs_disponiveis = []
    if PASTA_RESULTADOS.exists():
        csvs_disponiveis.extend(list(PASTA_RESULTADOS.rglob("*.csv")))
    if PASTA_LCA.exists():
        csvs_disponiveis.extend(list(PASTA_LCA.glob("*.csv")))

    csvs_validos = []
    for p in csvs_disponiveis:
        try:
            if p.stat().st_size <= 50 * 1024 * 1024:
                csvs_validos.append(p)
        except:
            pass

    if not csvs_validos:
        st.error("Nenhum arquivo CSV encontrado nas pastas de resultados.")
        st.info("Execute as fases de processamento para gerar dados.")
    else:
        csvs_filtrados = []
        if km_disponivel and 'nome_curso_arquivo' in dir():
            curso_lower = nome_curso_arquivo.lower()
            for p in csvs_validos:
                p_stem_lower = p.stem.lower()
                p_path_lower = str(p).lower()
                if curso_lower in p_stem_lower or curso_lower in p_path_lower:
                    csvs_filtrados.append(p)
            caminho_consolidado = PASTA_RESULTADOS / 'analise_por_ies_curso_enade.csv'
            if caminho_consolidado in csvs_validos and caminho_consolidado not in csvs_filtrados:
                csvs_filtrados.insert(0, caminho_consolidado)
            if not csvs_filtrados:
                csvs_filtrados = csvs_validos
        else:
            csvs_filtrados = csvs_validos

        csv_labels = {str(p.relative_to(p.parents[2]) if len(p.parents) > 2 else p.name): p
                      for p in sorted(set(csvs_filtrados))}
        labels_ordenados = sorted(csv_labels.keys())
        csv_default_idx = 0

        csv_selecionado = st.selectbox("Arquivo CSV:", labels_ordenados,
                                        index=csv_default_idx, key="csv_selector")
        caminho_csv = csv_labels[csv_selecionado]

        if km_disponivel and 'nome_curso_final' in dir():
            st.caption(f" Filtrado pelo curso: {nome_curso_final} / IES {ies_selecionada}")

        st.subheader("2. Prévia dos Dados")
        try:
            df_csv_preview = pd.read_csv(caminho_csv, sep=';', nrows=10,
                encoding='utf-8-sig', on_bad_lines='skip')
            st.dataframe(df_csv_preview, use_container_width=True)
            n_linhas = sum(1 for _ in open(caminho_csv, 'r', encoding='utf-8-sig')) - 1
            n_cols = len(df_csv_preview.columns)
            st.caption(f"Mostrando 10 de ~{n_linhas} linhas | {n_cols} colunas | {caminho_csv.name}")
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")
            st.stop()

        st.subheader("3. Personalize a Análise")
        
        col_model, col_tom = st.columns([1, 1])
        with col_model:
            modelo_id = st.selectbox(
                "Modelo IA:",
                options=list(MODELOS.keys()),
                format_func=lambda x: MODELOS[x],
                index=0,
                key="modelo_opencode"
            )
        
        with col_tom:
            tom = st.selectbox("Tom da explicação:",
                ["Didático e acessível (para leigos)",
                 "Técnico e detalhado (para coordenadores)",
                 "Resumo executivo (para diretores)",
                 "Crítico e propositivo (para melhoria)"],
                key="tom_ia")

        foco = st.text_area("Foco adicional (opcional):",
            placeholder="Ex: 'Destaque as principais deficiências em Matemática' ou 'Compare o desempenho desta IES com a média nacional'",
            key="foco_ia")

        st.subheader("4. Gerar Explicação")
        
        gerar = st.button(
            " Gerar Explicação com OpenCode",
            type="primary",
            use_container_width=True,
            disabled=not server_online
        )
        
        if not server_online:
            st.warning(" Servidor OpenCode não está ativo. Clique no botão abaixo para iniciar.")
            if st.button(" Iniciar Servidor OpenCode", use_container_width=True):
                with st.spinner("Iniciando servidor OpenCode..."):
                    if iniciar_servidor_opencode():
                        st.success(" Servidor iniciado!")
                        st.rerun()
                    else:
                        st.error(" Falha ao iniciar servidor. Verifique se o OpenCode está instalado.")

        if gerar:
            # Pré-processa dados
            df_full = pd.read_csv(caminho_csv, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
            resumo = {
                "arquivo": caminho_csv.name,
                "linhas": len(df_full),
                "colunas": list(df_full.columns),
                "colunas_numericas": [c for c in df_full.columns if pd.api.types.is_numeric_dtype(df_full[c])],
                "resumo_estatistico": {},
            }
            for c in resumo["colunas_numericas"][:10]:
                resumo["resumo_estatistico"][c] = {
                    "media": round(float(df_full[c].mean()), 2),
                    "min": round(float(df_full[c].min()), 2),
                    "max": round(float(df_full[c].max()), 2),
                }

            amostra = df_full.head(5).to_string()

            prompt = f"""Analise o seguinte conjunto de dados educacionais do ENADE e gere uma explicação em linguagem natural.

ARQUIVO: {caminho_csv.name}
TOM: {tom}
{'FOCO ADICIONAL: ' + foco if foco else ''}

RESUMO DOS DADOS:
- {resumo['linhas']} linhas, {len(resumo['colunas'])} colunas
- Colunas: {', '.join(resumo['colunas'][:15])}{'...' if len(resumo['colunas']) > 15 else ''}
- Colunas numéricas: {', '.join(resumo['colunas_numericas'][:8])}

ESTATÍSTICAS BÁSICAS:
{json.dumps(resumo['resumo_estatistico'], indent=2, ensure_ascii=False)}

AMOSTRA (primeiras 5 linhas):
{amostra}

Com base nestes dados, produza uma explicação que:
1. Contextualize o que são estes dados (de onde vêm, o que significam)
2. Destaque os principais achados, tendências e padrões interessantes
3. Aponte anomalias ou valores que merecem atenção
4. Conclua com recomendações práticas
5. Seja acessível para um público não-especialista em dados

Formate a resposta em MARKDOWN, com seções claras e linguagem didática.
"""
            
            # Mostra preview do prompt
            prompt_preview = prompt[:75] + "..." if len(prompt) > 75 else prompt
            st.info(f" Modelo: **{MODELOS[modelo_id]}** | Prompt: `{prompt_preview}`")
            
            # Container para streaming
            response_container = st.empty()
            timer_container = st.empty()
            
            # Timer thread
            class TimerThread(threading.Thread):
                def __init__(self):
                    super().__init__(daemon=True)
                    self.running = True
                    self.elapsed = 0
                
                def run(self):
                    while self.running:
                        time.sleep(1)
                        self.elapsed += 1
                
                def stop(self):
                    self.running = False
            
            timer = TimerThread()
            timer.start()
            
            try:
                with st.status("Enviando prompt para OpenCode...", expanded=True) as status:
                    status.update(label=f"Processando com {MODELOS[modelo_id]}... ⏱️ 0s", state="running")
                    
                    # Update timer display
                    def update_timer():
                        while timer.running:
                            timer_container.caption(f"⏱️ Tempo decorrido: {timer.elapsed}s")
                            time.sleep(1)
                    
                    timer_thread = threading.Thread(target=update_timer, daemon=True)
                    timer_thread.start()
                    
                    # Send prompt
                    resultado = enviar_prompt_opencode(prompt, model_id=modelo_id, timeout=300)
                    
                    timer.stop()
                    timer_container.empty()
                    
                    if resultado["success"]:
                        status.update(label=f" Explicação gerada em {timer.elapsed}s!", state="complete")
                        st.markdown(f"###  Explicação Gerada — {MODELOS[modelo_id]}")
                        st.markdown("---")
                        st.markdown(resultado["response"])
                        
                        st.download_button(
                            label=" Download da Explicação (.md)",
                            data=resultado["response"].encode('utf-8'),
                            file_name=f"explicacao_opencode_{caminho_csv.stem}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    else:
                        error_msg = resultado.get("error", "Erro desconhecido")
                        is_rate_limit = "limite" in error_msg.lower() or "excedido" in error_msg.lower() or "usage exceeded" in error_msg.lower()
                        
                        if is_rate_limit:
                            status.update(label=" Limite de uso Zen excedido", state="error")
                            reset_at = resultado.get("rate_limit_info", {}).get("reset_at", "desconhecido")
                            st.error(f" **Limite de uso gratuito do OpenCode Zen excedido.**")
                            st.info(f" Reset previsto para **{reset_at}**. Você pode adicionar créditos em [opencode.ai/zen](https://opencode.ai/zen) para uso ilimitado.")
                            st.warning(" Enquanto isso, use a **análise estatística automática** abaixo ou tente outro modelo quando o limite resetar.")
                        else:
                            status.update(label=f" {error_msg[:50]}...", state="error")
                            st.warning(f"**Erro OpenCode:** {error_msg}")
                        
                        # Fallback: statistical analysis
                        st.markdown("###  Análise Estatística Automática (fallback)")
                        st.markdown("---")
                        st.markdown(f"** Arquivo:** `{caminho_csv.name}`")
                        st.markdown(f"** Dimensões:** {resumo['linhas']} linhas × {len(resumo['colunas'])} colunas")
                        st.markdown("---")
                        st.markdown("** Colunas numéricas analisadas:**")
                        for col, stats in resumo["resumo_estatistico"].items():
                            st.markdown(f"- **{col}**: média={stats['media']}, min={stats['min']}, max={stats['max']}")
                        st.markdown("---")
                        if not is_rate_limit:
                            st.info(" Dica: Tente outro modelo ou aguarde alguns minutos e tente novamente.")
                        
            except Exception as e:
                timer.stop()
                timer_container.empty()
                st.error(f"Erro ao gerar explicação: {e}")
        else:
            st.info(" Selecione um CSV, personalize o tom e clique no botão para gerar a explicação com IA.")

        # =============================================================================
        # SEÇÃO 5: LEITOR DE RELATÓRIOS .md GERADOS
        # =============================================================================
        st.markdown("---")
        st.subheader("5. Relatórios .md Gerados")
        st.markdown("Visualize relatórios gerados pelo **K-Means (Fase 4)** ou pelo **script OpenCode headless**.")

        pasta_md = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS' / 'Dashboards_Markdown'
        pasta_md_old = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS' / 'Dashboards'

        mds = sorted(list(pasta_md.glob("*.md"))) + sorted(list(pasta_md_old.glob("*.md")))
        if mds:
            if km_disponivel and 'nome_curso_arquivo' in dir():
                mds_filtrados = [m for m in mds if nome_curso_arquivo in m.stem.lower()]
                if not mds_filtrados:
                    mds_filtrados = mds
            else:
                mds_filtrados = mds

            nomes_md = [m.name for m in mds_filtrados]
            md_selecionado = st.selectbox(
                "Selecione um relatório para visualizar:",
                nomes_md,
                key="md_selector"
            )
            caminho_md = next(m for m in mds_filtrados if m.name == md_selecionado)
            try:
                conteudo_md = caminho_md.read_text(encoding='utf-8')
                st.markdown("---")
                st.markdown(conteudo_md)
            except Exception as e:
                st.error(f"Erro ao ler o arquivo .md: {e}")
        else:
            st.info("Nenhum relatório .md encontrado. Gere relatórios com a **Fase 4** (K-Means) ou execute o script OpenCode headless.")
