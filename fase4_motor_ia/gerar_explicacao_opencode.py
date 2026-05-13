import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent

def formatar_nome(nome):
    import unicodedata
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_')

# =============================================================================
# 2. FUNÇÕES DE GERENCIAMENTO DO OPENCODE (mesma lógica do dashboard)
# =============================================================================

def procurar_opencode():
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


def servidor_ativo(porta=4096):
    try:
        req = urllib.request.Request(f"http://localhost:{porta}/global/health")
        resp = urllib.request.urlopen(req, timeout=3)
        return resp.status == 200
    except:
        return False


def iniciar_servidor(exe, porta=4096, senha=""):
    if servidor_ativo(porta):
        return True
    cmd = [exe, "serve", "--port", str(porta)]
    env = os.environ.copy() if senha else None
    if senha:
        env["OPENCODE_SERVER_PASSWORD"] = senha
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(DIRETORIO_RAIZ), env=env
    )
    for _ in range(15):
        if servidor_ativo(porta):
            return True
        time.sleep(1)
    proc.kill()
    return False


def parar_servidor(proc):
    if proc and proc.poll() is None:
        try:
            proc.kill()
        except:
            pass


# =============================================================================
# 3. CONSTRUÇÃO DO PROMPT (mesmo template do dashboard)
# =============================================================================

def construir_prompt(caminho_csv, tom, foco=""):
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
{ f'FOCO ADICIONAL: {foco}' if foco else '' }

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
    return prompt


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gera explicação em linguagem natural usando OpenCode.ai a partir de um CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python gerar_explicacao_opencode.py --csv "..\\arquivosgerados\\RESULTADOS\\analise_por_ies_curso_enade.csv" --output "explicacao.md"

  python gerar_explicacao_opencode.py --csv "..\\arquivosgerados\\RESULTADOS_LCA\\01_relatorio_geral_por_ies.csv" --tom "Técnico" --foco "Destaque as classes latentes" --output "explicacao_lca.md" --opencode-password "minha_senha"
        """
    )
    parser.add_argument("--csv", required=True, help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--tom", default="Didático e acessível (para leigos)",
                        choices=["Didático e acessível (para leigos)",
                                 "Técnico e detalhado (para coordenadores)",
                                 "Resumo executivo (para diretores)",
                                 "Crítico e propositivo (para melhoria)"],
                        help="Tom da explicação (default: Didático)")
    parser.add_argument("--foco", default="", help="Foco adicional para a análise (opcional)")
    parser.add_argument("--output", required=True, help="Caminho para salvar o arquivo .md gerado")
    parser.add_argument("--opencode-password", default="",
                        help="Senha do servidor OpenCode (opcional, apenas se o servidor remoto exigir)")
    parser.add_argument("--porta", type=int, default=4096, help="Porta do servidor OpenCode (default: 4096)")
    parser.add_argument("--keep-server", action="store_true",
                        help="Nao para o servidor ao final (útil para múltiplas execuções)")

    args = parser.parse_args()

    caminho_csv = Path(args.csv)
    if not caminho_csv.exists():
        print(f"❌ ERRO: CSV não encontrado: {caminho_csv}")
        sys.exit(1)

    senha = args.opencode_password or os.environ.get("OPENCODE_SERVER_PASSWORD", "")

    # ------------------------------------------------------------------
    # 1. Procura OpenCode
    # ------------------------------------------------------------------
    print("🔍 Procurando OpenCode...")
    exe = procurar_opencode()
    if not exe:
        print("❌ OpenCode não encontrado. Instale com: npm install -g opencode-ai")
        sys.exit(1)
    print(f"   ✅ Encontrado: {exe}")

    # ------------------------------------------------------------------
    # 2. Constrói o prompt
    # ------------------------------------------------------------------
    print(f"📄 Lendo CSV: {caminho_csv}")
    print(f"🎯 Tom: {args.tom}")
    if args.foco:
        print(f"🎯 Foco: {args.foco}")
    prompt = construir_prompt(caminho_csv, args.tom, args.foco)

    # ------------------------------------------------------------------
    # 3. Inicia servidor OpenCode
    # ------------------------------------------------------------------
    print(f"🌐 Iniciando servidor OpenCode na porta {args.porta}...")
    server_ok = iniciar_servidor(exe, args.porta, senha)
    if not server_ok:
        print("❌ Falha ao iniciar o servidor OpenCode.")
        sys.exit(1)
    print("   ✅ Servidor pronto!")

    # ------------------------------------------------------------------
    # 4. Executa OpenCode run
    # ------------------------------------------------------------------
    print("🤖 Gerando explicação com OpenCode...")
    servidor_proc = None  # We don't have a reference to the Popen object here
    # Re-get the server proc from subprocess internals (we didn't store it)
    # For simplicity, we just run the command
    try:
        cmd = [exe, "run", "--attach", f"http://localhost:{args.porta}"]
        if senha:
            cmd += ["--password", senha]
        cmd += [prompt]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(DIRETORIO_RAIZ),
            encoding='utf-8', errors='replace'
        )
        resposta = result.stdout if result.stdout else result.stderr
        if not resposta or len(resposta.strip()) < 20:
            print("⚠️  O OpenCode não produziu uma resposta válida.")
            print("   STDERR:", result.stderr[:500] if result.stderr else "vazio")
            resposta = None
    except subprocess.TimeoutExpired:
        print("❌ OpenCode excedeu o tempo limite (5 min).")
        resposta = None
    except Exception as e:
        print(f"❌ Erro ao executar OpenCode: {e}")
        resposta = None

    # ------------------------------------------------------------------
    # 5. Salva resposta
    # ------------------------------------------------------------------
    if resposta:
        caminho_output = Path(args.output)
        caminho_output.parent.mkdir(parents=True, exist_ok=True)

        cabecalho = f"""# 🤖 Explicação OpenCode — {caminho_csv.name}

**Data:** {time.strftime('%Y-%m-%d %H:%M')}
**Arquivo:** `{caminho_csv.name}`
**Tom:** {args.tom}
{f'**Foco:** {args.foco}' if args.foco else ''}
**Total de linhas:** {len(pd.read_csv(caminho_csv, sep=';', encoding='utf-8-sig', on_bad_lines='skip'))}

---

"""
        with open(caminho_output, 'w', encoding='utf-8') as f:
            f.write(cabecalho + resposta)

        print(f"\n✅ Explicação salva em: {caminho_output.resolve()}")
    else:
        print("\n❌ Nenhuma explicação foi gerada.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Para o servidor (a menos que --keep-server)
    # ------------------------------------------------------------------
    if not args.keep_server:
        print("🛑 Parando servidor OpenCode...")
        try:
            import requests
            requests.post(f"http://localhost:{args.porta}/global/shutdown", timeout=5)
        except:
            pass
        # Força kill em processos opencode
        try:
            subprocess.run(["taskkill", "/F", "/IM", "opencode*"], capture_output=True, timeout=5)
        except:
            pass

    print("\n🎉 Processo concluído com sucesso!")


if __name__ == "__main__":
    main()
