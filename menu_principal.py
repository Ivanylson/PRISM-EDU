import os
import sys
import subprocess
from pathlib import Path

# Caminho base do projeto (onde este menu está)
BASE_DIR = Path(__file__).resolve().parent

def executar_script(pasta, nome_arquivo):
    """Função para rodar um script python dentro de uma subpasta."""
    caminho_script = BASE_DIR / pasta / nome_arquivo
    
    print(f"\n{'='*60}")
    print(f"INICIANDO: {nome_arquivo} (na pasta {pasta})")
    print(f"{'='*60}")
    
    if not caminho_script.exists():
        print(f"\n ERRO: O arquivo '{nome_arquivo}' não foi encontrado na pasta '{pasta}'.")
        input("Pressione ENTER para continuar...")
        return

    try:
        subprocess.run([sys.executable, str(caminho_script)], check=True)
        print(f"\n SUCESSO: {nome_arquivo} finalizado.")
    except subprocess.CalledProcessError:
        print(f"\n ERRO: Falha ao executar {nome_arquivo}.")
        input("Pressione ENTER para continuar...")

def abrir_dashboard(pasta="fase5_visualizacao", nome_arquivo="dashboard_completo.py"):
    """Função específica para rodar o Streamlit."""
    caminho_script = BASE_DIR / pasta / nome_arquivo
    
    print(f"\n{'='*60}")
    print(f" ABRINDO DASHBOARD: {nome_arquivo}")
    print(f"{'='*60}")
    
    if not caminho_script.exists():
        print(f"\n ERRO: Dashboard não encontrado em {caminho_script}.")
        input("Pressione ENTER para continuar...")
        return
        
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(caminho_script)])
    except Exception as e:
        print(f"\n ERRO ao abrir o dashboard: {e}")
        input("Pressione ENTER para continuar...")

def menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n SISTEMA DE IA EDUCACIONAL - ENADE")
        print("-" * 50)
        print("1. Fase 1: Pré-processamento (Microdados e PDFs)")
        print("2. Fase 2: Cruzamento de Dados (Geral e Benchmarks)")
        print("3. Fase 3: Validação de IA (Batalha de Modelos)")
        print("4. Fase 4: Motor de IA (Agrupamento, Prescrição e LCA)")
        print("5. Fase 5: Gerar Relatórios HTML (Análise IE)")
        print("6. RODA TUDO: Pipeline K-Means + LCA Completo")
        print("-" * 50)
        print("7. ABRIR DASHBOARD INTERATIVO (completo)")
        print("8. EXECUTAR LCA (Classes Latentes - SBIE)")
        print("9. ABRIR OPCODE + IA EXPLICATIVA DIRETO")
        print("10. GERAR .md HEADLESS (OpenCode em lote para todos CSVs do curso)")
        print("0. Sair")
        print("-" * 50)
        
        escolha = input("Selecione uma opção: ")
        
        if escolha == '1':
            executar_script("fase1_pre_processamento", "preprocessamentoMicrodados.py")
            executar_script("fase1_pre_processamento", "preprocessamentoRelatorioSintese.py")
        elif escolha == '2':
            executar_script("fase2_cruzamento_dados", "analiseGeral.py")
            executar_script("fase2_cruzamento_dados", "analisesTeste.py")
        elif escolha == '3':
            executar_script("fase3_validacao_ia", "batalha_modelos_preditivos.py")
            executar_script("fase3_validacao_ia", "gerador_validacao_agrupamentos.py")
        elif escolha == '4':
            executar_script("fase4_motor_ia", "motor_agrupamento_triplo.py")
            executar_script("fase4_motor_ia", "mapeador_deficiencias.py")
            executar_script("fase4_motor_ia", "testeKMeans2.py")
        elif escolha == '5':
            executar_script("fase5_visualizacao", "analiseIE.py")
        elif escolha == '6':
            print("\nIniciando execução completa do sistema (K-Means + LCA)...")
            executar_script("fase1_pre_processamento", "preprocessamentoMicrodados.py")
            executar_script("fase1_pre_processamento", "preprocessamentoRelatorioSintese.py")
            executar_script("fase2_cruzamento_dados", "analiseGeral.py")
            executar_script("fase2_cruzamento_dados", "analisesTeste.py")
            executar_script("fase4_motor_ia", "motor_agrupamento_triplo.py")
            executar_script("fase4_motor_ia", "mapeador_deficiencias.py")
            executar_script("fase4_motor_ia", "testeKMeans2.py")
            executar_script("fase5_visualizacao", "analiseIE.py")
            # LCA
            lca_script = BASE_DIR / "fase4_motor_ia" / "LCA" / "0_pipeline_principal_LCA.py"
            if lca_script.exists():
                print("\n Executando LCA...")
                subprocess.run([sys.executable, str(lca_script)])
            print("\n PIPELINE COMPLETO FINALIZADO!")
            input("Pressione ENTER para voltar ao menu...")
        elif escolha == '7':
            abrir_dashboard()
        elif escolha == '8':
            print("\nExecutando LCA (Classes Latentes)...")
            lca_path = BASE_DIR / "fase4_motor_ia" / "LCA" / "0_pipeline_principal_LCA.py"
            if lca_path.exists():
                subprocess.run([sys.executable, str(lca_path)])
                print("\n LCA finalizado! Resultados em: arquivosgerados/RESULTADOS_LCA/")
            else:
                print(f"\n Script LCA não encontrado em: {lca_path}")
            input("Pressione ENTER para continuar...")
        elif escolha == '9':
            print("\n Abrindo dashboard direto na aba OpenCode...")
            caminho_dash = BASE_DIR / "fase5_visualizacao" / "dashboard_completo.py"
            if caminho_dash.exists():
                subprocess.run([sys.executable, "-m", "streamlit", "run", str(caminho_dash),
                    "--", "--tab", "opencode"])
            else:
                print(f"\n Dashboard não encontrado.")
            input("Pressione ENTER para continuar...")
        elif escolha == '10':
            print("\n GERADOR HEADLESS OPENCODE")
            print("-" * 50)
            print("Este script executa o OpenCode por trás dos panos para gerar")
            print("um arquivo .md explicativo a partir de um CSV.")
            print()
            caminho_csv = input("Caminho do CSV (ou ENTER para analise_por_ies_curso_enade.csv): ").strip()
            if not caminho_csv:
                caminho_csv = str(BASE_DIR / "arquivosgerados" / "RESULTADOS" / "analise_por_ies_curso_enade.csv")
            caminho_resolvido = Path(caminho_csv)
            if not caminho_resolvido.exists():
                print(f"\n Caminho não encontrado: {caminho_csv}")
            elif caminho_resolvido.is_dir():
                print(f"\n O caminho informado é um DIRETÓRIO, não um arquivo CSV.")
                print(f"   Forneça o caminho completo para um arquivo .csv, ex:")
                csvs = sorted(caminho_resolvido.rglob("*.csv"))
                if csvs:
                    print(f"   {csvs[0].relative_to(BASE_DIR)}")
                else:
                    print(f"   arquivosgerados/RESULTADOS/analise_por_ies_curso_enade.csv")
            else:
                tom_opcoes = {
                    "1": "Didático e acessível (para leigos)",
                    "2": "Técnico e detalhado (para coordenadores)",
                    "3": "Resumo executivo (para diretores)",
                    "4": "Crítico e propositivo (para melhoria)"
                }
                print("\nTom da explicação:")
                for k, v in tom_opcoes.items():
                    print(f"  {k}. {v}")
                tom_escolha = input("Escolha (ENTER = 1): ").strip() or "1"
                tom = tom_opcoes.get(tom_escolha, tom_opcoes["1"])
                foco = input("Foco adicional (opcional, ENTER para vazio): ").strip()
                senha = input("Senha OpenCode (opcional): ").strip()
                pasta_md = BASE_DIR / "arquivosgerados" / "RESULTADOS" / "Dashboards_Markdown"
                nome_output = f"explicacao_opencode_{caminho_resolvido.stem}.md"
                caminho_output = pasta_md / nome_output
                print(f"\n Gerando explicação...")
                print(f"   CSV : {caminho_resolvido}")
                print(f"   TOM : {tom}")
                print(f"   SAÍDA: {caminho_output}")
                script = BASE_DIR / "fase4_motor_ia" / "gerar_explicacao_opencode.py"
                if not script.exists():
                    print(f"\n Script não encontrado: {script}")
                else:
                    cmd = [sys.executable, str(script),
                        "--csv", str(caminho_resolvido),
                        "--tom", tom,
                        "--output", str(caminho_output)]
                    if foco:
                        cmd += ["--foco", foco]
                    if senha:
                        cmd += ["--opencode-password", senha]
                    try:
                        subprocess.run(cmd, check=True)
                        print(f"\n Relatório gerado! Abra o dashboard na aba OpenCode para visualizar o .md.")
                    except subprocess.CalledProcessError:
                        print(f"\n Falha ao executar o script.")
            input("Pressione ENTER para continuar...")
        elif escolha == '0':
            print("Encerrando o sistema...")
            break
        else:
            print("Opção inválida! Tente novamente.")
            input("Pressione ENTER para continuar...")

if __name__ == "__main__":
    menu()