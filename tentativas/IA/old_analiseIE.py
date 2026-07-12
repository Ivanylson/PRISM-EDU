import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. MAPEAMENTO DE CURSOS E FUNÇÕES ÚTEIS
# =============================================================================
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

def formatar_nome(nome):
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_')

def limpar_texto_html(texto):
    """Garante que o texto não tenha caracteres estranhos e fique limpo para o HTML."""
    if pd.isna(texto):
        return "Não especificado"
    texto = str(texto).replace('\n', ' ').strip()
    return texto

# =============================================================================
# 2. CONFIGURAÇÕES DE CAMINHOS
# =============================================================================
DIRETORIO_ATUAL = Path(__file__).resolve().parent
caminho_microdados = DIRETORIO_ATUAL / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_sintese = DIRETORIO_ATUAL / 'arquivosgerados' / 'resultadofinal_relatoriosintese' / 'csv'

# MUDANÇA AQUI: Nova pasta exclusiva para os relatórios por IES
pasta_relatorios_ies = DIRETORIO_ATUAL / 'arquivosgerados' / 'RESULTADOS' / 'relatorioIES'
pasta_relatorios_ies.mkdir(parents=True, exist_ok=True)

if not caminho_microdados.exists():
    print(f"ERRO: Base de dados principal não encontrada em:\n{caminho_microdados}")
    exit()

print("A carregar a base de dados principal (isto pode demorar alguns segundos)...")
df_micro = pd.read_csv(caminho_microdados, sep=';', dtype=str)
grupos_disponiveis = df_micro['CO_GRUPO'].dropna().unique()

print(f"Encontrados {len(grupos_disponiveis)} cursos diferentes na base. A gerar os Super Relatórios HTML por IES...\n")

# =============================================================================
# 3. MOTOR DE GERAÇÃO: TABELAS DETALHADAS POR IES (EM HTML)
# =============================================================================
for co_grupo in grupos_disponiveis:
    co_grupo_int = int(co_grupo)
    
    if co_grupo_int not in cursos_map:
        continue
        
    nome_curso = cursos_map[co_grupo_int]
    nome_formatado = formatar_nome(nome_curso)
    
    caminho_sintese = pasta_sintese / f"{nome_formatado}.csv"
    if not caminho_sintese.exists():
        continue
        
    df_sintese = pd.read_csv(caminho_sintese, sep=',')
    if 'POSIÇÃO' in df_sintese.columns:
        df_sintese.rename(columns={'POSIÇÃO': 'QUESTAO'}, inplace=True)
    df_sintese['QUESTAO'] = 'Q' + df_sintese['QUESTAO'].astype(str).str.strip()
    
    colunas_oc = [col for col in df_sintese.columns if str(col).startswith('OC')]
    
    df_curso = df_micro[df_micro['CO_GRUPO'] == str(co_grupo)].copy()
    
    if len(df_curso) < 10: 
        continue 
        
    ies_disponiveis = df_curso['CO_IES'].dropna().unique()
    
    # -------------------------------------------------------------------------
    # CABEÇALHO E ESTILOS CSS DO HTML
    # -------------------------------------------------------------------------
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório IES - {nome_curso}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 40px; color: #333; background-color: #f4f7f6; }}
        .container {{ background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        .resumo {{ background-color: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 16px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        th, td {{ padding: 12px 15px; border: 1px solid #ddd; text-align: left; vertical-align: middle; }}
        th {{ background-color: #34495e; color: #ffffff; text-transform: uppercase; font-size: 13px; letter-spacing: 0.5px; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f5f8; transition: background-color 0.3s; }}
        .gap-critico {{ color: #e74c3c; font-weight: bold; font-size: 15px; text-align: center; }}
        .centralizado {{ text-align: center; }}
        .destaque {{ font-weight: bold; color: #2c3e50; }}
        .footer {{ margin-top: 40px; font-size: 13px; color: #7f8c8d; text-align: center; border-top: 1px solid #ddd; padding-top: 15px; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ Relatório de Diagnóstico por IES: {nome_curso}</h1>
        
        <div class="resumo">
            <p><strong>Total de Instituições Analisadas:</strong> {len(ies_disponiveis)}</p>
            <p><strong>Total de Alunos Analisados no Curso:</strong> {len(df_curso)}</p>
        </div>
        
        <h2>📋 Tabela Geral de Deficiências Críticas por Instituição</h2>
        <p>A tabela abaixo destrincha, para <strong>cada Instituição de Ensino Superior (IES)</strong>, o resultado do agrupamento por Inteligência Artificial (K-Means). Ela aponta exatamente quais são as maiores deficiências relativas (GAPs) de cada perfil dentro daquela IES específica, indicando as competências e os objetos de conhecimento que exigem intervenção pedagógica urgente.</p>
        
        <table>
            <thead>
                <tr>
                    <th class="centralizado">Código IES</th>
                    <th>Perfil de Alunos</th>
                    <th class="centralizado">Questão Crítica</th>
                    <th class="centralizado">Erro Absoluto</th>
                    <th class="centralizado">GAP (vs IES)</th>
                    <th>Objetos de Conhecimento (OCs)</th>
                    <th>Competência em Déficit</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Roda a IA para CADA IES do curso
    for ies in ies_disponiveis:
        df_ies = df_curso[df_curso['CO_IES'] == ies].copy()
        
        # Ignora instituições com turmas muito pequenas
        if len(df_ies) < 5:
            continue
            
        cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df_ies.columns]
        for col in cols_questoes:
            df_ies[col] = np.where(df_ies[col] == '1', 1, 0)
            
        X = df_ies[cols_questoes]
        
        num_perfis = 3 if len(df_ies) >= 10 else 2
        
        kmeans = KMeans(n_clusters=num_perfis, random_state=42, n_init=10)
        df_ies['PERFIL_IA'] = kmeans.fit_predict(X)
        
        media_geral_ies = X.mean()
        
        perfis_data = []
        for perfil in range(num_perfis):
            alunos = df_ies[df_ies['PERFIL_IA'] == perfil]
            if len(alunos) == 0: continue
            nota_media = alunos[cols_questoes].sum(axis=1).mean()
            perfis_data.append((perfil, nota_media, alunos))
            
        perfis_data.sort(key=lambda x: x[1], reverse=True)
        
        titulos_perfis = ["Alto Desempenho", "Intermediário", "Risco Crítico"]
        
        for rank, (perfil_id, nota_media, alunos_grupo) in enumerate(perfis_data):
            nome_perfil = titulos_perfis[rank] if num_perfis == 3 else ("Alto Desempenho" if rank == 0 else "Risco Crítico")
            texto_perfil = f"<span class='destaque'>{nome_perfil}</span><br>({len(alunos_grupo)} alunos)"
            
            taxa_acertos_grupo = alunos_grupo[cols_questoes].mean()
            diferenca_para_media = taxa_acertos_grupo - media_geral_ies
            
            piores_gaps = diferenca_para_media.nsmallest(2).index.tolist()
            
            for questao in piores_gaps:
                gap_percentual = round(diferenca_para_media[questao] * 100, 1)
                erro_absoluto = round((1 - taxa_acertos_grupo[questao]) * 100, 1)
                
                info_q = df_sintese[df_sintese['QUESTAO'] == questao]
                
                if not info_q.empty:
                    comp = info_q['COMPETÊNCIAS'].values[0] if 'COMPETÊNCIAS' in info_q.columns else "Não especificada"
                    comp = limpar_texto_html(comp)
                    
                    lista_ocs = []
                    for coluna in colunas_oc:
                        valor_oc = info_q[coluna].values[0]
                        if pd.notna(valor_oc) and str(valor_oc).strip() != "":
                            lista_ocs.append(str(valor_oc).strip())
                            
                    texto_ocs = " + ".join(lista_ocs) if lista_ocs else "Não especificado"
                    texto_ocs = limpar_texto_html(texto_ocs)
                    
                    # Constrói a linha da tabela em HTML
                    linha_html = f"""
                <tr>
                    <td class="centralizado destaque">{ies}</td>
                    <td>{texto_perfil}</td>
                    <td class="centralizado destaque">{questao}</td>
                    <td class="centralizado">{erro_absoluto}%</td>
                    <td class="gap-critico">{gap_percentual}%</td>
                    <td>{texto_ocs}</td>
                    <td>{comp}</td>
                </tr>"""
                    html_content += linha_html
    
    # Fechamento do HTML
    html_content += """
            </tbody>
        </table>
        
        <div class="footer">
            Relatório gerado automaticamente por motor de Inteligência Artificial Educacional com base no algoritmo K-Means.
        </div>
    </div>
</body>
</html>
"""
    
    # Guarda o ficheiro na nova pasta relatorioIES
    nome_ficheiro_html = f"{nome_formatado}_relatorioIESgeral.html"
    caminho_html = pasta_relatorios_ies / nome_ficheiro_html
    
    with open(caminho_html, 'w', encoding='utf-8') as file:
        file.write(html_content)
        
    print(f"✅ Dashboard HTML gerado: {nome_ficheiro_html}")

print(f"\n{'='*75}")
print(f"🚀 PROCESSO CONCLUÍDO! Todas as tabelas HTML por IES foram salvas em:\n{pasta_relatorios_ies}")
print(f"{'='*75}")