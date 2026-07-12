import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
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

# Pasta onde os Rankings serão salvos
pasta_relatorios_ies = DIRETORIO_ATUAL / 'arquivosgerados' / 'RESULTADOS' / 'relatorioIES'
pasta_relatorios_ies.mkdir(parents=True, exist_ok=True)

if not caminho_microdados.exists():
    print(f"ERRO: Base de dados principal não encontrada em:\n{caminho_microdados}")
    exit()

print("A carregar a base de dados principal...")
df_micro = pd.read_csv(caminho_microdados, sep=';', dtype=str)
grupos_disponiveis = df_micro['CO_GRUPO'].dropna().unique()

print("A gerar os Rankings Gerais de Piores Questões por IES...\n")

# =============================================================================
# 3. MOTOR DE GERAÇÃO: RANKING GERAL IES (HTML)
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
    <title>Ranking Crítico - {nome_curso}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 40px; color: #333; background-color: #f4f7f6; }}
        .container {{ background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #c0392b; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        .resumo {{ background-color: #fdf2e9; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 16px; border-left: 5px solid #e67e22; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        th, td {{ padding: 12px 15px; border: 1px solid #ddd; text-align: left; vertical-align: middle; }}
        th {{ background-color: #2c3e50; color: #ffffff; text-transform: uppercase; font-size: 13px; letter-spacing: 0.5px; position: sticky; top: 0; }}
        .linha-ies {{ background-color: #ecf0f1; font-weight: bold; font-size: 16px; text-align: center; color: #2c3e50; }}
        tr:hover {{ background-color: #f1f5f8; transition: background-color 0.3s; }}
        .erro-critico {{ color: #c0392b; font-weight: bold; font-size: 15px; text-align: center; }}
        .centralizado {{ text-align: center; }}
        .posicao {{ font-weight: bold; color: #e67e22; }}
        .footer {{ margin-top: 40px; font-size: 13px; color: #7f8c8d; text-align: center; border-top: 1px solid #ddd; padding-top: 15px; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚨 Ranking das Piores Questões por IES: {nome_curso}</h1>
        
        <div class="resumo">
            <p><strong>Objetivo:</strong> Este relatório consolida o desempenho geral de todos os alunos (agrupando todos os perfis) para identificar os <strong>3 maiores gargalos absolutos</strong> de cada Instituição de Ensino Superior.</p>
            <p><strong>Total de Instituições Analisadas:</strong> {len(ies_disponiveis)} | <strong>Total de Alunos:</strong> {len(df_curso)}</p>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th class="centralizado" style="width: 10%;">Código IES</th>
                    <th class="centralizado" style="width: 15%;">Posição / Questão</th>
                    <th class="centralizado" style="width: 10%;">Erro Geral (%)</th>
                    <th style="width: 30%;">Objetos de Conhecimento (Matérias)</th>
                    <th style="width: 35%;">Competência Exigida</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Roda a análise para CADA IES do curso
    for ies in ies_disponiveis:
        df_ies = df_curso[df_curso['CO_IES'] == ies].copy()
        
        if len(df_ies) < 5:
            continue
            
        cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df_ies.columns]
        for col in cols_questoes:
            df_ies[col] = np.where(df_ies[col] == '1', 1, 0)
            
        # Calcula a taxa de erro GERAL da IES (todos os alunos juntos)
        taxa_erro_geral = (1 - df_ies[cols_questoes].mean()) * 100
        
        # Pega as 3 piores questões (maiores taxas de erro)
        piores_questoes = taxa_erro_geral.nlargest(3)
        
        if piores_questoes.empty:
            continue
            
        # Cria a primeira linha com o ROWSPAN (para a célula da IES ficar grande e bonita)
        primeira_linha = True
        posicao = 1
        
        for questao, erro_percentual in piores_questoes.items():
            info_q = df_sintese[df_sintese['QUESTAO'] == questao]
            
            comp = "Não especificada"
            texto_ocs = "Não especificado"
            
            if not info_q.empty:
                comp = limpar_texto_html(info_q['COMPETÊNCIAS'].values[0] if 'COMPETÊNCIAS' in info_q.columns else "Não especificada")
                
                lista_ocs = []
                for coluna in colunas_oc:
                    valor_oc = info_q[coluna].values[0]
                    if pd.notna(valor_oc) and str(valor_oc).strip() != "":
                        lista_ocs.append(str(valor_oc).strip())
                        
                texto_ocs = " + ".join(lista_ocs) if lista_ocs else "Não especificado"
                texto_ocs = limpar_texto_html(texto_ocs)
            
            erro_formatado = round(erro_percentual, 1)
            
            if primeira_linha:
                # Na primeira linha, colocamos a célula do Código IES com rowspan=3
                html_content += f"""
                <tr>
                    <td rowspan="3" class="linha-ies">{ies}<br><span style="font-size: 12px; font-weight: normal; color: #7f8c8d;">({len(df_ies)} alunos)</span></td>
                    <td class="centralizado"><span class="posicao">1º Pior</span><br><b>{questao}</b></td>
                    <td class="erro-critico">{erro_formatado}%</td>
                    <td>{texto_ocs}</td>
                    <td>{comp}</td>
                </tr>"""
                primeira_linha = False
            else:
                # Nas linhas seguintes, não precisamos colocar a célula da IES novamente
                html_content += f"""
                <tr>
                    <td class="centralizado"><span class="posicao">{posicao}º Pior</span><br><b>{questao}</b></td>
                    <td class="erro-critico">{erro_formatado}%</td>
                    <td>{texto_ocs}</td>
                    <td>{comp}</td>
                </tr>"""
            posicao += 1
            
    # Fechamento do HTML
    html_content += """
            </tbody>
        </table>
        <div class="footer">Relatório de Diagnóstico Macro gerado automaticamente.</div>
    </div>
</body>
</html>
"""
    
    # Guarda o ficheiro na mesma pasta relatorioIES, mas com nome diferente
    nome_ficheiro_html = f"{nome_formatado}_ranking_questoes_criticas.html"
    caminho_html = pasta_relatorios_ies / nome_ficheiro_html
    
    with open(caminho_html, 'w', encoding='utf-8') as file:
        file.write(html_content)
        
    print(f"✅ Ranking HTML gerado: {nome_ficheiro_html}")

print(f"\n{'='*75}")
print(f"🚀 PROCESSO CONCLUÍDO! Todos os rankings gerais por IES foram salvos em:\n{pasta_relatorios_ies}")
print(f"{'='*75}")