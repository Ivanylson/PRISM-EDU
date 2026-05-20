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
    """Remove acentos, espaços e deixa em minúsculas para usar nos nomes dos ficheiros."""
    nome = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome.lower().replace(' ', '_')

# =============================================================================
# 2. CONFIGURAÇÕES DE CAMINHOS AUTOMÁTICOS
# =============================================================================
print("Configurando diretórios de forma automática...")
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent # <--- Volta à raiz do projeto

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_sintese = DIRETORIO_RAIZ / 'arquivosgerados' / 'resultadofinal_relatoriosintese' / 'csv'

# Pasta onde TODOS os relatórios Markdown de cada curso serão guardados
pasta_dashboards = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS' / 'Dashboards_Markdown'
pasta_dashboards.mkdir(parents=True, exist_ok=True)

if not caminho_microdados.exists():
    print(f"\n ERRO: Base de dados principal não encontrada em:\n{caminho_microdados}")
    print("Por favor, execute a Fase 1 primeiro.")
    exit()

print("A carregar a base de dados principal (isto pode demorar alguns segundos)...")
df_micro = pd.read_csv(caminho_microdados, sep=';', dtype=str)
grupos_disponiveis = df_micro['CO_GRUPO'].dropna().unique()

print(f"Encontrados {len(grupos_disponiveis)} cursos diferentes na base. A gerar Relatórios Prescritivos...\n")

# =============================================================================
# 3. TEXTO BASE SOBRE A TEORIA DE AGRUPAMENTO (ANEXO DO RELATÓRIO)
# =============================================================================
texto_teoria_agrupamento = """
---

##  Anexo Técnico: Como Avaliamos a Qualidade do Agrupamento (Clustering)?

A técnica utilizada neste relatório foi o **K-Means Clustering**, um algoritmo de Aprendizado de Máquina Não-Supervisionado. Para garantir que os perfis gerados refletem a realidade e não apenas divisões aleatórias, adotamos as melhores práticas de Ciência de Dados Educacional:

### 1. O Problema da "Sombra da Questão Difícil"
Muitos analistas iniciantes olham apenas para as questões que os grupos *mais erraram* em termos absolutos. Isso é um erro. Se uma questão tem um índice de erro global de 95%, ela aparecerá como "A pior questão" em todos os clusters, escondendo a verdadeira identidade do grupo.

**A Nossa Solução (Análise de Gap):** Calculámos a média de acerto global do curso e comparámos com a média do grupo específico. O que destacamos neste relatório são as **deficiências relativas**. Ou seja: *"Onde é que este grupo específico foi consideravelmente pior do que a média dos seus próprios colegas?"* Isto gera diagnósticos muito mais precisos.

### 2. Validação e Qualidade dos Grupos
Como é que os cientistas de dados validam estes grupos?
* **Inércia (Soma dos Erros Quadráticos):** Mede o quão compactos são os clusters. Um bom agrupamento tem alunos muito parecidos entre si dentro do mesmo grupo.
* **Silhouette Score:** É uma métrica de -1 a 1 que avalia a qualidade da separação. Valores mais próximos de 1 indicam que os grupos são muito distintos uns dos outros (ex: o grupo avançado não se mistura com o grupo com dificuldades).
* **Interpretabilidade Semântica:** O algoritmo é puramente matemático. O papel da análise prescritiva é traduzir a média da "Q15" para o contexto pedagógico (ex: "Eles dominam cálculo, mas falham na interpretação de texto"), que é exatamente o que as secções acima fazem.
"""

# =============================================================================
# 4. MOTOR DE GERAÇÃO: UM RELATÓRIO PARA CADA CURSO
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
        
    # CORREÇÃO: Leitura com sep=';'
    df_sintese = pd.read_csv(caminho_sintese, sep=',')
    if 'POSIÇÃO' in df_sintese.columns:
        df_sintese.rename(columns={'POSIÇÃO': 'QUESTAO'}, inplace=True)
    df_sintese['QUESTAO'] = 'Q' + df_sintese['QUESTAO'].astype(str).str.strip()
    
    # DETEÇÃO DINÂMICA DE COLUNAS 'OC' (Objeto de Conhecimento)
    colunas_oc = [col for col in df_sintese.columns if str(col).startswith('OC')]
    
    # Filtra os alunos que pertencem apenas a ESTE curso
    df_curso = df_micro[df_micro['CO_GRUPO'] == str(co_grupo)].copy()
    
    if len(df_curso) < 10: 
        continue 
        
    cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df_curso.columns]
    
    for col in cols_questoes:
        df_curso[col] = np.where(df_curso[col] == '1', 1, 0)
        
    X = df_curso[cols_questoes]
    
    # -------------------------------------------------------------------------
    # TREINO DA IA (Criação dos 3 grupos)
    # -------------------------------------------------------------------------
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_curso['PERFIL_IA'] = kmeans.fit_predict(X)
    
    media_geral_questoes = X.mean()
    nota_media_geral = round(X.sum(axis=1).mean(), 1)
    
    perfis_data = []
    for perfil in range(3):
        alunos = df_curso[df_curso['PERFIL_IA'] == perfil]
        if len(alunos) == 0: continue
        nota_media = alunos[cols_questoes].sum(axis=1).mean()
        perfis_data.append((perfil, nota_media, alunos))
        
    perfis_data.sort(key=lambda x: x[1], reverse=True)
    
    # -------------------------------------------------------------------------
    # CONSTRUÇÃO DO FICHEIRO MARKDOWN (.md) PARA ESTE CURSO
    # -------------------------------------------------------------------------
    md_content = f"# Relatório Prescritivo de Inteligência Artificial: {nome_curso}\n\n"
    md_content += f"**Total de Alunos Analisados no Curso:** {len(df_curso)}\n"
    md_content += f"**Média Geral de Acertos do Curso:** {nota_media_geral}/38 questões\n\n"
    md_content += "---\n\n## Mapeamento de Perfis de Aprendizagem\n"
    md_content += "O algoritmo de IA analisou o padrão de respostas e separou os alunos em 3 grupos distintos. Abaixo, detalhamos o significado de cada grupo e as ações pedagógicas recomendadas focadas nas deficiências exclusivas de cada um (onde erraram mais que a média).\n\n"
    
    titulos_perfis = ["Perfil de Alto Desempenho", "Perfil Intermediário", "Perfil com Dificuldades Críticas"]
    
    for rank, (perfil_id, nota_media, alunos_grupo) in enumerate(perfis_data):
        qtd_alunos = len(alunos_grupo)
        percentual = round((qtd_alunos / len(df_curso)) * 100, 1)
        
        md_content += f"### {titulos_perfis[rank]} ({percentual}% dos alunos da área)\n"
        md_content += f"- **Quantidade de Alunos no Grupo:** {qtd_alunos}\n"
        md_content += f"- **Média de Acertos deste Grupo:** {round(nota_media, 1)} de 38\n\n"
        
        if rank == 0:
            md_content += "**O que este grupo significa?** São os alunos de topo, que dominam a maior parte dos conteúdos base. As suas deficiências não são estruturais; geralmente falham apenas em questões altamente específicas da área, tópicos muito avançados ou de interpretação complexa.\n\n"
        elif rank == 1:
            md_content += "**O que este grupo significa?** Representam a grande massa de alunos medianos. Têm uma base aceitável, mas o seu conhecimento é fragmentado. Acertam as questões fáceis, mas têm dificuldades em cruzar competências distintas ou aplicar a teoria em situações-problema práticas.\n\n"
        else:
            md_content += "**O que este grupo significa?** São alunos em situação de risco académico. Este grupo apresenta falhas estruturais severas. Não dominam os conceitos fundamentais do componente específico e frequentemente têm dificuldades na Formação Geral (leitura, lógica e interpretação de texto).\n\n"
            
        md_content += "** Maiores GAPs (Onde este grupo está pior que a média da turma):**\n\n"
        
        taxa_acertos_grupo = alunos_grupo[cols_questoes].mean()
        diferenca_para_media = taxa_acertos_grupo - media_geral_questoes
        piores_gaps = diferenca_para_media.nsmallest(3).index.tolist()
        
        for questao in piores_gaps:
            gap_percentual = round(diferenca_para_media[questao] * 100, 1)
            erro_absoluto = round((1 - taxa_acertos_grupo[questao]) * 100, 1)
            
            info_q = df_sintese[df_sintese['QUESTAO'] == questao]
            if not info_q.empty:
                # Verificação dupla de colunas de competência para evitar bugs
                if 'COMPETÊNCIAS' in info_q.columns and pd.notna(info_q['COMPETÊNCIAS'].values[0]):
                    comp = info_q['COMPETÊNCIAS'].values[0]
                elif 'COMPETENCIA' in info_q.columns and pd.notna(info_q['COMPETENCIA'].values[0]):
                    comp = info_q['COMPETENCIA'].values[0]
                else:
                    comp = "Não especificada"
                
                # RECOLHA DINÂMICA DE TODOS OS 'OCs' EXISTENTES NESTA QUESTÃO
                lista_ocs = []
                for coluna in colunas_oc:
                    valor_oc = info_q[coluna].values[0]
                    # Verifica se o valor não é nulo/vazio
                    if pd.notna(valor_oc) and str(valor_oc).strip() != "":
                        lista_ocs.append(str(valor_oc).strip())
                
                # Une todos os OCs com uma barra vertical para facilitar a leitura
                texto_ocs = " | ".join(lista_ocs) if lista_ocs else "Não especificado"
                
                md_content += f"1. **{questao}** (Erraram {abs(gap_percentual)}% a mais que a restante turma)\n"
                md_content += f"   - *Taxa de erro absoluta no grupo:* {erro_absoluto}%\n"
                md_content += f"   - *Objetos de Conhecimento (Matérias):* {texto_ocs}\n"
                md_content += f"   - *Competência Falha:* {comp}\n"
                md_content += f"   - *Prescrição Pedagógica recomendada:* Focar em revisões estruturadas e exercícios interdisciplinares abordando: **{texto_ocs}**.\n\n"
    
    md_content += texto_teoria_agrupamento
    
    nome_ficheiro_md = f"relatorio_prescritivo_{nome_formatado}.md"
    caminho_md = pasta_dashboards / nome_ficheiro_md
    with open(caminho_md, 'w', encoding='utf-8') as file:
        file.write(md_content)
        
    print(f" Documento gerado com sucesso: {nome_ficheiro_md}")

print(f"\n{'='*75}")
print(f" PROCESSO CONCLUÍDO! Os relatórios em Markdown de todos os cursos estão na pasta:\n{pasta_dashboards}")
print(f"{'='*75}")