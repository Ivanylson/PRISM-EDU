import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. MAPEAMENTO E FUNÇÕES ÚTEIS
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
    """Remove acentos, espaços e deixa em minúsculo para usar em pastas/arquivos."""
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

# Para manter os resultados organizados, criamos uma subpasta específica
pasta_resultados = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS' / 'prescricoes_basicas_kmeans'
pasta_resultados.mkdir(parents=True, exist_ok=True)

if not caminho_microdados.exists():
    print(f"\n❌ ERRO: Base de dados principal não encontrada em:\n{caminho_microdados}")
    exit()

print("Carregando a base de dados principal (isso pode levar alguns segundos)...")
df_micro = pd.read_csv(caminho_microdados, sep=';', dtype=str)

# Pega todos os cursos (CO_GRUPO) únicos que existem na base lida
grupos_disponiveis = df_micro['CO_GRUPO'].dropna().unique()

print(f"Encontrados {len(grupos_disponiveis)} cursos diferentes na base para analisar.\n")

# =============================================================================
# 3. MOTOR DE INTELIGÊNCIA ARTIFICIAL (GERAÇÃO EM MASSA)
# =============================================================================
for co_grupo in grupos_disponiveis:
    co_grupo_int = int(co_grupo)
    
    # Se o curso não estiver no nosso dicionário, pulamos
    if co_grupo_int not in cursos_map:
        continue
        
    nome_curso = cursos_map[co_grupo_int]
    nome_formatado = formatar_nome(nome_curso)
    
    # 1. Cria a pasta específica para este curso dentro de RESULTADOS
    pasta_do_curso = pasta_resultados / nome_formatado
    pasta_do_curso.mkdir(parents=True, exist_ok=True)
    
    # 2. Carrega o arquivo de síntese deste curso
    caminho_sintese = pasta_sintese / f"{nome_formatado}.csv"
    if not caminho_sintese.exists():
        print(f"⚠️ Aviso: Arquivo de síntese '{nome_formatado}.csv' não encontrado. Pulando {nome_curso}...")
        continue
        
    # CORREÇÃO: Lê o ficheiro de síntese com sep=';' para evitar quebra de colunas
    df_sintese = pd.read_csv(caminho_sintese, sep=',')
    if 'POSIÇÃO' in df_sintese.columns:
        df_sintese.rename(columns={'POSIÇÃO': 'QUESTAO'}, inplace=True)
    df_sintese['QUESTAO'] = 'Q' + df_sintese['QUESTAO'].astype(str).str.strip()
    
    # Detecta dinamicamente todas as colunas que começam com 'OC'
    colunas_oc = [col for col in df_sintese.columns if str(col).startswith('OC')]
    
    # Filtra os alunos apenas deste curso
    df_curso = df_micro[df_micro['CO_GRUPO'] == str(co_grupo)].copy()
    ies_disponiveis = df_curso['CO_IES'].unique()
    
    print(f"Processando: {nome_curso} | {len(ies_disponiveis)} Instituições encontradas.")
    
    # 3. Roda a IA para CADA IES dentro deste curso
    for ies in ies_disponiveis:
        df_alvo = df_curso[df_curso['CO_IES'] == ies].copy()
        
        # Regra de Segurança: IA precisa de volume de dados. Turmas muito pequenas são ignoradas.
        if len(df_alvo) < 5:
            continue
            
        # Prepara as questões (Features)
        cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df_alvo.columns]
        for col in cols_questoes:
            df_alvo[col] = np.where(df_alvo[col] == '1', 1, 0)
            
        X = df_alvo[cols_questoes]
        
        # Define o número de perfis: 3, ou menos se a turma for pequena (entre 5 e 9 alunos)
        num_perfis = 3 if len(df_alvo) >= 10 else 2
        
        # Treina a IA
        kmeans = KMeans(n_clusters=num_perfis, random_state=42, n_init=10)
        df_alvo['PERFIL_IA'] = kmeans.fit_predict(X)
        
        # Analisa os resultados
        resultados_ia = []
        for perfil in range(num_perfis):
            alunos_do_perfil = df_alvo[df_alvo['PERFIL_IA'] == perfil]
            qtd_alunos = len(alunos_do_perfil)
            
            if qtd_alunos == 0: continue
            
            taxa_acertos = alunos_do_perfil[cols_questoes].mean()
            piores_questoes = taxa_acertos.nsmallest(3).index.tolist()
            
            for questao in piores_questoes:
                taxa_erro = (1 - taxa_acertos[questao]) * 100
                info_questao = df_sintese[df_sintese['QUESTAO'] == questao]
                
                if not info_questao.empty:
                    # Coleta a competência
                    competencia = info_questao['COMPETÊNCIAS'].values[0] if 'COMPETÊNCIAS' in info_questao.columns else "Não especificada"
                    
                    # Coleta todos os OCs disponíveis para esta questão
                    lista_ocs = []
                    for coluna in colunas_oc:
                        valor_oc = info_questao[coluna].values[0]
                        if pd.notna(valor_oc) and str(valor_oc).strip() != "":
                            lista_ocs.append(str(valor_oc).strip())
                    
                    # Une os OCs encontrados usando " | " como separador
                    texto_ocs = " | ".join(lista_ocs) if lista_ocs else "Não especificado"
                    
                    # Adiciona ao relatório
                    resultados_ia.append({
                        'IES': ies,
                        'CURSO': nome_curso,
                        'GRUPO_DE_ALUNOS': f'Perfil {perfil + 1} ({qtd_alunos} alunos)',
                        'QUESTAO_CRITICA': questao,
                        'TAXA_DE_ERRO_%': round(taxa_erro, 2),
                        'O_QUE_PRECISAM_MELHORAR_OCs': texto_ocs,
                        'COMPETENCIA_FALTANTE': competencia,
                        'ACAO_RECOMENDADA': f'Revisar conceitos de: {texto_ocs}'
                    })
                    
        # Salva o resultado DESTA IES na pasta DO CURSO correspondente
        if resultados_ia:
            df_prescricao = pd.DataFrame(resultados_ia)
            nome_arquivo_saida = f'ia_prescricao_ies{ies}_{nome_formatado}.csv'
            caminho_saida = pasta_do_curso / nome_arquivo_saida
            df_prescricao.to_csv(caminho_saida, sep=';', index=False, encoding='utf-8-sig')

print(f"\n{'='*70}")
print("✅ PROCESSAMENTO EM MASSA (K-MEANS BÁSICO) CONCLUÍDO!")
print(f"Todas as análises foram organizadas por curso na pasta:\n{pasta_resultados}")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("PROCESSAMENTO EM MASSA CONCLUÍDO COM SUCESSO!")
print(f"Todas as análises foram organizadas por curso na pasta:\n{pasta_resultados}")
print(f"{'='*70}")


# Qual e a questão que foi impactante naquele grupo 
#quantas questões de cada grupo
# indice de silhueta, Importante: mudar a ideia de 3 grupos. 
# Qual e questão de cada grupo a resposta e média da resposta, para cada item fazer a media das respostas
# Média do resultado dos quantativos de cada resposta de cada grupo
# O que vai mostrar se o grupo e intermediario, avançado ou fácil ou não
