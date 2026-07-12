import pandas as pd
import numpy as np
from pathlib import Path
import re
import unicodedata
import warnings
import csv

warnings.filterwarnings('ignore')

# =============================================================================
# 1. MAPEAMENTO DE CURSOS E CONFIGURAÇÕES
# =============================================================================
print("Inicializando Fase 6: Diagnóstico Cognitivo via Matriz-Q (Leitura Direta de XLSX)...")

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

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_sintese = DIRETORIO_RAIZ / 'arquivosgerados' / 'resultadofinal_relatoriosintese' / 'csv'

pasta_resultados_cdm = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_FASE6_CDM_XLSX'
pasta_resultados_cdm.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. FUNÇÕES AUXILIARES DE LIMPEZA E LEITURA ROBUSTA
# =============================================================================
def extrair_numero_questao(texto):
    numeros = re.findall(r'\d+', str(texto))
    return int(numeros[0]) if numeros else 0

def formatar_nome_arquivo(nome):
    nome_sem_acento = ''.join(ch for ch in unicodedata.normalize('NFKD', nome) if not unicodedata.combining(ch))
    return nome_sem_acento.lower().replace(' ', '_')

def detectar_separador(caminho_arquivo):
    encodings_para_testar = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for enc in encodings_para_testar:
        try:
            with open(caminho_arquivo, 'r', encoding=enc) as f:
                amostra = f.read(4096)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(amostra, delimiters=';,')
                return dialect.delimiter
        except:
            continue
    return ';' 

def ler_arquivo_sintese(caminho):
    """Lê o arquivo de síntese, dando preferência absoluta para arquivos Excel (.xlsx)"""
    if caminho.suffix.lower() == '.xlsx':
        # Se for Excel, lê perfeitamente as células, ignorando problemas de vírgula e quebra de linha
        return pd.read_excel(caminho, engine='openpyxl')
    else:
        # Se for CSV, usa engine='python' para evitar o erro de Buffer Overflow do C
        sep = detectar_separador(caminho)
        encodings_para_testar = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        for enc in encodings_para_testar:
            try:
                return pd.read_csv(caminho, sep=sep, encoding=enc, engine='python', on_bad_lines='skip', quotechar='"')
            except Exception:
                continue
        raise ValueError(f"Impossível ler o arquivo CSV: {caminho}")

def limpar_texto_matriz(texto):
    if pd.isna(texto) or str(texto).lower() == 'nan':
        return ""
    t = str(texto).strip()
    t = t.replace('"', '').replace('\n', ' ').replace('\r', '')
    t = re.sub(r'[,.\s;]+$', '', t)
    return ' '.join(t.split())

# =============================================================================
# 3. CARREGAMENTO DA BASE PRINCIPAL (Microdados)
# =============================================================================
print("\nA ler microdados gerais (isto pode demorar um pouco)...")
try:
    sep_microdados = detectar_separador(caminho_microdados)
    encodings_micro = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    df_microdados_full = None
    for enc in encodings_micro:
        try:
            df_microdados_full = pd.read_csv(caminho_microdados, sep=sep_microdados, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
            
    if df_microdados_full is None:
        raise ValueError("Falha de encoding ao ler microdados.")
        
    df_microdados_full = df_microdados_full[df_microdados_full['TP_PR_GER'] > 0]
except Exception as e:
    print(f"ERRO FATAL: Não foi possível ler os microdados. Detalhes: {e}")
    exit()

# =============================================================================
# 4. MOTOR DE PROCESSAMENTO EM LOTE
# =============================================================================
resultados_gerais = []

for codigo_curso, nome_curso in cursos_map.items():
    print(f"\n{'-'*70}")
    print(f"PROCESSANDO: {codigo_curso} - {nome_curso}")
    print(f"{'-'*70}")
    
    try:
        nome_base = formatar_nome_arquivo(nome_curso)
        
        # PROCURA AGORA PELOS ARQUIVOS CORRETOS (.XLSX) COMO PRIORIDADE
        possiveis_nomes = [
            f"{nome_base}_anexoIX.xlsx", 
            f"{nome_base}_anexoIX.csv",
            f"{nome_base}_anexoIX.xlsx - Table 1.csv",
            f"{nome_base}_conteudo.csv", 
            f"{nome_base}_sintese.csv",
            f"{nome_base}.csv"
        ]
        
        caminho_arq_sintese = None
        for pn in possiveis_nomes:
            if (pasta_sintese / pn).exists():
                caminho_arq_sintese = pasta_sintese / pn
                break
                
        if not caminho_arq_sintese:
            print(f"[AVISO] Nenhum arquivo encontrado para {nome_curso}. Buscamos por _anexoIX.xlsx e outros. Pulando.")
            resultados_gerais.append({'CURSO': nome_curso, 'ALUNOS': 0, 'STATUS': 'ARQUIVO_AUSENTE'})
            continue
            
        print(f"Lendo base de competências: {caminho_arq_sintese.name}")
            
        df_curso = df_microdados_full[df_microdados_full['CO_CURSO'] == codigo_curso].copy()
        if df_curso.empty and 'CO_GRUPO' in df_microdados_full.columns:
             df_curso = df_microdados_full[df_microdados_full['CO_GRUPO'] == codigo_curso].copy()

        if df_curso.empty:
            print(f"[AVISO] Nenhum aluno encontrado para o curso {codigo_curso}. Pulando.")
            resultados_gerais.append({'CURSO': nome_curso, 'ALUNOS': 0, 'STATUS': 'SEM_ALUNOS'})
            continue
            
        colunas_q = [f'Q{i}' for i in range(1, 39)]
        df_respostas = df_curso[['ALUNO'] + colunas_q].copy()
        
        for col in colunas_q:
            df_respostas[col] = df_respostas[col].apply(lambda x: 1 if str(x).strip() == '1' else 0)

        # --- PASSO B: LEITURA SEGURA (XLSX ou CSV) ---
        df_sintese = ler_arquivo_sintese(caminho_arq_sintese)
        
        # Limpeza robusta das colunas
        df_sintese.columns = [str(c).strip().upper().replace('\n', ' ').replace('\r', '') for c in df_sintese.columns]
        
        col_posicao = next((c for c in df_sintese.columns if 'POSI' in c), None)
        if not col_posicao: 
            raise ValueError(f"Coluna de posição não encontrada. Colunas lidas foram: {list(df_sintese.columns)}")
            
        df_sintese['NUM_QUESTAO'] = df_sintese[col_posicao].apply(extrair_numero_questao)
        df_sintese = df_sintese[df_sintese['NUM_QUESTAO'] > 0].sort_values('NUM_QUESTAO')
        
        colunas_oc = [c for c in df_sintese.columns if 'OC' in c and 'UNIFICADO' not in c]
        
        todas_disciplinas = set()
        for col in colunas_oc:
            for val in df_sintese[col].dropna():
                v_clean = limpar_texto_matriz(val)
                if v_clean:
                    todas_disciplinas.add(v_clean)
                    
        disciplinas_principais = sorted(list(todas_disciplinas))
        
        habilidades_unicas = ['Conhecimentos Gerais'] + disciplinas_principais
        n_habilidades = len(habilidades_unicas)
        n_questoes = 38
        Q_matrix = np.zeros((n_questoes, n_habilidades))
        
        for idx, row in df_sintese.iterrows():
            q_idx = row['NUM_QUESTAO'] - 1
            if q_idx < 38:
                tem_skill = False
                for col in colunas_oc:
                    val_clean = limpar_texto_matriz(row.get(col, ''))
                    
                    if val_clean in disciplinas_principais:
                        hab_idx = habilidades_unicas.index(val_clean)
                        Q_matrix[q_idx, hab_idx] = 1
                        tem_skill = True
                
                if not tem_skill:
                    Q_matrix[q_idx, 0] = 1
                
        alunos_ids = df_respostas['ALUNO'].tolist()
        matriz_respostas_numpy = df_respostas[colunas_q].values
        
        print(f"A diagnosticar ({len(alunos_ids)} alunos, {n_habilidades} Matérias Totais)...")
        
        proficiencias = np.zeros((len(alunos_ids), n_habilidades))
        
        for hab_idx in range(n_habilidades):
            questoes_da_habilidade = np.where(Q_matrix[:, hab_idx] == 1)[0]
            
            if len(questoes_da_habilidade) > 0:
                proficiencias[:, hab_idx] = np.mean(matriz_respostas_numpy[:, questoes_da_habilidade], axis=1)
            else:
                proficiencias[:, hab_idx] = 0.0
                
        df_diagnostico = pd.DataFrame(proficiencias, columns=habilidades_unicas)
        df_diagnostico.insert(0, 'ALUNO', alunos_ids)
        df_diagnostico.insert(1, 'CO_CURSO', codigo_curso)
        df_diagnostico.insert(2, 'NOME_CURSO', nome_curso)
        
        for col in habilidades_unicas:
            df_diagnostico[col] = (df_diagnostico[col] * 100).round(2)
            
        def diagnosticar_aluno(row):
            notas = row[habilidades_unicas]
            notas_especificas = notas.drop('Conhecimentos Gerais', errors='ignore')
            if notas_especificas.empty:
                return pd.Series(['Conhecimentos Gerais', 'Conhecimentos Gerais'])
            return pd.Series([notas_especificas.idxmax(), notas_especificas.idxmin()])

        df_diagnostico[['MAIOR_DOMINIO_DISCIPLINA', 'PIOR_DEFICIENCIA_DISCIPLINA']] = df_diagnostico.apply(diagnosticar_aluno, axis=1)
        
        nome_arq_saida = f"diagnostico_cognitivo_{nome_base}.xlsx"
        arquivo_saida = pasta_resultados_cdm / nome_arq_saida
        
        df_diagnostico.to_excel(arquivo_saida, index=False, engine='openpyxl')
        
        print(f"[SUCESSO] Diagnóstico salvo: {nome_arq_saida}")
        resultados_gerais.append({'CURSO': nome_curso, 'ALUNOS': len(alunos_ids), 'STATUS': 'SUCESSO'})
        
    except Exception as e:
        print(f"[ERRO] Falha ao processar {nome_curso}. Detalhes: {e}")
        resultados_gerais.append({'CURSO': nome_curso, 'ALUNOS': 0, 'STATUS': f'ERRO: {str(e)}'})

# =============================================================================
# 5. RELATÓRIO FINAL DO LOTE
# =============================================================================
print(f"\n{'='*70}")
print("RESUMO DO PROCESSAMENTO EM LOTE (FASE 6 - Leitura Inteligente XLSX):")
df_resumo = pd.DataFrame(resultados_gerais)
print(df_resumo.to_string(index=False))
print(f"{'='*70}")
print(f"Todos os perfis psicométricos foram salvos em:\n{pasta_resultados_cdm}")