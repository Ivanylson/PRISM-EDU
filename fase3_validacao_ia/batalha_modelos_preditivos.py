import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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

# =============================================================================
# 2. CONFIGURAÇÕES DE CAMINHOS AUTOMÁTICOS
# =============================================================================
print("Configurando diretórios de forma automática...")
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent # <--- Retorna à raiz do projeto

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_sintese = DIRETORIO_RAIZ / 'arquivosgerados' / 'resultadofinal_relatoriosintese' / 'csv'

# Criando a nova pasta solicitada para as Análises Preditivas
pasta_analises_preditivas = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS' / 'analisesPredetivos'
pasta_analises_preditivas.mkdir(parents=True, exist_ok=True)

if not caminho_microdados.exists():
    print(f"\n❌ ERRO: Base de dados não encontrada em:\n{caminho_microdados}")
    exit()

print("A carregar a base de dados principal (Isto pode demorar um pouco)...")
df_micro = pd.read_csv(caminho_microdados, sep=';', dtype=str)
grupos_disponiveis = df_micro['CO_GRUPO'].dropna().unique()

print(f"Iniciando o treino de Machine Learning para {len(grupos_disponiveis)} cursos...\n")

# =============================================================================
# 3. MOTOR DE MACHINE LEARNING POR CURSO
# =============================================================================
for co_grupo in grupos_disponiveis:
    co_grupo_int = int(co_grupo)
    
    if co_grupo_int not in cursos_map:
        continue
        
    nome_curso = cursos_map[co_grupo_int]
    nome_formatado = formatar_nome(nome_curso)
    
    # Carrega a síntese do curso (para sabermos as matérias de cada questão)
    caminho_sintese = pasta_sintese / f"{nome_formatado}.csv"
    if not caminho_sintese.exists():
        continue
        
    # CORREÇÃO: Alterado de sep=',' para sep=';' para ler os nossos ficheiros limpos
    df_sintese = pd.read_csv(caminho_sintese, sep=';')
    
    if 'POSIÇÃO' in df_sintese.columns:
        df_sintese.rename(columns={'POSIÇÃO': 'QUESTAO'}, inplace=True)
    df_sintese['QUESTAO'] = 'Q' + df_sintese['QUESTAO'].astype(str).str.strip()
    
    colunas_oc = [col for col in df_sintese.columns if str(col).startswith('OC')]
    
    # Filtra os dados apenas deste curso
    df_curso = df_micro[df_micro['CO_GRUPO'] == str(co_grupo)].copy()
    
    # Para Machine Learning, precisamos de um bom volume de dados. 
    # Ignoramos se o curso tiver menos de 100 alunos no Brasil.
    if len(df_curso) < 100:
        continue
        
    print(f"Treinando IA para: {nome_curso} ({len(df_curso)} alunos)")
    
    # Prepara as Features (Variáveis Independentes: Questões 1 a 38)
    cols_questoes = [f'Q{i}' for i in range(1, 39) if f'Q{i}' in df_curso.columns]
    for col in cols_questoes:
        df_curso[col] = np.where(df_curso[col] == '1', 1, 0)
        
    # Define o Target (O que queremos prever)
    # Target: O aluno tirou nota acima da média geral do curso? (1 = Sim, 0 = Não)
    df_curso['NOTA_TOTAL'] = df_curso[cols_questoes].sum(axis=1)
    media_curso = df_curso['NOTA_TOTAL'].mean()
    df_curso['ALTO_DESEMPENHO'] = np.where(df_curso['NOTA_TOTAL'] >= media_curso, 1, 0)
    
    X = df_curso[cols_questoes]
    y = df_curso['ALTO_DESEMPENHO']
    
    # Verifica se há as duas classes (Bons e Ruins) para treinar. Se não, salta.
    if len(y.unique()) < 2:
        continue
        
    # Separação: 80% para Treino, 20% para Teste (Prova cega do modelo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define os Modelos
    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }
    
    resultados_metricas = []
    
    # Treina e Avalia cada modelo
    for nome_modelo, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        # O try-except evita erro no AUC caso o teste não tenha ambas as classes
        try:
            y_proba = modelo.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = np.nan
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        resultados_metricas.append({
            "Curso": nome_curso,
            "Modelo": nome_modelo,
            "Acuracia": round(acc, 4),
            "F1_Score": round(f1, 4),
            "AUC_ROC": round(auc, 4)
        })
        
    # Salva o arquivo 1: O Resultado da Batalha (Métricas)
    df_metricas = pd.DataFrame(resultados_metricas)
    caminho_metricas = pasta_analises_preditivas / f"metricas_modelos_{nome_formatado}.csv"
    df_metricas.to_csv(caminho_metricas, sep=';', index=False, encoding='utf-8-sig')
    
    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE (Extraindo a inteligência do XGBoost)
    # -------------------------------------------------------------------------
    modelo_vencedor = modelos["XGBoost"]
    importancias = modelo_vencedor.feature_importances_
    
    df_importancia = pd.DataFrame({
        'QUESTAO': cols_questoes,
        'PESO_IMPORTANCIA_%': np.round(importancias * 100, 2)
    })
    
    # Cruza os pesos com a matriz do MEC (df_sintese) para saber o que é a questão
    df_pesos_finais = pd.merge(df_importancia, df_sintese, on='QUESTAO', how='left')
    
    # Ordena para as questões que mais impactam ficarem no topo
    df_pesos_finais = df_pesos_finais.sort_values(by='PESO_IMPORTANCIA_%', ascending=False)
    
    # Une os OCs em uma coluna só, como fizemos no Dashboard
    def unir_ocs(linha):
        lista_ocs = []
        for col in colunas_oc:
            val = linha.get(col)
            if pd.notna(val) and str(val).strip() != "" and str(val).lower() != "nan":
                lista_ocs.append(str(val).strip())
        return " + ".join(lista_ocs) if lista_ocs else "Não especificado"
        
    df_pesos_finais['TODOS_OS_OCs'] = df_pesos_finais.apply(unir_ocs, axis=1)
    
    # Limpa colunas desnecessárias e organiza
    cols_para_salvar = ['QUESTAO', 'PESO_IMPORTANCIA_%', 'TODOS_OS_OCs']
    if 'COMPETÊNCIAS' in df_pesos_finais.columns: cols_para_salvar.append('COMPETÊNCIAS')
    elif 'COMPETENCIA' in df_pesos_finais.columns: cols_para_salvar.append('COMPETENCIA')
        
    df_pesos_finais = df_pesos_finais[cols_para_salvar]
    
    # Salva o arquivo 2: O peso das matérias
    caminho_pesos = pasta_analises_preditivas / f"importancia_variaveis_{nome_formatado}.csv"
    df_pesos_finais.to_csv(caminho_pesos, sep=';', index=False, encoding='utf-8-sig')

print(f"\n{'='*75}")
print(f" PROCESSO CONCLUÍDO! Todas as análises preditivas foram guardadas em:\n{pasta_analises_preditivas}")
print(f"{'='*75}")