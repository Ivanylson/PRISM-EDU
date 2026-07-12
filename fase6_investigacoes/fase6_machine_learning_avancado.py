import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Sklearn & Modelos Avançados
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Tentativa de importação de bibliotecas externas de ponta
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES DE CAMINHOS
# =============================================================================
print("Inicializando Fase 6 Avançada: Machine Learning sob Medida (Métricas Reais)...")

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_resultados_ml = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_FASE6_ML_AVANCADO_PSICOMETRIA'
pasta_resultados_ml.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. CARREGAMENTO DOS DADOS E ENGENHARIA DE SINAIS
# =============================================================================
print("\nCarregando microdados...")
try:
    df = pd.read_csv(caminho_microdados, sep=';', encoding='utf-8', low_memory=False)
    # Filtrar alunos presentes na prova geral
    if 'TP_PR_GER' in df.columns:
        df = df[df['TP_PR_GER'] > 0].copy()
except Exception as e:
    print(f"Erro ao ler microdados: {e}")
    exit()

# Padronizar colunas para maiúsculas e remover espaços
df.columns = [str(c).strip().upper() for c in df.columns]

colunas_q = [f'Q{i}' for i in range(1, 39)]

print("Processando respostas objetivas (Q1 a Q38)...")
# Garantir que as respostas são numéricas binárias (0 ou 1)
for col in colunas_q:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).apply(lambda x: 1 if x == 1 else 0)
    else:
        df[col] = 0

# --- ESTRATÉGIA DE DESEMPENHO EM PROVA OBJETIVA ---
# Como não temos NT_GER, calculamos a proporção de acertos nas 38 questões objetivas
df['PERC_ACERTO_OBJETIVO'] = df[colunas_q].mean(axis=1) * 100

# Definir classes extremas baseadas na taxa de acertos objetiva
corte_alto = df['PERC_ACERTO_OBJETIVO'].quantile(0.70)
corte_baixo = df['PERC_ACERTO_OBJETIVO'].quantile(0.30)

def rotular_classe(taxa):
    if taxa >= corte_alto: return 1  # Alto Desempenho
    elif taxa <= corte_baixo: return 0  # Risco Crítico
    return -1

df['CLASSE_DESEMPENHO'] = df['PERC_ACERTO_OBJETIVO'].apply(rotular_classe)
df_classificacao = df[df['CLASSE_DESEMPENHO'] != -1].copy()

# Escolhendo a variável alvo para Regressão: Nota Discursiva do Componente Específico
coluna_reg_alvo = 'NT_DIS_CE' if 'NT_DIS_CE' in df_classificacao.columns else ('NT_CE_D1' if 'NT_CE_D1' in df_classificacao.columns else 'PERC_ACERTO_OBJETIVO')

# Converte alvo da regressão de string/vírgula para float
df_classificacao[coluna_reg_alvo] = pd.to_numeric(df_classificacao[coluna_reg_alvo].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

X_questoes = df_classificacao[colunas_q].values
y_classe = df_classificacao['CLASSE_DESEMPENHO'].values
y_continua = df_classificacao[coluna_reg_alvo].values

print(f"Total de alunos selecionados (extremos de desempenho): {len(df_classificacao)}")
print(f"Variável alvo selecionada para Regressão Avançada: '{coluna_reg_alvo}'")

# =============================================================================
# 3. DETECÇÃO DE OUTLIERS / CHUTES (Isolation Forest)
# =============================================================================
print("\n[IF] Rodando Isolation Forest para remover padrões de 'chute' / anomalias...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_questoes)

# Filtrando dados normais
X_limpo = X_questoes[outliers == 1]
y_classe_limpo = y_classe[outliers == 1]
y_continua_limpo = y_continua[outliers == 1]

print(f"Removidos {np.sum(outliers == -1)} alunos com respostas sob suspeita de anomalia/chute.")

# =============================================================================
# 4. REDUÇÃO DE DIMENSIONALIDADE (PCA)
# =============================================================================
print("\n[PCA] Reduzindo as 38 dimensões para Componentes Principais...")
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X_limpo)

pca = PCA(n_components=6, random_state=42)
X_pca = pca.fit_transform(X_escalado)

variancia_explicada = np.sum(pca.explained_variance_ratio_) * 100
print(f"PCA Concluído. 6 componentes explicam {variancia_explicada:.2f}% da variância coletiva.")

# =============================================================================
# 5. SPLIT DE DADOS E SMOTE
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_classe_limpo, test_size=0.25, random_state=42)

if SMOTE:
    print("\n[SMOTE] Balanceando classes de treinamento via superamostragem sintética...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Base balanceada com sucesso: {np.bincount(y_train_res)}")
else:
    print("\n[AVISO] Biblioteca imbalanced-learn ausente. Ignorando SMOTE.")
    X_train_res, y_train_res = X_train, y_train

# =============================================================================
# 6. BATALHA DE CLASSIFICAÇÃO (MLP, LightGBM, AdaBoost)
# =============================================================================
print("\n--- INICIANDO BATALHA DE MODELOS DE CLASSIFICAÇÃO ---")

modelos_clf = {
    "Rede Neural (MLP)": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
}

if LGBMClassifier:
    modelos_clf["LightGBM"] = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)

resultados_clf = []

for nome, modelo in modelos_clf.items():
    print(f"Treinando {nome}...")
    modelo.fit(X_train_res, y_train_res)
    preds = modelo.predict(X_test)
    acc = accuracy_score(y_test, preds)
    resultados_clf.append({"Modelo": nome, "Acurácia": acc})
    print(f"Acurácia {nome}: {acc:.4f}")

df_res_clf = pd.DataFrame(resultados_clf)
df_res_clf.to_csv(pasta_resultados_ml / 'resultado_classificacao.csv', index=False, sep=';')

# =============================================================================
# [NOVA TÉCNICA] 6B. INTELIGÊNCIA ARTIFICIAL EXPLICATIVA (XAI) VIA SHAP / FEATURE IMPORTANCE
# =============================================================================
print("\n--- APLICANDO IA EXPLICATIVA (XAI) ---")
# Como o SHAP pode exigir instalação extra (pip install shap), 
# usamos uma abordagem híbrida robusta baseada nos pesos do AdaBoost 
# mapeados de volta aos componentes do PCA para explicar as decisões da IA.

if "AdaBoost" in modelos_clf:
    modelo_adaboost = modelos_clf["AdaBoost"]
    importancias = modelo_adaboost.feature_importances_
    
    df_importancia = pd.DataFrame({
        "Componente": [f"PCA_COMP_{i}" for i in range(1, 7)],
        "Impacto_Decisao": importancias
    }).sort_values(by="Impacto_Decisao", ascending=False)
    
    # Salvar o resultado para o Dashboard ler
    df_importancia.to_csv(pasta_resultados_ml / 'ia_explicativa_shap.csv', index=False, sep=';')
    print("Sucesso! Pesos explicativos da IA extraídos e salvos.")

# =============================================================================
# 7. REGRESSÃO AVANÇADA (Ridge, Lasso, ElasticNet)
# =============================================================================
print("\n--- INICIANDO PREVISÃO DA NOTA (REGRESSÃO AVANÇADA) ---")
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_pca, y_continua_limpo, test_size=0.25, random_state=42)

modelos_reg = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

resultados_reg = []

for nome, modelo in modelos_reg.items():
    print(f"Treinando {nome}...")
    modelo.fit(X_train_reg, y_train_reg)
    preds = modelo.predict(X_test_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
    r2 = r2_score(y_test_reg, preds)
    resultados_reg.append({"Modelo": nome, "RMSE": rmse, "R2 Score": r2})
    print(f"{nome} -> RMSE: {rmse:.2f} | R²: {r2:.4f}")

df_res_reg = pd.DataFrame(resultados_reg)
df_res_reg.to_csv(pasta_resultados_ml / 'resultado_regressao.csv', index=False, sep=';')

# =============================================================================
# 8. EXPORTAÇÃO DO DATASET LIMPO E COMPACTADO PARA O K-MEANS
# =============================================================================
df_final_kmeans = pd.DataFrame(X_pca, columns=[f'PCA_COMP_{i}' for i in range(1, 7)])
df_final_kmeans.insert(0, 'ALUNO', df_classificacao[outliers == 1]['ALUNO'].values)
df_final_kmeans.insert(1, 'CO_GRUPO', df_classificacao[outliers == 1]['CO_GRUPO'].values)
df_final_kmeans.insert(2, 'CO_IES', df_classificacao[outliers == 1]['CO_IES'].values)

df_final_kmeans.to_csv(pasta_resultados_ml / 'microdados_processados_pca.csv', index=False, sep=';')

print(f"\n=============================================================================")
print(f"Pipeline Executado! Artefatos salvos em:\n{pasta_resultados_ml}")
print(f"O arquivo comprimido por PCA está pronto para otimizar os seus agrupamentos!")
print(f"=============================================================================")