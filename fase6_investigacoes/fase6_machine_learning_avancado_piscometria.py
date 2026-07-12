import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Sklearn & Componentes de Fronteira
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, AdaBoostClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Bibliotecas de Fronteira
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier, LGBMRegressor = None, None

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES DE CAMINHOS
# =============================================================================
print("Inicializando Fase 6 de Fronteira: Psicometria Computacional & IA Educacional...")

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent

caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_resultados_ml = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_FASE6_ML_AVANCADO_PSICOMETRIA'
pasta_resultados_ml.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. CARREGAMENTO E ENGENHARIA DE SINAIS PSICOMÉTRICOS (Simulação IRT 2PL)
# =============================================================================
print("\nCarregando microdados...")
try:
    df = pd.read_csv(caminho_microdados, sep=';', encoding='utf-8', low_memory=False)
    if 'TP_PR_GER' in df.columns:
        df = df[df['TP_PR_GER'] > 0].copy()
except Exception as e:
    print(f"Erro ao ler microdados: {e}")
    exit()

df.columns = [str(c).strip().upper() for c in df.columns]
colunas_q = [f'Q{i}' for i in range(1, 39)]

print("Mapeando Respostas e Estimando Parâmetros de Dificuldade/Discriminação (Aproximação IRT 2PL)...")
for col in colunas_q:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).apply(lambda x: 1 if x == 1 else 0)
    else:
        df[col] = 0

# --- ABORDAGEM DE FRONTEIRA: IRT 2PL DATA-DRIVEN ---
# Calculamos a taxa de acerto de cada questão (Inverso da Dificuldade 'b')
# E a correlação ponto-bisserial de cada questão com a nota final (Discriminação 'a')
df['SOMA_OBJETIVA'] = df[colunas_q].sum(axis=1)
pesos_irt = {}

for col in colunas_q:
    taxa_acerto = df[col].mean()
    dificuldade_b = 1.0 - taxa_acerto # Quanto maior, mais difícil
    # Correlação de Pearson como proxy de discriminação 'a'
    discriminacao_a = df[col].corr(df['SOMA_OBJETIVA'])
    if np.isnan(discriminacao_a) or discriminacao_a < 0.1: 
        discriminacao_a = 0.1
    pesos_irt[col] = {"a": discriminacao_a, "b": dificuldade_b}

# =============================================================================
# NOVA EXPORTAÇÃO COMPATÍVEL COM O DASHBOARD DE PSICOMETRIA
# =============================================================================
df_pesos_irt = pd.DataFrame.from_dict(pesos_irt, orient='index').reset_index()
df_pesos_irt.columns = ['Questao', 'Discriminacao_A', 'Dificuldade_B']
df_pesos_irt.to_csv(pasta_resultados_ml / 'metricas_irt_questoes.csv', index=False, sep=';')
# =============================================================================

# Aplicando a transformação nos inputs do aluno (Ponderação Bayesiana/IRT)
X_irt = np.zeros((len(df), len(colunas_q)))
for idx, col in enumerate(colunas_q):
    # Transforma o acerto/erro usando o peso de discriminação e dificuldade da questão
    X_irt[:, idx] = df[col].values * pesos_irt[col]['a'] * (1 + pesos_irt[col]['b'])

df['PERC_ACERTO_OBJETIVO'] = df[colunas_q].mean(axis=1) * 100
corte_alto = df['PERC_ACERTO_OBJETIVO'].quantile(0.70)
corte_baixo = df['PERC_ACERTO_OBJETIVO'].quantile(0.30)

df['CLASSE_DESEMPENHO'] = df['PERC_ACERTO_OBJETIVO'].apply(lambda x: 1 if x >= corte_alto else (0 if x <= corte_baixo else -1))
df_classificacao = df[df['CLASSE_DESEMPENHO'] != -1].copy()

coluna_reg_alvo = 'NT_DIS_CE' if 'NT_DIS_CE' in df_classificacao.columns else ('NT_CE_D1' if 'NT_CE_D1' in df_classificacao.columns else 'PERC_ACERTO_OBJETIVO')
df_classificacao[coluna_reg_alvo] = pd.to_numeric(df_classificacao[coluna_reg_alvo].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

X_questoes_psicometricas = X_irt[df['CLASSE_DESEMPENHO'] != -1]
y_classe = df_classificacao['CLASSE_DESEMPENHO'].values
y_continua = df_classificacao[coluna_reg_alvo].values

# =============================================================================
# 3. DETECÇÃO DE ANOMALIAS COGNITIVAS (Isolation Forest)
# =============================================================================
print("\n[IF] Filtrando anomalias cognitivas e padrões de chute...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_questoes_psicometricas)

X_limpo = X_questoes_psicometricas[outliers == 1]
y_classe_limpo = y_classe[outliers == 1]
y_continua_limpo = y_continua[outliers == 1]

# =============================================================================
# 4. REDUÇÃO DE DIMENSIONALIDADE AVANÇADA (PCA sobre Matriz Psicométrica)
# =============================================================================
print("\n[PCA] Extraindo Componentes Principais da Matriz Ponderada por IRT...")
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X_limpo)

pca = PCA(n_components=6, random_state=42)
X_pca = pca.fit_transform(X_escalado)

# =============================================================================
# 5. CLASSIFICAÇÃO EXTREMA (MLP vs AdaBoost vs LightGBM)
# =============================================================================
print("\n--- TREINANDO MODELOS DE CLASSIFICAÇÃO ---")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_classe_limpo, test_size=0.25, random_state=42)

modelos_clf = {
    "Rede Neural (MLP)": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
}
if LGBMClassifier:
    modelos_clf["LightGBM"] = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)

resultados_clf = []
for nome, modelo in modelos_clf.items():
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    acc = accuracy_score(y_test, preds)
    resultados_clf.append({"Modelo": nome, "Acurácia": acc})

pd.DataFrame(resultados_clf).to_csv(pasta_resultados_ml / 'resultado_classificacao.csv', index=False, sep=';')

# =============================================================================
# 6. REGRESSÃO DE FRONTEIRA (Mapeamento de Interação Student-Item)
# Em vez de regressão linear simples, aplicamos aproximações de Matrix Factorization / Deep Learning
# =============================================================================
print("\n--- INICIANDO PREVISÃO DE FRONTEIRA DA NOTA DISCURSIVA (REGRESSÃO NÃO-LINEAR) ---")
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_pca, y_continua_limpo, test_size=0.25, random_state=42)

modelos_reg = {
    "Ridge (Regularização L2)": Ridge(alpha=1.0),
    "Rede Neural Regressora (MLP-R)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
    "Random Forest Regressor (RFR)": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
}
if LGBMRegressor:
    modelos_reg["LightGBM Regressor"] = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)

resultados_reg = []
for nome, modelo in modelos_reg.items():
    print(f"Treinando {nome}...")
    modelo.fit(X_train_reg, y_train_reg)
    preds = modelo.predict(X_test_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
    r2 = r2_score(y_test_reg, preds)
    resultados_reg.append({"Modelo": nome, "RMSE": rmse, "R2 Score": r2})
    print(f"{nome} -> RMSE: {rmse:.2f} | R²: {r2:.4f}")

pd.DataFrame(resultados_reg).to_csv(pasta_resultados_ml / 'resultado_regressao.csv', index=False, sep=';')

# =============================================================================
# 7. EXPORTAÇÃO EXPLICATIVA (Pesos dos Itens baseados no Ensemble)
# Em vez de SHAP puro (pesado), extraímos a Importância de Atributo do Random Forest mapeada nos Itens
# =============================================================================
if "Random Forest Regressor (RFR)" in modelos_reg:
    rf_mod = modelos_reg["Random Forest Regressor (RFR)"]
    # Mapeia a importância dos componentes de volta para o impacto visual
    df_importancia = pd.DataFrame({
        "Componente": [f"PCA_COMP_{i}" for i in range(1, 7)],
        "Impacto_Decisao": rf_mod.feature_importances_
    }).sort_values(by="Impacto_Decisao", ascending=False)
    df_importancia.to_csv(pasta_resultados_ml / 'ia_explicativa_shap.csv', index=False, sep=';')

# Guardar base limpa para o K-Means Otimizado
df_final_kmeans = pd.DataFrame(X_pca, columns=[f'PCA_COMP_{i}' for i in range(1, 7)])
df_final_kmeans.insert(0, 'ALUNO', df_classificacao[outliers == 1]['ALUNO'].values)
df_final_kmeans.insert(1, 'CO_GRUPO', df_classificacao[outliers == 1]['CO_GRUPO'].values)
df_final_kmeans.insert(2, 'CO_IES', df_classificacao[outliers == 1]['CO_IES'].values)
df_final_kmeans.to_csv(pasta_resultados_ml / 'microdados_processados_pca.csv', index=False, sep=';')

print("\nPipeline de Fronteira concluído com sucesso!")