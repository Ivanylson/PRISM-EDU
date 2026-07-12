import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
import warnings

warnings.filterwarnings('ignore')

print("=============================================================================")
print("Módulo Corrigido: NLP Educacional (TF-IDF sobre Cadeias de Acertos)")
print("=============================================================================")

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent
caminho_microdados = DIRETORIO_RAIZ / 'arquivosgerados' / 'relatorio_final_enade_2023.csv'
pasta_resultados = DIRETORIO_RAIZ / 'arquivosgerados' / 'RESULTADOS_FASE6_ML_AVANCADO'

try:
    df = pd.read_csv(caminho_microdados, sep=';', encoding='utf-8', low_memory=False)
    df = df[df['TP_PR_GER'] > 0].copy().head(5000) # Processamento ágil e seguro
except Exception as e:
    print(f"Erro ao ler microdados: {e}")
    exit()

colunas_q = [f'Q{i}' for i in range(1, 39)]
for col in colunas_q:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).apply(lambda x: "ACERTO" if x == 1 else "ERRO")

# Construindo o corpus textual a partir da sequência lógica de respostas de cada aluno
print("[NLP] Transformando a matriz de respostas em documentos textuais combinados...")
df['CORPUS_RESPOSTAS'] = df[colunas_q].astype(str).agg(' '.join, axis=1)

# Aplicando TF-IDF sobre os n-gramas lógicos de acertos e erros
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=100)
X_tfidf = vectorizer.fit_transform(df['CORPUS_RESPOSTAS']).toarray()

# Definição do problema Multirrótulo: Prever se o aluno obteve nota acima da média nas discursivas disponíveis
multirrotulos_alvo = ['NT_FG_D1_PT', 'NT_FG_D_CT']
y_multi = np.zeros((len(df), len(multirrotulos_alvo)))

for idx, col in enumerate(multirrotulos_alvo):
    if col in df.columns:
        valores_num = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0).values
        mediana_alvo = np.median(valores_num)
        y_multi[:, idx] = np.where(valores_num >= mediana_alvo, 1, 0)

print("[ML-RÓTULO] Executando classificação multirrótulo paralela via florestas randômicas textuais...")
clf_multitask = MultiOutputClassifier(RandomForestClassifier(n_estimators=10, random_state=42), n_jobs=-1)
clf_multitask.fit(X_tfidf, y_multi)
preds_multi = clf_multitask.predict(X_tfidf)

perda_hamming = hamming_loss(y_multi, preds_multi)
print(f"NLP Multirrótulo executado com sucesso! Hamming Loss: {perda_hamming:.4f}")

df_nlp = pd.DataFrame([{"Métrica": "Hamming Loss (Cadeias de Itens)", "Valor": perda_hamming}])
df_nlp.to_csv(pasta_resultados / 'resultado_nlp_multilabel.csv', index=False, sep=';')