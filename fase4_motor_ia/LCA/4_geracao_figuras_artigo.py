"""
================================================================================
GERAÇÃO DAS FIGURAS DO ARTIGO SBIE
================================================================================

Lê os arquivos de saída do pipeline (etapa 0) e gera as figuras 1 a 5 do
artigo, com padrão visual unificado (300 dpi, paleta acessível, formato
1-coluna SBIE). As figuras 6 e 7 (diagnóstico de OCs) são geradas pelo
script 5_diagnostico_OCs.py.

Pré-requisitos:
    - 01_relatorio_geral_por_ies.csv
    - 02_caracterizacao_das_classes.csv
    - 03_criterios_de_selecao_k.csv

Saída:
    fig1_diagnostico.png     - Indicadores gerais da metodologia
    figN_heterogeneidade.png - Conceito vertical vs horizontal (didática)
    fig3_arquetipos.png      - Perfis arquetípicos por curso
    fig5_universalidade.png  - Universalidade dos arquétipos
    fig6_bic.png             - BIC × k para os três cursos
    figN_relatorio_professor.png - Mockup do relatório diagnóstico
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

PASTA_DADOS = Path("/caminho/para/saida/pipeline")  # AJUSTAR
PASTA_SAIDA = Path("./figuras_artigo")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})
COR_A = '#2E5984'; COR_B = '#C44536'; COR_C = '#7A8450'
CURSOS_FOCO = ['Engenharia Civil', 'Medicina', 'Enfermagem']


# ==============================================================================
# CARREGAR DADOS
# ==============================================================================
df_geral = pd.read_csv(PASTA_DADOS / '01_relatorio_geral_por_ies.csv',
                        sep=';', decimal=',')
df_classes = pd.read_csv(PASTA_DADOS / '02_caracterizacao_das_classes.csv',
                          sep=';', decimal=',')
df_crit = pd.read_csv(PASTA_DADOS / '03_criterios_de_selecao_k.csv',
                       sep=';', decimal=',')


# ==============================================================================
# FIGURA 1 - Diagnóstico geral da metodologia
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
counts = df_geral['k_escolhido'].value_counts().sort_index()
axes[0,0].bar(counts.index, counts.values, color=COR_A, alpha=0.85,
               edgecolor='black', linewidth=0.5)
axes[0,0].set_xlabel('k'); axes[0,0].set_ylabel('Número de IES')
axes[0,0].set_title('(a) Distribuição do k selecionado')
axes[0,1].hist(df_geral['entropia_normalizada'], bins=30, color=COR_C,
                alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0,1].axvline(0.80, color=COR_B, linestyle='--', label='Limiar 0,80')
axes[0,1].set_xlabel('Entropia normalizada')
axes[0,1].set_title('(b) Separação entre classes')
axes[0,1].legend()
axes[1,0].hist(df_geral['jaccard_medio_bootstrap'], bins=30, color=COR_A,
                alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1,0].set_xlabel('Jaccard (bootstrap)')
axes[1,0].set_title('(c) Estabilidade dos perfis')
axes[1,1].hist(df_geral['ari_lca_vs_kmeans_mca'], bins=30, color=COR_C,
                alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1,1].set_xlabel('Adjusted Rand Index')
axes[1,1].set_title('(d) Concordância LCA × K-Means+MCA')
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'fig1_diagnostico.png', dpi=300, bbox_inches='tight')
plt.close()


# ==============================================================================
# FIGURA N - Heterogeneidade horizontal vs vertical (didática)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
oc_labels = [f'OC{i}' for i in range(1, 11)]
np.random.seed(7)
alto_A = np.array([0.75,0.78,0.72,0.80,0.76,0.74,0.79,0.73,0.77,0.75])
baixo_A = np.array([0.30,0.32,0.28,0.31,0.29,0.27,0.33,0.30,0.28,0.31])
alpha_B = np.array([0.78,0.75,0.80,0.76,0.74,0.32,0.30,0.28,0.31,0.29])
beta_B = np.array([0.30,0.32,0.29,0.27,0.31,0.78,0.76,0.74,0.79,0.75])
x = np.arange(len(oc_labels)); w = 0.4
axes[0].bar(x-w/2, alto_A, w, color=COR_A, alpha=0.85, label='Grupo "alto"')
axes[0].bar(x+w/2, baixo_A, w, color=COR_B, alpha=0.85, label='Grupo "baixo"')
axes[0].set_xticks(x); axes[0].set_xticklabels(oc_labels)
axes[0].set_title('(a) Estratificação vertical')
axes[0].legend()
axes[1].bar(x-w/2, alpha_B, w, color=COR_A, alpha=0.85, label='Perfil α')
axes[1].bar(x+w/2, beta_B, w, color=COR_B, alpha=0.85, label='Perfil β')
axes[1].set_xticks(x); axes[1].set_xticklabels(oc_labels)
axes[1].set_title('(b) Heterogeneidade horizontal')
axes[1].legend()
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'figN_heterogeneidade.png', dpi=300, bbox_inches='tight')
plt.close()


# ==============================================================================
# FIGURA 3 - Perfis arquetípicos por curso
# ==============================================================================
fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5))
prob_cols = [f'PROB_Q{i}' for i in range(1, 39)]
qs = [f'Q{i}' for i in range(1, 39)]
for idx, curso in enumerate(CURSOS_FOCO):
    sub = df_classes[df_classes['curso']==curso].copy().reset_index(drop=True)
    cols_g = [f'GAP_Q{i}' for i in range(9, 39)]
    X = sub[cols_g].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    Z = linkage(pdist(X_norm), method='ward')
    sub['arq'] = fcluster(Z, t=2, criterion='maxclust')
    for arq in [1, 2]:
        sub_a = sub[sub['arq']==arq]
        prob = sub_a[prob_cols].mean()
        cor = COR_A if arq == 1 else COR_B
        axes[idx].plot(qs, prob.values, marker='o', markersize=4, color=cor,
                        label=f'Arquétipo {arq} ({len(sub_a)/len(sub)*100:.0f}%)')
    axes[idx].axvspan(-0.5, 7.5, alpha=0.08, color='gray')
    axes[idx].set_title(f'{curso} (n={sub["ies"].nunique()} IES)')
    axes[idx].legend(loc='upper right'); axes[idx].grid(alpha=0.3)
    axes[idx].set_ylim(-0.05, 1.05)
    axes[idx].tick_params(axis='x', rotation=90, labelsize=7)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'fig3_arquetipos.png', dpi=300, bbox_inches='tight')
plt.close()


# ==============================================================================
# FIGURA 5 - Universalidade dos arquétipos
# ==============================================================================
universalidade = []
for curso in CURSOS_FOCO:
    sub = df_classes[df_classes['curso']==curso].copy().reset_index(drop=True)
    cols_g = [f'GAP_Q{i}' for i in range(9, 39)]
    X = sub[cols_g].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    Z = linkage(pdist(X_norm), method='ward')
    sub['arq'] = fcluster(Z, t=2, criterion='maxclust')
    n_ies = sub['ies'].nunique()
    combos = sub.groupby('ies')['arq'].agg(lambda s: tuple(sorted(s.unique())))
    n_ambos = (combos == (1, 2)).sum()
    universalidade.append({'curso': curso, 'n_ies': n_ies,
                           'pct_ambos': 100*n_ambos/n_ies})
df_univ = pd.DataFrame(universalidade)

fig, ax = plt.subplots(figsize=(7, 3))
y_pos = np.arange(len(df_univ))
ax.barh(y_pos, df_univ['pct_ambos'], color=COR_A, alpha=0.85,
         edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['curso']}\n(n={r['n_ies']})"
                     for _, r in df_univ.iterrows()])
ax.set_xlabel('% das IES'); ax.set_xlim(0, 105)
for i, r in df_univ.iterrows():
    ax.text(r['pct_ambos']/2, i, f"{r['pct_ambos']:.1f}%",
             ha='center', va='center', color='white', fontweight='bold')
ax.set_title('Universalidade dos arquétipos nacionais')
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'fig5_universalidade.png', dpi=300, bbox_inches='tight')
plt.close()


# ==============================================================================
# FIGURA 6 - BIC × k
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
for curso, cor in zip(CURSOS_FOCO, [COR_A, COR_B, COR_C]):
    sub = df_crit[df_crit['curso']==curso]
    bic_med = sub.groupby('k')['bic'].median()
    bic_norm = bic_med - bic_med.loc[2]
    ax.plot(bic_norm.index, bic_norm.values, marker='o',
             label=curso, color=cor, linewidth=2)
ax.axhline(0, color='black', linestyle=':', alpha=0.5)
ax.set_xlabel('k'); ax.set_ylabel('BIC relativo a k=2')
ax.set_title('Critério BIC × k para os três cursos')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'fig6_bic.png', dpi=300, bbox_inches='tight')
plt.close()


print(f"Figuras geradas em: {PASTA_SAIDA}")
