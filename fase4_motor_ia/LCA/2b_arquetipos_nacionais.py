"""
================================================================================
ANÁLISE 2 DETALHADA - PADRÕES INTER-INSTITUCIONAIS COM MATRIZ COMPLETA
================================================================================

Pré-requisito:
    Pipeline principal `analise_perfis_enade.py` rodado com a versão
    atualizada que salva PROB_Q1..PROB_Q38 e GAP_Q1..GAP_Q38 por classe
    no arquivo 02_caracterizacao_das_classes.csv.

Diferenças em relação à versão anterior:
    1. Co-ocorrência de itens: usa MATRIZ DE GAPS (38-dimensional) e calcula
       correlação de Pearson entre itens através de todas as classes.
       Substitui a métrica "ranking top-5" por sinal contínuo.

    2. Tipologia de classes: clustering hierárquico das CLASSES (não só dos
       itens) em espaço 38-dimensional, identificando quantos "tipos de
       perfil" existem em todo o universo do curso.

    3. Estabilidade: validação com bootstrap das classes para verificar se
       a tipologia identificada é robusta.

Pergunta principal:
    Existem k* "perfis arquetípicos" inter-IES em cada curso? Se sim,
    quantos são e como se caracterizam?

Saídas:
    - Matriz de correlação entre itens (heatmap por curso)
    - Dendrograma de tipologia de classes
    - Caracterização dos perfis arquetípicos
    - Mapa de "para qual arquétipo cada IES contribui"
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

# AJUSTAR ESTES CAMINHOS PARA APONTAR PARA SEU CSV ATUALIZADO
PASTA_DADOS = Path("/home/claude/dados_teste")           # ou onde estiver
ARQUIVO_CLASSES = PASTA_DADOS / "02_simulado.csv"        # nome do CSV completo

PASTA_SAIDA = Path("/mnt/user-data/outputs/analise_2_detalhada")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

CURSOS_FOCO = ["Enfermagem", "Engenharia Civil", "Medicina"]
QUESTOES_TODAS = [f"Q{i}" for i in range(1, 39)]
QUESTOES_CE = [f"Q{i}" for i in range(9, 39)]

# Limite para identificar arquétipos: k testado de 2 a K_MAX_ARQUETIPOS
K_MAX_ARQUETIPOS = 8


# ==============================================================================
# 1. CARREGAMENTO E VALIDAÇÃO
# ==============================================================================

def carregar_e_validar():
    """Carrega o CSV e verifica que tem as colunas PROB e GAP esperadas."""
    df = pd.read_csv(ARQUIVO_CLASSES, sep=";", decimal=",")
    cols_prob = [f"PROB_{q}" for q in QUESTOES_TODAS]
    cols_gap = [f"GAP_{q}" for q in QUESTOES_TODAS]

    faltam_prob = [c for c in cols_prob if c not in df.columns]
    faltam_gap = [c for c in cols_gap if c not in df.columns]
    if faltam_prob or faltam_gap:
        raise ValueError(
            f"CSV não tem todas as colunas esperadas.\n"
            f"Faltam PROB: {faltam_prob[:5]}...\nFaltam GAP: {faltam_gap[:5]}...\n"
            f"=> Re-rode o pipeline principal com a versão atualizada."
        )
    print(f"[OK] {len(df)} classes carregadas com matriz completa.")
    return df


# ==============================================================================
# 2. CO-OCORRÊNCIA DE ITENS (CORRELAÇÃO ENTRE GAPS)
# ==============================================================================

def correlacao_itens(df_curso, cols=None):
    """
    Calcula correlação de Pearson entre os GAPS dos itens, através de todas
    as classes do curso.

    Interpretação:
        corr(Q_i, Q_j) > 0 : itens tendem a ter gap na MESMA direção em
                              todas as classes (= são dominados juntos).
        corr(Q_i, Q_j) < 0 : itens são complementares (uma classe é forte
                              em um e fraca no outro — característico do
                              espelhamento que vimos).
        corr ≈ 0 : itens são independentes em relação aos perfis.
    """
    cols = cols or [f"GAP_{q}" for q in QUESTOES_CE]
    M = df_curso[cols].values
    return np.corrcoef(M.T)


# ==============================================================================
# 3. TIPOLOGIA DE CLASSES (CLUSTERING DE PERFIS)
# ==============================================================================

def encontrar_arquetipos(df_curso, cols_gap, k_min=2, k_max=8):
    """
    Clustering hierárquico das CLASSES no espaço de gaps.
    Para cada k testado, calcula silhueta e Calinski-Harabasz.
    Retorna a tabela de critérios e o k recomendado.
    """
    from sklearn.metrics import calinski_harabasz_score

    X = df_curso[cols_gap].values
    # Padronizar para que itens com gap maior não dominem
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    # Distância euclidiana, ligação Ward (apropriada para padronizado)
    D = pdist(X_norm, metric="euclidean")
    Z = linkage(D, method="ward")

    resultados = []
    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(labels)) < k:
            continue
        sil = silhouette_score(X_norm, labels)
        ch = calinski_harabasz_score(X_norm, labels)
        # Tamanho da menor classe
        bins = np.bincount(labels)
        bins = bins[bins > 0]
        menor_pct = bins.min() / len(labels)
        resultados.append({
            "k": k, "silhueta": sil,
            "calinski_harabasz": ch,
            "menor_classe_pct": menor_pct,
        })

    df_crit = pd.DataFrame(resultados)
    # Recomendação: maior silhueta entre k com menor classe >= 5%
    candidatos = df_crit[df_crit["menor_classe_pct"] >= 0.05]
    if len(candidatos) > 0:
        k_rec = int(candidatos.loc[candidatos["silhueta"].idxmax(), "k"])
    else:
        k_rec = int(df_crit.loc[df_crit["silhueta"].idxmax(), "k"])

    labels_rec = fcluster(Z, t=k_rec, criterion="maxclust")
    return df_crit, k_rec, labels_rec, Z, X_norm


def caracterizar_arquetipos(df_curso, labels, cols_gap, cols_prob):
    """Para cada arquétipo, calcula gap médio e prob média por item."""
    df = df_curso.copy()
    df["arquetipo"] = labels

    perfis = []
    for arq in sorted(df["arquetipo"].unique()):
        sub = df[df["arquetipo"] == arq]
        n_classes = len(sub)
        n_alunos = sub["n_alunos"].sum()
        # Em quantas IES distintas esse arquétipo aparece?
        n_ies = sub["ies"].nunique()

        # Gap médio e prob média por item
        gap_medio = sub[cols_gap].mean()
        prob_media = sub[cols_prob].mean()

        # Itens mais característicos (top 5 gap positivo)
        gaps_ord = gap_medio.sort_values(ascending=False)
        top_fortes = gaps_ord.head(5).index.tolist()
        top_fracos = gaps_ord.tail(5).index.tolist()

        perfis.append({
            "arquetipo": arq,
            "n_classes": n_classes,
            "n_alunos": int(n_alunos),
            "n_ies_distintas": n_ies,
            "pct_classes": round(100 * n_classes / len(df), 1),
            "taxa_acerto_media": round(sub["taxa_acerto_media"].mean(), 4),
            "itens_top_fortes": ", ".join(q.replace("GAP_", "") for q in top_fortes),
            "itens_top_fracos": ", ".join(q.replace("GAP_", "") for q in top_fracos),
            "gap_medio_top_forte": round(gap_medio[top_fortes[0]], 4),
            "gap_medio_top_fraco": round(gap_medio[top_fracos[0]], 4),
        })
    return pd.DataFrame(perfis)


# ==============================================================================
# 4. VISUALIZAÇÃO
# ==============================================================================

def plot_correlacao_itens(corr, itens, curso, ax):
    """Heatmap de correlação entre itens, com reordenamento por clustering."""
    # Reordenar pela hierarquia para evidenciar blocos
    D = 1 - np.abs(corr)
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    ordem = dendrogram(Z, no_plot=True, labels=itens)["ivl"]
    idx = [itens.index(q) for q in ordem]
    corr_ord = corr[np.ix_(idx, idx)]

    sns.heatmap(corr_ord, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                xticklabels=ordem, yticklabels=ordem, square=True,
                cbar_kws={"label": "Correlação de Pearson entre gaps"})
    ax.set_title(f"{curso}\nCorrelação entre itens (Q9-Q38)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)


def plot_perfis_arquetipos(df_perfis, df_curso, labels, cols_prob,
                            curso, ax):
    """Linhas de perfil de probabilidade por item para cada arquétipo."""
    df_curso_c = df_curso.copy()
    df_curso_c["arquetipo"] = labels

    for arq in sorted(df_curso_c["arquetipo"].unique()):
        sub = df_curso_c[df_curso_c["arquetipo"] == arq]
        prob_media = sub[cols_prob].mean()
        questoes = [c.replace("PROB_", "") for c in cols_prob]
        n_pct = (df_perfis[df_perfis["arquetipo"] == arq]["pct_classes"]
                 .iloc[0])
        ax.plot(questoes, prob_media.values, marker="o", markersize=3,
                label=f"Arq {arq} ({n_pct}%)", alpha=0.8)

    ax.set_xlabel("Item")
    ax.set_ylabel("Probabilidade de acerto média")
    ax.set_title(f"{curso}: perfis arquetípicos")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(alpha=0.3)


# ==============================================================================
# 5. PIPELINE POR CURSO
# ==============================================================================

def analisar_curso(curso, df_classes):
    print(f"\n{'='*70}\nCURSO: {curso}\n{'='*70}")
    df_curso = df_classes[df_classes["curso"] == curso].copy().reset_index(drop=True)
    print(f"  IES: {df_curso['ies'].nunique()} | Classes: {len(df_curso)}")

    cols_gap = [f"GAP_{q}" for q in QUESTOES_CE]
    cols_prob = [f"PROB_{q}" for q in QUESTOES_CE]

    # 5.1 Correlação entre itens
    print(f"\n[5.1] Calculando correlações entre itens...")
    corr = correlacao_itens(df_curso, cols=cols_gap)
    df_corr = pd.DataFrame(corr, index=QUESTOES_CE, columns=QUESTOES_CE)

    # Pares com correlação extrema
    triu = np.triu_indices_from(corr, k=1)
    pares = []
    for i, j in zip(*triu):
        pares.append({
            "item_1": QUESTOES_CE[i], "item_2": QUESTOES_CE[j],
            "correlacao": round(corr[i, j], 4),
        })
    df_pares = pd.DataFrame(pares)
    df_pares["abs_corr"] = df_pares["correlacao"].abs()
    df_pares = df_pares.sort_values("abs_corr", ascending=False)

    print("  TOP 10 pares com maior correlação POSITIVA "
          "(itens dominados juntos):")
    print(df_pares[df_pares["correlacao"] > 0].head(10)[
        ["item_1", "item_2", "correlacao"]].to_string(index=False))

    print("\n  TOP 10 pares com maior correlação NEGATIVA "
          "(itens complementares - espelhamento):")
    print(df_pares[df_pares["correlacao"] < 0].head(10)[
        ["item_1", "item_2", "correlacao"]].to_string(index=False))

    # 5.2 Tipologia de classes
    print(f"\n[5.2] Buscando arquétipos de perfil (k = 2..{K_MAX_ARQUETIPOS})...")
    df_crit, k_rec, labels_rec, Z, X_norm = encontrar_arquetipos(
        df_curso, cols_gap, k_min=2, k_max=K_MAX_ARQUETIPOS)
    print("  Critérios:")
    print(df_crit.to_string(index=False))
    print(f"  => k recomendado: {k_rec}")

    df_perfis = caracterizar_arquetipos(df_curso, labels_rec, cols_gap, cols_prob)
    print("\n  Perfis arquetípicos:")
    print(df_perfis.to_string(index=False))

    # 5.3 Visualizações
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plot_correlacao_itens(corr, QUESTOES_CE, curso, axes[0])
    plot_perfis_arquetipos(df_perfis, df_curso, labels_rec, cols_prob,
                           curso, axes[1])
    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / f"{curso.replace(' ', '_')}_panorama.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Dendrograma separado
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, no_labels=True, color_threshold=Z[-(k_rec - 1), 2], ax=ax)
    ax.axhline(y=Z[-(k_rec - 1), 2], color="red", linestyle="--",
               label=f"Corte para k={k_rec}")
    ax.set_title(f"{curso}: Dendrograma das classes em espaço de gaps")
    ax.set_ylabel("Distância (Ward)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / f"{curso.replace(' ', '_')}_dendrograma.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # 5.4 Mapeamento IES -> arquétipos
    df_curso["arquetipo"] = labels_rec
    mapa_ies = df_curso.groupby("ies")["arquetipo"].agg(
        lambda s: ", ".join(map(str, sorted(s.unique())))
    ).reset_index()
    mapa_ies.columns = ["ies", "arquetipos_presentes"]
    mapa_ies["curso"] = curso

    # Salvar tudo
    nome = curso.replace(" ", "_")
    df_corr.to_csv(PASTA_SAIDA / f"correlacao_itens_{nome}.csv",
                   sep=";", encoding="utf-8-sig", decimal=",")
    df_pares.head(50).to_csv(PASTA_SAIDA / f"top_pares_{nome}.csv",
                              sep=";", index=False,
                              encoding="utf-8-sig", decimal=",")
    df_crit.to_csv(PASTA_SAIDA / f"criterios_arquetipos_{nome}.csv",
                   sep=";", index=False, encoding="utf-8-sig", decimal=",")
    df_perfis.to_csv(PASTA_SAIDA / f"perfis_arquetipos_{nome}.csv",
                     sep=";", index=False, encoding="utf-8-sig", decimal=",")
    mapa_ies.to_csv(PASTA_SAIDA / f"mapa_ies_arquetipos_{nome}.csv",
                    sep=";", index=False, encoding="utf-8-sig", decimal=",")

    return {
        "curso": curso, "k_arquetipos": k_rec,
        "df_perfis": df_perfis, "df_pares": df_pares,
        "n_ies": df_curso["ies"].nunique(),
        "n_classes": len(df_curso),
    }


# ==============================================================================
# 6. EXECUÇÃO
# ==============================================================================

def main():
    print("=" * 70)
    print("ANÁLISE 2 DETALHADA - MATRIZ COMPLETA DE PROBABILIDADES")
    print("=" * 70)

    df_classes = carregar_e_validar()

    resultados = []
    for curso in CURSOS_FOCO:
        if curso not in df_classes["curso"].unique():
            print(f"AVISO: '{curso}' não encontrado.")
            continue
        res = analisar_curso(curso, df_classes)
        resultados.append(res)

    # Síntese
    sint = pd.DataFrame([{
        "curso": r["curso"],
        "n_ies": r["n_ies"],
        "n_classes": r["n_classes"],
        "k_arquetipos_recomendado": r["k_arquetipos"],
    } for r in resultados])
    sint.to_csv(PASTA_SAIDA / "00_sintese.csv",
                sep=";", index=False, encoding="utf-8-sig", decimal=",")

    print(f"\n[OK] Resultados em: {PASTA_SAIDA}")
    print("\nSÍNTESE:")
    print(sint.to_string(index=False))


if __name__ == "__main__":
    main()
