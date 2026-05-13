"""
================================================================================
ANÁLISE 2 - ESTUDO DE CASO APROFUNDADO POR CURSO
================================================================================

Pergunta de pesquisa:
    Dado que os perfis se especializam DENTRO do Componente Específico
    (Achado da Análise 1), existe um padrão CONSISTENTE entre IES de um
    mesmo curso? Ou seja: as Q9-Q38 se agrupam em "blocos temáticos" que
    se repetem em diferentes IES?

Método:
    1. Co-ocorrência: matriz de quantas vezes pares de itens aparecem
       juntos como "fortes" da mesma classe (entre todas as IES do curso).
    2. Clustering hierárquico (Ward) sobre a matriz de coocorrência.
    3. Identificação dos "blocos temáticos latentes" — conjuntos de itens
       que tendem a ser dominados pelos mesmos alunos.
    4. Caracterização dos perfis típicos do curso.

Cursos analisados (parametrizado):
    - Enfermagem (230 IES, maior amostra)
    - Engenharia Civil (127 IES)
    - Medicina (218 IES)

Saída:
    - Heatmap de co-ocorrência por curso
    - Dendrograma de blocos temáticos
    - Tabela de perfis típicos do curso
    - Quanto cada IES adere a esses perfis típicos

REFERÊNCIAS:
    Murtagh, F., & Legendre, P. (2014). Ward's hierarchical agglomerative
    clustering method. Journal of Classification, 31(3), 274-295.
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from itertools import combinations
from collections import Counter

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

PASTA_DADOS = Path("/mnt/user-data/uploads")
PASTA_SAIDA = Path("/mnt/user-data/outputs/analise_2_estudo_caso")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

CURSOS_FOCO = ["Enfermagem", "Engenharia Civil", "Medicina"]
QUESTOES_TODAS = [f"Q{i}" for i in range(1, 39)]
QUESTOES_CE = [f"Q{i}" for i in range(9, 39)]


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def parse_itens(s):
    if pd.isna(s) or s in ("-", "", "nan"):
        return []
    return [q.strip() for q in str(s).split(",") if q.strip()]


def matriz_coocorrencia(df_curso_classes, lista_itens):
    """
    Constrói matriz de co-ocorrência: quantas classes têm o item i E o
    item j AMBOS como itens diferencialmente fortes.
    """
    n = len(lista_itens)
    idx = {q: k for k, q in enumerate(lista_itens)}
    M = np.zeros((n, n), dtype=int)

    for _, r in df_curso_classes.iterrows():
        fortes = parse_itens(r["itens_diferencialmente_fortes"])
        fortes = [q for q in fortes if q in idx]
        # Diagonal: contagem de ocorrência individual
        for q in fortes:
            M[idx[q], idx[q]] += 1
        # Fora-diagonal: pares
        for q1, q2 in combinations(fortes, 2):
            i, j = idx[q1], idx[q2]
            M[i, j] += 1
            M[j, i] += 1
    return M, lista_itens


def normalizar_jaccard(M):
    """
    Converte matriz de coocorrência em similaridade tipo Jaccard:
        sim(i,j) = M[i,j] / (M[i,i] + M[j,j] - M[i,j])
    """
    n = M.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                S[i, j] = 1.0
            else:
                denom = M[i, i] + M[j, j] - M[i, j]
                S[i, j] = M[i, j] / denom if denom > 0 else 0
    return S


def clustering_hierarquico(S, lista_itens, t_corte=0.5):
    """Ward sobre a matriz de DISTÂNCIA (1 - similaridade)."""
    D = 1 - S
    np.fill_diagonal(D, 0)
    # Garantir simetria perfeita
    D = (D + D.T) / 2
    D_cond = squareform(D, checks=False)
    Z = linkage(D_cond, method="average")
    grupos = fcluster(Z, t=t_corte, criterion="distance")
    return Z, dict(zip(lista_itens, grupos))


def identificar_blocos(grupos_dict):
    """Agrupa itens por seu cluster temático."""
    blocos = {}
    for q, g in grupos_dict.items():
        blocos.setdefault(g, []).append(q)
    return {k: sorted(v, key=lambda x: int(x[1:])) for k, v in blocos.items()}


# ==============================================================================
# ANÁLISE PRINCIPAL POR CURSO
# ==============================================================================

def analisar_curso(curso_nome, df_classes):
    print(f"\n{'='*70}")
    print(f"CURSO: {curso_nome}")
    print(f"{'='*70}")

    df_curso = df_classes[df_classes["curso"] == curso_nome].copy()
    n_ies = df_curso["ies"].nunique()
    n_classes = len(df_curso)
    print(f"  IES: {n_ies} | Classes: {n_classes}")

    # ---- Matrizes de coocorrência: TODAS as questões e SÓ específicas ----
    M_full, _ = matriz_coocorrencia(df_curso, QUESTOES_TODAS)
    S_full = normalizar_jaccard(M_full)

    M_ce, _ = matriz_coocorrencia(df_curso, QUESTOES_CE)
    S_ce = normalizar_jaccard(M_ce)

    # ---- Clustering hierárquico nas Q específicas (foco da análise) ----
    Z_ce, grupos_ce = clustering_hierarquico(S_ce, QUESTOES_CE, t_corte=0.6)
    blocos_ce = identificar_blocos(grupos_ce)
    print(f"\n  Blocos temáticos identificados (Q9-Q38, Ward, corte=0.6):")
    for b_id, itens in sorted(blocos_ce.items(), key=lambda x: -len(x[1])):
        print(f"    Bloco {b_id} ({len(itens)} itens): {', '.join(itens)}")

    # ---- Heatmap de coocorrência ----
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Reordenar pelos blocos para visualização
    ordem = sorted(QUESTOES_CE, key=lambda q: (grupos_ce[q], int(q[1:])))
    idx_ord = [QUESTOES_CE.index(q) for q in ordem]
    S_ord = S_ce[np.ix_(idx_ord, idx_ord)]

    sns.heatmap(S_ord, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=0.6,
                xticklabels=ordem, yticklabels=ordem, square=True,
                cbar_kws={"label": "Similaridade (Jaccard)"})
    axes[0].set_title(f"{curso_nome}: Co-ocorrência de itens\ncomo 'fortes' do mesmo perfil (Q9-Q38)")
    axes[0].tick_params(axis="x", rotation=90, labelsize=8)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8)

    # Dendrograma
    dendrogram(Z_ce, labels=QUESTOES_CE, leaf_rotation=90,
               leaf_font_size=8, ax=axes[1], color_threshold=0.6)
    axes[1].axhline(y=0.6, color="red", linestyle="--", alpha=0.6,
                    label="Corte (0.6)")
    axes[1].set_title(f"{curso_nome}: Dendrograma dos blocos temáticos")
    axes[1].set_ylabel("Distância (1 - Jaccard)")
    axes[1].legend()

    plt.tight_layout()
    arq_fig = PASTA_SAIDA / f"coocorrencia_{curso_nome.replace(' ', '_')}.png"
    plt.savefig(arq_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura salva: {arq_fig.name}")

    # ---- Tabela de blocos com frequências individuais ----
    contagem_individual = {q: M_ce[QUESTOES_CE.index(q), QUESTOES_CE.index(q)]
                           for q in QUESTOES_CE}
    linhas_blocos = []
    for b_id, itens in sorted(blocos_ce.items(), key=lambda x: -len(x[1])):
        for q in itens:
            linhas_blocos.append({
                "curso": curso_nome,
                "bloco_tematico": b_id,
                "questao": q,
                "freq_como_forte": contagem_individual[q],
                "pct_classes_que_dominam": round(
                    100 * contagem_individual[q] / n_classes, 1),
            })
    df_blocos = pd.DataFrame(linhas_blocos)

    # ---- Adesão das IES aos perfis típicos ----
    # Cada IES tem 2+ classes; ver se cada classe "encaixa" em um bloco temático
    adesao = []
    for (curso, ies), g in df_curso.groupby(["curso", "ies"]):
        for _, r in g.iterrows():
            fortes = parse_itens(r["itens_diferencialmente_fortes"])
            fortes_ce = [q for q in fortes if q in QUESTOES_CE]
            if not fortes_ce:
                continue
            # Para cada bloco, conta quantos itens da classe estão no bloco
            blocos_contagem = Counter(grupos_ce[q] for q in fortes_ce)
            bloco_dominante, n_no_bloco = blocos_contagem.most_common(1)[0]
            pct_no_bloco = n_no_bloco / len(fortes_ce)
            adesao.append({
                "curso": curso, "ies": ies, "classe_id": r["classe_id"],
                "n_alunos": r["n_alunos"], "pct_alunos": r["pct_alunos"],
                "taxa_acerto_media": r["taxa_acerto_media"],
                "bloco_dominante": bloco_dominante,
                "n_itens_no_bloco": n_no_bloco,
                "pct_alinhamento": round(pct_no_bloco * 100, 1),
                "itens_fortes_CE": ", ".join(fortes_ce),
            })
    df_adesao = pd.DataFrame(adesao)

    print(f"\n  Adesão das classes aos blocos típicos:")
    print(f"    % com >=80% dos itens no bloco dominante: "
          f"{(df_adesao['pct_alinhamento']>=80).mean()*100:.1f}%")
    print(f"    Alinhamento médio: {df_adesao['pct_alinhamento'].mean():.1f}%")

    # ---- Salvar tabelas ----
    arq_blocos = PASTA_SAIDA / f"blocos_tematicos_{curso_nome.replace(' ', '_')}.csv"
    arq_adesao = PASTA_SAIDA / f"adesao_classes_{curso_nome.replace(' ', '_')}.csv"
    df_blocos.to_csv(arq_blocos, sep=";", index=False,
                     encoding="utf-8-sig", decimal=",")
    df_adesao.to_csv(arq_adesao, sep=";", index=False,
                     encoding="utf-8-sig", decimal=",")

    return {"curso": curso_nome, "n_ies": n_ies, "n_classes": n_classes,
            "blocos": blocos_ce, "df_blocos": df_blocos, "df_adesao": df_adesao}


# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def main():
    print("=" * 70)
    print("ANÁLISE 2: ESTUDO DE CASO - PADRÕES INTER-INSTITUCIONAIS POR CURSO")
    print("=" * 70)

    df_classes = pd.read_csv(PASTA_DADOS / "02_caracterizacao_das_classes.csv",
                              sep=";", decimal=",")

    todos_blocos = []
    toda_adesao = []
    for curso in CURSOS_FOCO:
        if curso not in df_classes["curso"].unique():
            print(f"AVISO: curso '{curso}' não encontrado.")
            continue
        res = analisar_curso(curso, df_classes)
        todos_blocos.append(res["df_blocos"])
        toda_adesao.append(res["df_adesao"])

    # Consolidar
    pd.concat(todos_blocos, ignore_index=True).to_csv(
        PASTA_SAIDA / "00_consolidado_blocos_tematicos.csv",
        sep=";", index=False, encoding="utf-8-sig", decimal=",")
    pd.concat(toda_adesao, ignore_index=True).to_csv(
        PASTA_SAIDA / "00_consolidado_adesao.csv",
        sep=";", index=False, encoding="utf-8-sig", decimal=",")

    print(f"\n[OK] Resultados em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
