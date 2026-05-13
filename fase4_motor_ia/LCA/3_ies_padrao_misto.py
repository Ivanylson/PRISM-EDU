"""
================================================================================
ANÁLISE 3 - INVESTIGAÇÃO DAS IES COM PADRÃO MISTO OU FRACO
================================================================================

Pergunta de pesquisa:
    144 IES (8,8% das k=2) não exibiram espelhamento claro entre as classes.
    O que essas IES têm em comum? São artefatos do método? São casos
    pedagogicamente diferentes (ex: turmas mais homogêneas)? Devemos
    excluí-las, tratá-las separadamente, ou interpretá-las?

Hipóteses:
    H1: IES "mistas" são MENORES (instabilidade estatística da LCA)
    H2: IES "mistas" têm MAIOR HOMOGENEIDADE (taxa de acerto similar entre
        as classes; perfil único na verdade)
    H3: IES "mistas" têm MENOR ENTROPIA (separação fraca entre classes)
    H4: IES "mistas" têm MENOR ESTABILIDADE de bootstrap
    H5: IES "mistas" se concentram em CURSOS específicos
    H6: IES "mistas" têm DESEMPENHO geral diferente (alta ou baixa)

Decisão prática:
    Com base nos achados, recomendar uma das três opções para a dissertação:
    (a) Manter as 144 IES como categoria separada de análise
    (b) Excluí-las com justificativa estatística
    (c) Reanalisá-las com k=1 (perfil único) ou método alternativo
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

PASTA_DADOS = Path("/mnt/user-data/uploads")
PASTA_SAIDA = Path("/mnt/user-data/outputs/analise_3_ies_mistas")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# FUNÇÕES
# ==============================================================================

def parse_itens(s):
    if pd.isna(s) or s in ("-", "", "nan"):
        return []
    return [q.strip() for q in str(s).split(",") if q.strip()]


def classificar_divisao(g):
    """Replica a lógica de classificação que rodamos antes."""
    if len(g) != 2:
        return None
    g = g.sort_values("classe_id")
    c0, c1 = g.iloc[0], g.iloc[1]
    f0 = set(parse_itens(c0["itens_diferencialmente_fortes"]))
    f1 = set(parse_itens(c1["itens_diferencialmente_fortes"]))
    fra1 = set(parse_itens(c1["itens_diferencialmente_fracos"]))
    fra0 = set(parse_itens(c0["itens_diferencialmente_fracos"]))

    espelha_01 = len(f0 & fra1) / max(len(f0 | fra1), 1)
    espelha_10 = len(f1 & fra0) / max(len(f1 | fra0), 1)
    espelhamento = (espelha_01 + espelha_10) / 2
    gap = abs(c0["taxa_acerto_media"] - c1["taxa_acerto_media"])

    if espelhamento >= 0.5:
        tipo = "PERFIL_POR_CONTEUDO"
    elif gap >= 0.15:
        tipo = "NIVEL_GERAL"
    else:
        tipo = "MISTO_OU_FRACO"

    return {"tipo_divisao": tipo, "espelhamento": espelhamento, "gap_taxa": gap,
            "taxa_classe_menor_id": c0["taxa_acerto_media"],
            "taxa_classe_maior_id": c1["taxa_acerto_media"],
            "pct_classe_minoritaria": min(c0["pct_alunos"], c1["pct_alunos"])}


# ==============================================================================
# ANÁLISE PRINCIPAL
# ==============================================================================

def main():
    print("=" * 70)
    print("ANÁLISE 3: INVESTIGAÇÃO DAS IES COM PADRÃO MISTO OU FRACO")
    print("=" * 70)

    df_geral = pd.read_csv(PASTA_DADOS / "01_relatorio_geral_por_ies.csv",
                            sep=";", decimal=",")
    df_classes = pd.read_csv(PASTA_DADOS / "02_caracterizacao_das_classes.csv",
                              sep=";", decimal=",")

    # Reaplicar classificação para IES com k=2
    classificacoes = []
    for (curso, ies), g in df_classes.groupby(["curso", "ies"]):
        if len(g) == 2:
            res = classificar_divisao(g)
            if res:
                res["curso"] = curso
                res["ies"] = ies
                classificacoes.append(res)

    df_class = pd.DataFrame(classificacoes)
    df_full = df_geral.merge(df_class, on=["curso", "ies"], how="left")

    print(f"\n[1] DISTRIBUIÇÃO DAS CATEGORIAS")
    print(df_full["tipo_divisao"].value_counts(dropna=False))

    # ---- Filtrar k=2 com classificação ----
    df_k2 = df_full[df_full["k_escolhido"] == 2].copy()
    print(f"\nIES com k=2 a analisar: {len(df_k2)}")

    # ==========================================================================
    # H1: TAMANHO AMOSTRAL
    # ==========================================================================
    print(f"\n{'='*70}\n[H1] IES mistas são MENORES?\n{'='*70}")
    print("\nTamanho (n_alunos) por tipo de divisão:")
    print(df_k2.groupby("tipo_divisao")["n_alunos"].describe().round(1))

    grupos = {t: df_k2[df_k2["tipo_divisao"] == t]["n_alunos"].values
              for t in df_k2["tipo_divisao"].dropna().unique()}
    if "MISTO_OU_FRACO" in grupos and "PERFIL_POR_CONTEUDO" in grupos:
        u, p = stats.mannwhitneyu(grupos["MISTO_OU_FRACO"],
                                    grupos["PERFIL_POR_CONTEUDO"],
                                    alternative="two-sided")
        print(f"\nMann-Whitney U: U={u:.0f}, p={p:.4e}")
        med_misto = np.median(grupos["MISTO_OU_FRACO"])
        med_cont = np.median(grupos["PERFIL_POR_CONTEUDO"])
        print(f"Mediana MISTO: {med_misto:.0f} | Mediana CONTEÚDO: {med_cont:.0f}")
        print(f"=> {'CONFIRMADA' if p<0.05 and med_misto<med_cont else 'NÃO confirmada'}: "
              f"hipótese de que IES mistas são menores")

    # ==========================================================================
    # H2: HOMOGENEIDADE INTERNA
    # ==========================================================================
    print(f"\n{'='*70}\n[H2] IES mistas são mais HOMOGÊNEAS internamente?\n{'='*70}")
    print("\nGap de taxa de acerto entre as duas classes:")
    print(df_k2.groupby("tipo_divisao")["gap_taxa"].describe().round(4))

    print("\n% da classe minoritária:")
    print(df_k2.groupby("tipo_divisao")["pct_classe_minoritaria"].describe().round(2))

    # ==========================================================================
    # H3: ENTROPIA NORMALIZADA
    # ==========================================================================
    print(f"\n{'='*70}\n[H3] IES mistas têm ENTROPIA mais baixa?\n{'='*70}")
    print(df_k2.groupby("tipo_divisao")["entropia_normalizada"].describe().round(4))
    if "MISTO_OU_FRACO" in grupos:
        ent_misto = df_k2[df_k2["tipo_divisao"] == "MISTO_OU_FRACO"]["entropia_normalizada"]
        ent_cont = df_k2[df_k2["tipo_divisao"] == "PERFIL_POR_CONTEUDO"]["entropia_normalizada"]
        u, p = stats.mannwhitneyu(ent_misto, ent_cont, alternative="two-sided")
        print(f"\nMann-Whitney U: U={u:.0f}, p={p:.4e}")
        print(f"Mediana MISTO: {ent_misto.median():.4f} | Mediana CONTEÚDO: {ent_cont.median():.4f}")

    # ==========================================================================
    # H4: ESTABILIDADE BOOTSTRAP
    # ==========================================================================
    print(f"\n{'='*70}\n[H4] IES mistas são MENOS ESTÁVEIS no bootstrap?\n{'='*70}")
    print(df_k2.groupby("tipo_divisao")["jaccard_medio_bootstrap"].describe().round(4))
    if "MISTO_OU_FRACO" in grupos:
        j_misto = df_k2[df_k2["tipo_divisao"] == "MISTO_OU_FRACO"]["jaccard_medio_bootstrap"]
        j_cont = df_k2[df_k2["tipo_divisao"] == "PERFIL_POR_CONTEUDO"]["jaccard_medio_bootstrap"]
        u, p = stats.mannwhitneyu(j_misto, j_cont, alternative="two-sided")
        print(f"\nMann-Whitney U: U={u:.0f}, p={p:.4e}")
        print(f"Mediana MISTO: {j_misto.median():.4f} | Mediana CONTEÚDO: {j_cont.median():.4f}")

    # ==========================================================================
    # H5: CONCENTRAÇÃO POR CURSO
    # ==========================================================================
    print(f"\n{'='*70}\n[H5] IES mistas se CONCENTRAM em cursos específicos?\n{'='*70}")
    tab_curso = pd.crosstab(df_k2["curso"], df_k2["tipo_divisao"])
    tab_curso["total"] = tab_curso.sum(axis=1)
    tab_curso["pct_misto"] = (tab_curso.get("MISTO_OU_FRACO", 0) /
                               tab_curso["total"] * 100).round(1)
    tab_curso = tab_curso.sort_values("pct_misto", ascending=False)
    print("\nCursos com maior % de IES mistas:")
    print(tab_curso[tab_curso["total"] >= 10].head(10))

    # Teste qui-quadrado de independência
    if "MISTO_OU_FRACO" in tab_curso.columns:
        cont = tab_curso[["MISTO_OU_FRACO", "PERFIL_POR_CONTEUDO"]].dropna()
        cont = cont[(cont.sum(axis=1) >= 5)]
        chi2, p, _, _ = stats.chi2_contingency(cont)
        print(f"\nQui-quadrado independência curso × tipo: chi²={chi2:.2f}, p={p:.4e}")

    # ==========================================================================
    # H6: DESEMPENHO GERAL
    # ==========================================================================
    print(f"\n{'='*70}\n[H6] IES mistas têm DESEMPENHO geral diferente?\n{'='*70}")
    df_k2["taxa_media_ies"] = (df_k2["taxa_classe_menor_id"] + df_k2["taxa_classe_maior_id"]) / 2
    print(df_k2.groupby("tipo_divisao")["taxa_media_ies"].describe().round(4))

    # ==========================================================================
    # VISUALIZAÇÕES
    # ==========================================================================
    print(f"\n{'='*70}\nGerando visualizações...\n{'='*70}")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # H1
    sns.boxplot(data=df_k2, x="tipo_divisao", y="n_alunos", ax=axes[0, 0],
                order=["PERFIL_POR_CONTEUDO", "MISTO_OU_FRACO", "NIVEL_GERAL"])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("H1: Tamanho amostral (escala log)")
    axes[0, 0].tick_params(axis="x", rotation=20)

    # H2
    sns.boxplot(data=df_k2, x="tipo_divisao", y="gap_taxa", ax=axes[0, 1],
                order=["PERFIL_POR_CONTEUDO", "MISTO_OU_FRACO", "NIVEL_GERAL"])
    axes[0, 1].set_title("H2: Gap entre taxas de acerto das classes")
    axes[0, 1].tick_params(axis="x", rotation=20)

    # H3
    sns.boxplot(data=df_k2, x="tipo_divisao", y="entropia_normalizada",
                ax=axes[0, 2],
                order=["PERFIL_POR_CONTEUDO", "MISTO_OU_FRACO", "NIVEL_GERAL"])
    axes[0, 2].set_title("H3: Entropia normalizada")
    axes[0, 2].tick_params(axis="x", rotation=20)

    # H4
    sns.boxplot(data=df_k2, x="tipo_divisao", y="jaccard_medio_bootstrap",
                ax=axes[1, 0],
                order=["PERFIL_POR_CONTEUDO", "MISTO_OU_FRACO", "NIVEL_GERAL"])
    axes[1, 0].set_title("H4: Estabilidade (Jaccard bootstrap)")
    axes[1, 0].tick_params(axis="x", rotation=20)

    # H5
    pct_curso = tab_curso[tab_curso["total"] >= 10]["pct_misto"].sort_values()
    pct_curso.plot.barh(ax=axes[1, 1], color="coral")
    axes[1, 1].set_title("H5: % de IES mistas por curso")
    axes[1, 1].set_xlabel("%")

    # H6
    sns.boxplot(data=df_k2, x="tipo_divisao", y="taxa_media_ies", ax=axes[1, 2],
                order=["PERFIL_POR_CONTEUDO", "MISTO_OU_FRACO", "NIVEL_GERAL"])
    axes[1, 2].set_title("H6: Taxa média de acerto da IES")
    axes[1, 2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / "diagnostico_ies_mistas.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # ---- Salvar tabelas ----
    df_full.to_csv(PASTA_SAIDA / "00_relatorio_com_classificacao.csv",
                   sep=";", index=False, encoding="utf-8-sig", decimal=",")
    tab_curso.to_csv(PASTA_SAIDA / "01_distribuicao_por_curso.csv",
                     sep=";", encoding="utf-8-sig", decimal=",")
    df_k2[df_k2["tipo_divisao"] == "MISTO_OU_FRACO"].to_csv(
        PASTA_SAIDA / "02_lista_ies_mistas.csv",
        sep=";", index=False, encoding="utf-8-sig", decimal=",")

    # ==========================================================================
    # SÍNTESE E RECOMENDAÇÃO
    # ==========================================================================
    print(f"\n{'='*70}\nSÍNTESE E RECOMENDAÇÃO PARA A DISSERTAÇÃO\n{'='*70}")

    n_misto = (df_k2["tipo_divisao"] == "MISTO_OU_FRACO").sum()
    n_total = len(df_k2)
    print(f"\nTotal de IES com k=2: {n_total}")
    print(f"IES classificadas como MISTAS: {n_misto} ({n_misto/n_total*100:.1f}%)")

    print(f"\n[OK] Resultados salvos em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
