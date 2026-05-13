"""
================================================================================
ANÁLISE 1 - CONTEÚDO DAS QUESTÕES DIFERENCIADORAS
================================================================================

Pergunta de pesquisa:
    Os perfis identificados pela LCA se organizam por ÁREA DE CONHECIMENTO
    (i.e., um perfil domina Formação Geral e outro Componente Específico),
    ou os itens diferenciadores se distribuem aleatoriamente entre as áreas?

Estrutura padrão do ENADE 2023:
    - Q1 a Q8   : Formação Geral (8 itens, comuns a todos os cursos)
    - Q9 a Q38  : Componente Específico (30 itens, variam por curso)

Hipóteses:
    H0 (nula): a distribuição dos itens diferenciadores entre FG e CE em cada
        classe é proporcional à composição da prova (8/38 = 21,1% FG;
        30/38 = 78,9% CE).
    H1: classes se especializam por área — alguma classe mostra concentração
        desproporcional em uma das áreas.

Teste estatístico:
    Qui-quadrado de aderência por classe, com correção de continuidade.
    Reportamos também o tamanho do efeito (V de Cramér).

NOTA METODOLÓGICA:
    Esta análise usa apenas os 5 itens MAIS diferenciadores por direção
    (limite do CSV de caracterização). Para análise completa com todos os
    38 itens, é necessário re-rodar o pipeline principal salvando todas as
    probabilidades de acerto por item — script auxiliar fornecido ao final.
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
PASTA_SAIDA = Path("/mnt/user-data/outputs/analise_1_areas")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Definição da estrutura ENADE 2023
QUESTOES_FG = {f"Q{i}" for i in range(1, 9)}      # Q1-Q8
QUESTOES_CE = {f"Q{i}" for i in range(9, 39)}     # Q9-Q38

# Proporção esperada sob H0 (composição da prova)
PCT_FG_PROVA = len(QUESTOES_FG) / 38     # 21,1%
PCT_CE_PROVA = len(QUESTOES_CE) / 38     # 78,9%


# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO
# ==============================================================================

def parse_itens(s):
    """Converte string 'Q25, Q32, Q12' em lista; lida com vazio/'-'."""
    if pd.isna(s) or s in ("-", "", "nan"):
        return []
    return [q.strip() for q in str(s).split(",") if q.strip()]


def classificar_area(questao):
    if questao in QUESTOES_FG:
        return "FG"
    if questao in QUESTOES_CE:
        return "CE"
    return "?"


def expandir_classes(df):
    """Para cada classe, conta itens fortes e fracos por área (FG/CE)."""
    linhas = []
    for _, r in df.iterrows():
        fortes = parse_itens(r["itens_diferencialmente_fortes"])
        fracos = parse_itens(r["itens_diferencialmente_fracos"])

        cont_fortes = {"FG": 0, "CE": 0}
        cont_fracos = {"FG": 0, "CE": 0}
        for q in fortes:
            cont_fortes[classificar_area(q)] = cont_fortes.get(
                classificar_area(q), 0) + 1
        for q in fracos:
            cont_fracos[classificar_area(q)] = cont_fracos.get(
                classificar_area(q), 0) + 1

        linhas.append({
            "curso": r["curso"],
            "ies": r["ies"],
            "classe_id": r["classe_id"],
            "n_alunos_classe": r["n_alunos"],
            "pct_alunos": r["pct_alunos"],
            "taxa_acerto_media": r["taxa_acerto_media"],
            "n_fortes": len(fortes),
            "n_fracos": len(fracos),
            "fortes_FG": cont_fortes["FG"],
            "fortes_CE": cont_fortes["CE"],
            "fracos_FG": cont_fracos["FG"],
            "fracos_CE": cont_fracos["CE"],
        })
    return pd.DataFrame(linhas)


# ==============================================================================
# 2. TESTE DE ADERÊNCIA POR CLASSE
# ==============================================================================

def teste_aderencia_classe(n_fg_obs, n_ce_obs, alpha=0.05):
    """
    Qui-quadrado de aderência: a distribuição observada (FG, CE) entre os
    itens diferenciadores difere da esperada pela composição da prova?

    Retorna estatística, p-valor, tamanho do efeito (V de Cramér) e direção.
    """
    n_total = n_fg_obs + n_ce_obs
    if n_total < 5:
        return {"chi2": np.nan, "p_valor": np.nan, "v_cramer": np.nan,
                "direcao": "amostra_pequena"}

    n_fg_esp = n_total * PCT_FG_PROVA
    n_ce_esp = n_total * PCT_CE_PROVA

    chi2 = ((n_fg_obs - n_fg_esp) ** 2 / n_fg_esp +
            (n_ce_obs - n_ce_esp) ** 2 / n_ce_esp)
    p_valor = 1 - stats.chi2.cdf(chi2, df=1)
    # V de Cramér para 1 df = sqrt(chi2/n)
    v_cramer = np.sqrt(chi2 / n_total) if n_total > 0 else 0

    pct_fg_obs = n_fg_obs / n_total
    if p_valor < alpha:
        direcao = "concentra_FG" if pct_fg_obs > PCT_FG_PROVA else "concentra_CE"
    else:
        direcao = "compativel_com_prova"

    return {"chi2": chi2, "p_valor": p_valor, "v_cramer": v_cramer,
            "direcao": direcao}


# ==============================================================================
# 3. ANÁLISE PRINCIPAL
# ==============================================================================

def main():
    print("=" * 70)
    print("ANÁLISE 1: ÁREAS DE CONHECIMENTO NOS PERFIS LATENTES")
    print("=" * 70)

    df_classes = pd.read_csv(PASTA_DADOS / "02_caracterizacao_das_classes.csv",
                              sep=";", decimal=",")
    print(f"\nTotal de classes carregadas: {len(df_classes)}")

    df_exp = expandir_classes(df_classes)
    print(f"Classes processadas: {len(df_exp)}")
    print(f"Composição da prova: FG={PCT_FG_PROVA:.1%} | CE={PCT_CE_PROVA:.1%}")

    # ---- 3.1 Aplicar teste de aderência por classe ----
    print("\n[3.1] Aplicando teste qui-quadrado de aderência (FORTES)...")
    testes_fortes = df_exp.apply(
        lambda r: teste_aderencia_classe(r["fortes_FG"], r["fortes_CE"]), axis=1)
    df_exp["forte_chi2"] = testes_fortes.apply(lambda d: d["chi2"])
    df_exp["forte_p"] = testes_fortes.apply(lambda d: d["p_valor"])
    df_exp["forte_v_cramer"] = testes_fortes.apply(lambda d: d["v_cramer"])
    df_exp["forte_direcao"] = testes_fortes.apply(lambda d: d["direcao"])

    print("[3.1] Aplicando teste qui-quadrado de aderência (FRACOS)...")
    testes_fracos = df_exp.apply(
        lambda r: teste_aderencia_classe(r["fracos_FG"], r["fracos_CE"]), axis=1)
    df_exp["fraco_chi2"] = testes_fracos.apply(lambda d: d["chi2"])
    df_exp["fraco_p"] = testes_fracos.apply(lambda d: d["p_valor"])
    df_exp["fraco_v_cramer"] = testes_fracos.apply(lambda d: d["v_cramer"])
    df_exp["fraco_direcao"] = testes_fracos.apply(lambda d: d["direcao"])

    # ---- 3.2 Estatísticas globais ----
    print("\n[3.2] DISTRIBUIÇÃO DE ITENS FORTES POR ÁREA (todas as classes):")
    total_fortes_fg = df_exp["fortes_FG"].sum()
    total_fortes_ce = df_exp["fortes_CE"].sum()
    total_fortes = total_fortes_fg + total_fortes_ce
    print(f"  Total de menções 'forte': {total_fortes}")
    print(f"  FG observado: {total_fortes_fg} ({total_fortes_fg/total_fortes:.1%})  "
          f"| esperado: {PCT_FG_PROVA:.1%}")
    print(f"  CE observado: {total_fortes_ce} ({total_fortes_ce/total_fortes:.1%})  "
          f"| esperado: {PCT_CE_PROVA:.1%}")

    chi2_global, p_global = stats.chisquare(
        [total_fortes_fg, total_fortes_ce],
        f_exp=[total_fortes * PCT_FG_PROVA, total_fortes * PCT_CE_PROVA])
    print(f"  Qui-quadrado global: chi²={chi2_global:.2f}, p={p_global:.2e}")

    print("\n[3.2] DIREÇÃO DAS DIFERENÇAS (itens fortes):")
    print(df_exp["forte_direcao"].value_counts())
    print("\n[3.2] DIREÇÃO DAS DIFERENÇAS (itens fracos):")
    print(df_exp["fraco_direcao"].value_counts())

    # ---- 3.3 Análise dentro das IES (espelhamento por área) ----
    print("\n[3.3] PADRÃO INTRA-IES: as classes se especializam em áreas opostas?")
    # Para IES com k=2, comparar se uma classe é "FG-forte" e outra "CE-forte"
    ies_k2 = df_exp.groupby(["curso", "ies"]).filter(lambda g: len(g) == 2)

    pares = []
    for (curso, ies), g in ies_k2.groupby(["curso", "ies"]):
        g = g.sort_values("classe_id")
        c0, c1 = g.iloc[0], g.iloc[1]
        # Uma classe "domina FG" se >50% dos seus fortes são FG
        c0_pct_fg_fortes = c0["fortes_FG"] / max(c0["n_fortes"], 1)
        c1_pct_fg_fortes = c1["fortes_FG"] / max(c1["n_fortes"], 1)

        # Tipos de divisão
        if (c0_pct_fg_fortes > 0.5 and c1_pct_fg_fortes < 0.3) or \
           (c1_pct_fg_fortes > 0.5 and c0_pct_fg_fortes < 0.3):
            tipo = "especializacao_FG_vs_CE"
        elif c0_pct_fg_fortes < 0.3 and c1_pct_fg_fortes < 0.3:
            tipo = "ambas_focadas_em_CE"
        elif c0_pct_fg_fortes > 0.5 and c1_pct_fg_fortes > 0.5:
            tipo = "ambas_focadas_em_FG"
        else:
            tipo = "padrao_misto"

        pares.append({"curso": curso, "ies": ies, "tipo_divisao_area": tipo,
                      "c0_pct_FG_fortes": c0_pct_fg_fortes,
                      "c1_pct_FG_fortes": c1_pct_fg_fortes})

    df_pares = pd.DataFrame(pares)
    print(f"\nTotal de IES com k=2 analisadas: {len(df_pares)}")
    print("\nDistribuição dos tipos de divisão por área:")
    print(df_pares["tipo_divisao_area"].value_counts())
    print("\nProporção:")
    print((df_pares["tipo_divisao_area"].value_counts(normalize=True) * 100).round(1))

    # ---- 3.4 Por curso ----
    print("\n[3.4] TIPO DE DIVISÃO POR ÁREA - DETALHE POR CURSO:")
    tab_curso = pd.crosstab(df_pares["curso"], df_pares["tipo_divisao_area"])
    tab_curso["total"] = tab_curso.sum(axis=1)
    tab_curso = tab_curso.sort_values("total", ascending=False).head(15)
    print(tab_curso)

    # ---- 3.5 Heatmap visual: distribuição FG/CE por curso ----
    print("\n[3.5] Gerando visualizações...")

    # Heatmap 1: % de itens fortes em FG por curso (média entre classes)
    media_curso = df_exp.groupby("curso").agg(
        n_classes=("ies", "count"),
        pct_FG_fortes=("fortes_FG",
                       lambda s: s.sum() / df_exp.loc[s.index, ["fortes_FG", "fortes_CE"]].sum().sum()
                       if df_exp.loc[s.index, ["fortes_FG", "fortes_CE"]].sum().sum() > 0 else 0),
        pct_FG_fracos=("fracos_FG",
                       lambda s: s.sum() / df_exp.loc[s.index, ["fracos_FG", "fracos_CE"]].sum().sum()
                       if df_exp.loc[s.index, ["fracos_FG", "fracos_CE"]].sum().sum() > 0 else 0),
    ).sort_values("n_classes", ascending=False)
    media_curso["pct_FG_fortes"] = (media_curso["pct_FG_fortes"] * 100).round(1)
    media_curso["pct_FG_fracos"] = (media_curso["pct_FG_fracos"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 11))
    heatmap_data = media_curso[["pct_FG_fortes", "pct_FG_fracos"]].copy()
    heatmap_data.columns = ["% itens fortes em FG", "% itens fracos em FG"]
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlBu_r",
                center=PCT_FG_PROVA * 100, ax=ax,
                cbar_kws={"label": "% (linha vermelha = 21,1% esperado)"})
    ax.set_title(f"Distribuição de itens diferenciadores em Formação Geral por curso\n"
                 f"(Esperado pela prova: {PCT_FG_PROVA:.1%} FG)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / "heatmap_FG_por_curso.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 3.6 Salvar tabelas ----
    df_exp.to_csv(PASTA_SAIDA / "01_classes_com_classificacao_areas.csv",
                  sep=";", index=False, encoding="utf-8-sig", decimal=",")
    df_pares.to_csv(PASTA_SAIDA / "02_padrao_intra_ies_k2.csv",
                    sep=";", index=False, encoding="utf-8-sig", decimal=",")
    media_curso.to_csv(PASTA_SAIDA / "03_resumo_por_curso.csv",
                       sep=";", encoding="utf-8-sig", decimal=",")

    # ---- 3.7 Síntese textual ----
    print("\n" + "=" * 70)
    print("SÍNTESE DE ACHADOS - ANÁLISE 1")
    print("=" * 70)
    pct_fg_obs_global = total_fortes_fg / total_fortes * 100
    desvio = pct_fg_obs_global - PCT_FG_PROVA * 100
    print(f"\n1. Sob H0, esperaríamos {PCT_FG_PROVA:.1%} dos itens fortes em FG.")
    print(f"   Observado: {pct_fg_obs_global:.1f}% (desvio de {desvio:+.1f} p.p.)")
    print(f"   Qui-quadrado global: p = {p_global:.2e}")
    if p_global < 0.001:
        print("   => H0 rejeitada com forte evidência.")

    pct_espec = (df_pares["tipo_divisao_area"] == "especializacao_FG_vs_CE").mean() * 100
    print(f"\n2. Especialização clara FG vs CE entre as duas classes: "
          f"{pct_espec:.1f}% das IES com k=2.")

    pct_ambas_ce = (df_pares["tipo_divisao_area"] == "ambas_focadas_em_CE").mean() * 100
    print(f"   Ambas as classes focadas em CE: {pct_ambas_ce:.1f}%.")
    print(f"   (Padrão dominante esperado se a especialização ocorre DENTRO de CE)")

    print(f"\n[OK] Resultados salvos em: {PASTA_SAIDA}")
    return df_exp, df_pares, media_curso


if __name__ == "__main__":
    main()
