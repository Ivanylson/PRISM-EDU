"""
================================================================================
REANÁLISE: HETEROGENEIDADE HORIZONTAL vs. VERTICAL SEM CENTRALIZAÇÃO
================================================================================

Motivação (resposta ao Revisor 3 / SBIE 2026):
    O critério original de classificação usava os GAPs centralizados na média
    da IES. Com k=2, gap_A(j) = pi_B * (p_A(j) - p_B(j)) e
    gap_B(j) = -pi_A * (p_A(j) - p_B(j)): os sinais são opostos por identidade
    algébrica, o que favorece mecanicamente o "espelhamento" entre classes.

    Esta reanálise usa APENAS as probabilidades de acerto brutas por classe
    (PROB_Q1..Q38, não centralizadas), define um índice formal de
    verticalidade e calibra o critério contra dois modelos nulos simulados:
    (a) IES homogênea (uma única população); (b) IES puramente vertical
    (uma classe uniformemente melhor que a outra).

Definições:
    Para cada IES com k=2, sejam p_A(j), p_B(j) as taxas de acerto observadas
    da classe A e B no item j, e d(j) = p_A(j) - p_B(j).

    Item informativo: |d(j)| >= DELTA (default 0.10) E diferença significativa
    em teste de duas proporções (alpha = 0.05), para descartar ruído amostral.

    Índice de verticalidade:
        V = |#{d(j) > 0} - #{d(j) < 0}| / #informativos, em [0, 1].
        V = 1  -> uma classe domina em todos os itens informativos (vertical)
        V = 0  -> cada classe domina em metade dos itens (horizontal pura)

    Classificação (limiar default V_CUT = 0.5, isto é, a classe "minoritária"
    domina em pelo menos 25% dos itens informativos):
        V <= V_CUT  e  #informativos >= MIN_ITENS  -> HORIZONTAL
        V >  V_CUT  e  #informativos >= MIN_ITENS  -> VERTICAL
        #informativos < MIN_ITENS                   -> INDIFERENCIADA

Modelos nulos (simulação binomial com os n de classe observados):
    NULO-H (homogêneo): p_A* = p_B* = p_pool(j). Mede quantos itens
        "informativos" espúrios o ruído amostral gera e o risco de
        classificar como horizontal uma IES sem estrutura.
    NULO-V (vertical): p_A*(j) = p_pool(j) + shift/2, p_B*(j) = p_pool(j)
        - shift/2, com shift = diferença observada entre as taxas médias das
        classes. Mede o risco de o critério rotular como horizontal uma IES
        genuinamente vertical (taxa de falso-horizontal).

Sensibilidade: DELTA em {0.05, 0.10, 0.15} x V_CUT em {0.4, 0.5, 0.6}.

Entradas esperadas (mesma pasta ou ajustar PASTA_DADOS):
    02_caracterizacao_das_classes.csv  (sep=';', decimal=',', utf-8-sig)
    01_relatorio_geral_por_ies.csv     (opcional, para cruzar k_escolhido)

Saídas (PASTA_SAIDA):
    reanalise_por_ies.csv        - índice V, classificação e nulos por IES
    reanalise_resumo.csv         - % horizontal/vertical por curso e total
    reanalise_sensibilidade.csv  - grade DELTA x V_CUT
    comparacao_criterio_antigo.txt - síntese para a camera-ready
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

# ------------------------------------------------------------------ CONFIG --
PASTA_DADOS = Path(__file__).resolve().parent
PASTA_SAIDA = PASTA_DADOS / "reanalise_saida"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

DELTA = 0.10          # magnitude mínima de |p_A - p_B| para item informativo
ALPHA = 0.05          # significância do teste de duas proporções
V_CUT = 0.50          # V <= V_CUT -> horizontal
MIN_ITENS = 4         # mínimo de itens informativos para classificar
N_SIM_NULO = 500      # simulações por IES em cada modelo nulo
SEED = 42

GRADE_DELTA = [0.05, 0.10, 0.15]
GRADE_VCUT = [0.40, 0.50, 0.60]

rng_global = np.random.default_rng(SEED)


# --------------------------------------------------------------- FUNÇÕES ----
def teste_duas_proporcoes(p1, n1, p2, n2):
    """z-teste bilateral para diferença de proporções. Retorna p-valor (array)."""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(np.clip(p_pool * (1 - p_pool), 1e-12, None) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return 2 * (1 - norm.cdf(np.abs(z)))


def indice_verticalidade(pA, pB, nA, nB, delta=DELTA, alpha=ALPHA):
    """Retorna (V, n_informativos, n_pro_A, n_pro_B)."""
    d = pA - pB
    pvals = teste_duas_proporcoes(pA, nA, pB, nB)
    informativo = (np.abs(d) >= delta) & (pvals < alpha)
    n_info = int(informativo.sum())
    if n_info == 0:
        return np.nan, 0, 0, 0
    pos = int((d[informativo] > 0).sum())
    neg = n_info - pos
    V = abs(pos - neg) / n_info
    return V, n_info, pos, neg


def classificar(V, n_info, v_cut=V_CUT, min_itens=MIN_ITENS):
    if n_info < min_itens or np.isnan(V):
        return "INDIFERENCIADA"
    return "HORIZONTAL" if V <= v_cut else "VERTICAL"


def simular_nulo(p_base_A, p_base_B, nA, nB, n_sim, rng,
                 delta=DELTA, alpha=ALPHA, v_cut=V_CUT, min_itens=MIN_ITENS):
    """Réplicas binomiais vetorizadas; retorna fração classificada HORIZONTAL."""
    J = len(p_base_A)
    pA_hat = rng.binomial(nA, p_base_A, size=(n_sim, J)) / nA
    pB_hat = rng.binomial(nB, p_base_B, size=(n_sim, J)) / nB
    d = pA_hat - pB_hat
    p_pool = (pA_hat * nA + pB_hat * nB) / (nA + nB)
    se = np.sqrt(np.clip(p_pool * (1 - p_pool), 1e-12, None)
                 * (1 / nA + 1 / nB))
    pvals = 2 * (1 - norm.cdf(np.abs(d) / se))
    informativo = (np.abs(d) >= delta) & (pvals < alpha)
    n_info = informativo.sum(axis=1)
    pos = ((d > 0) & informativo).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.abs(2 * pos - n_info) / n_info
    horizontal = (n_info >= min_itens) & (V <= v_cut)
    return float(horizontal.mean())


# ------------------------------------------------------------------ MAIN ----
def main():
    df = pd.read_csv(PASTA_DADOS / "02_caracterizacao_das_classes.csv",
                     sep=";", decimal=",", encoding="utf-8-sig")
    prob_cols = [c for c in df.columns if c.startswith("PROB_Q")]
    assert prob_cols, "Colunas PROB_Q* não encontradas — arquivo errado?"
    print(f"{len(prob_cols)} colunas PROB_Q encontradas; "
          f"{df.groupby(['curso', 'ies']).ngroups} pares curso-IES no arquivo.")

    linhas = []
    for (curso, ies), g in df.groupby(["curso", "ies"]):
        if len(g) != 2:
            continue  # somente soluções k=2, como no artigo (1.643 IES)
        g = g.sort_values("classe_id")
        pA = g.iloc[0][prob_cols].to_numpy(dtype=float)
        pB = g.iloc[1][prob_cols].to_numpy(dtype=float)
        nA = int(g.iloc[0]["n_alunos"])
        nB = int(g.iloc[1]["n_alunos"])

        V, n_info, pro_A, pro_B = indice_verticalidade(pA, pB, nA, nB)
        rotulo = classificar(V, n_info)

        # nulos calibrados na própria IES
        p_pool = (pA * nA + pB * nB) / (nA + nB)
        shift = float(pA.mean() - pB.mean())
        p_vert_A = np.clip(p_pool + shift / 2, 0.01, 0.99)
        p_vert_B = np.clip(p_pool - shift / 2, 0.01, 0.99)
        rng = np.random.default_rng(SEED + hash((str(curso), str(ies))) % 2**31)
        fh_homog = simular_nulo(np.clip(p_pool, 0.01, 0.99),
                                np.clip(p_pool, 0.01, 0.99),
                                nA, nB, N_SIM_NULO, rng)
        fh_vert = simular_nulo(p_vert_A, p_vert_B, nA, nB, N_SIM_NULO, rng)

        linhas.append({
            "curso": curso, "ies": ies, "n_A": nA, "n_B": nB,
            "V": round(V, 4) if not np.isnan(V) else np.nan,
            "n_itens_informativos": n_info,
            "itens_pro_A": pro_A, "itens_pro_B": pro_B,
            "gap_taxa_media": round(abs(shift), 4),
            "classificacao": rotulo,
            "falso_horizontal_nulo_homogeneo": round(fh_homog, 3),
            "falso_horizontal_nulo_vertical": round(fh_vert, 3),
        })

    res = pd.DataFrame(linhas)
    res.to_csv(PASTA_SAIDA / "reanalise_por_ies.csv", sep=";", index=False,
               encoding="utf-8-sig", decimal=",")

    # ------------------------------------------------------------- resumo --
    total = len(res)
    dist = res["classificacao"].value_counts()
    pct_horiz = 100 * dist.get("HORIZONTAL", 0) / total
    pct_vert = 100 * dist.get("VERTICAL", 0) / total
    pct_ind = 100 * dist.get("INDIFERENCIADA", 0) / total

    resumo = (res.groupby("curso")["classificacao"]
                 .value_counts(normalize=True).unstack(fill_value=0)
                 .mul(100).round(1))
    resumo["n_ies"] = res.groupby("curso").size()
    resumo.to_csv(PASTA_SAIDA / "reanalise_resumo.csv", sep=";",
                  encoding="utf-8-sig", decimal=",")

    # ------------------------------------------------------ sensibilidade --
    sens = []
    for dlt in GRADE_DELTA:
        for vc in GRADE_VCUT:
            rotulos = []
            for (curso, ies), g in df.groupby(["curso", "ies"]):
                if len(g) != 2:
                    continue
                g = g.sort_values("classe_id")
                pA = g.iloc[0][prob_cols].to_numpy(dtype=float)
                pB = g.iloc[1][prob_cols].to_numpy(dtype=float)
                nA = int(g.iloc[0]["n_alunos"])
                nB = int(g.iloc[1]["n_alunos"])
                V, n_info, _, _ = indice_verticalidade(pA, pB, nA, nB, dlt)
                rotulos.append(classificar(V, n_info, vc))
            s = pd.Series(rotulos).value_counts(normalize=True) * 100
            sens.append({"delta": dlt, "v_cut": vc,
                         "pct_horizontal": round(s.get("HORIZONTAL", 0), 1),
                         "pct_vertical": round(s.get("VERTICAL", 0), 1),
                         "pct_indiferenciada": round(s.get("INDIFERENCIADA", 0), 1)})
    pd.DataFrame(sens).to_csv(PASTA_SAIDA / "reanalise_sensibilidade.csv",
                              sep=";", index=False, encoding="utf-8-sig",
                              decimal=",")

    # ------------------------------------------------------------ síntese --
    fh_v_mediano = res["falso_horizontal_nulo_vertical"].median()
    fh_h_mediano = res["falso_horizontal_nulo_homogeneo"].median()
    texto = (
        "REANALISE SEM CENTRALIZACAO - SINTESE\n"
        + "=" * 60 + "\n"
        + f"IES k=2 analisadas: {total}\n"
        + f"HORIZONTAL: {dist.get('HORIZONTAL', 0)} ({pct_horiz:.1f}%)\n"
        + f"VERTICAL:   {dist.get('VERTICAL', 0)} ({pct_vert:.1f}%)\n"
        + f"INDIFERENCIADA: {dist.get('INDIFERENCIADA', 0)} ({pct_ind:.1f}%)\n"
        + f"\nCriterio: itens informativos = |d| >= {DELTA} e p < {ALPHA}; "
        + f"horizontal se V <= {V_CUT} com >= {MIN_ITENS} itens informativos.\n"
        + f"\nNulo vertical: mediana da taxa de falso-horizontal = {fh_v_mediano:.3f}\n"
        + f"Nulo homogeneo: mediana = {fh_h_mediano:.3f}\n"
        + "\nComparacao: o artigo reportou 90,3% de heterogeneidade horizontal "
        + "com o criterio de espelhamento sobre gaps centralizados.\n"
    )
    (PASTA_SAIDA / "comparacao_criterio_antigo.txt").write_text(
        texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
