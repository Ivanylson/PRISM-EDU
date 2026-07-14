"""
================================================================================
ROBUSTEZ AO DESENGAJAMENTO: BRANCOS COMO FALTANTES E POSIÇÃO DO ITEM
================================================================================

Motivação (resposta ao Revisor 2 / SBIE 2026):
    O ENADE não afeta a nota individual do estudante; respostas em branco
    podem refletir desengajamento, não ausência de domínio. A convenção do
    INEP (adotada no pipeline) codifica branco como erro. Este script testa
    se os OCs cronicamente deficientes sobrevivem à codificação alternativa,
    em que a probabilidade de acerto é calculada apenas entre estudantes que
    tentaram o item ('1' ou '0' no vetor de acertos), e se a taxa de brancos
    cresce com a posição do item na prova (assinatura de fadiga/abandono).

Códigos do vetor DS_VT_ACE_* preservados nas colunas Q1..Q38:
    '1' acerto | '0' erro | '9' branco/rasura | '8' anulada | '.' discursiva

Entradas (mesma pasta ou ajustar caminhos):
    relatorio_final_enade_2023.csv   (saída da fase 1)
    mestre_questao_oc.csv            (mapeamento item-OC do INEP)
    diagnostico_oc_arquetipos.csv    (diagnóstico por arquétipo)

Saídas:
    robustez_brancos_por_item.csv
    robustez_brancos_por_oc.csv
    e um relatório impresso: sobrevivência dos OCs cronicamente deficientes
    e correlação posição x taxa de branco por curso.
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parent
CURSOS = {"5710": "Engenharia Civil", "12": "Medicina", "23": "Enfermagem"}
MIN_ALUNOS_IES = 50   # mesmo corte do pipeline principal
QCOLS = [f"Q{i}" for i in range(1, 39)]


def main():
    df = pd.read_csv(PASTA / "relatorio_final_enade_2023.csv", sep=";",
                     dtype=str, usecols=["CO_GRUPO", "CO_IES"] + QCOLS)
    df = df[df["CO_GRUPO"].isin(CURSOS)].copy()
    df["curso"] = df["CO_GRUPO"].map(CURSOS)
    tam = df.groupby(["curso", "CO_IES"]).size()
    validas = tam[tam >= MIN_ALUNOS_IES].index
    df = df.set_index(["curso", "CO_IES"]).loc[validas].reset_index()
    print(f"alunos nos 3 cursos (IES >= {MIN_ALUNOS_IES}): {len(df)} | "
          f"IES: {len(validas)}")

    oc = pd.read_csv(PASTA / "mestre_questao_oc.csv", sep=";")
    oc = oc[oc["curso"].isin(CURSOS.values())][
        ["curso", "questao", "oc_principal", "area_prova", "posicao"]]

    linhas = []
    for curso, g in df.groupby("curso"):
        for q in QCOLS:
            v = g[q]
            n1 = (v == "1").sum(); n0 = (v == "0").sum(); n9 = (v == "9").sum()
            tot = n1 + n0 + n9
            if tot == 0:
                continue
            linhas.append({
                "curso": curso, "questao": q,
                "p_inep": n1 / tot,                              # branco = erro
                "p_tentou": n1 / (n1 + n0) if n1 + n0 else np.nan,  # branco = faltante
                "taxa_branco": n9 / tot, "n_valid": tot})
    item = pd.DataFrame(linhas).merge(oc, on=["curso", "questao"], how="left")
    item.to_csv(PASTA / "robustez_brancos_por_item.csv", sep=";", index=False,
                decimal=",", encoding="utf-8-sig")

    ko = (item.dropna(subset=["oc_principal"])
              .groupby(["curso", "oc_principal"])
              .agg(p_inep=("p_inep", "mean"), p_tentou=("p_tentou", "mean"),
                   taxa_branco=("taxa_branco", "mean"),
                   n_questoes=("questao", "count"))
              .reset_index())
    ko.to_csv(PASTA / "robustez_brancos_por_oc.csv", sep=";", index=False,
              decimal=",", encoding="utf-8-sig")

    d = pd.read_csv(PASTA / "diagnostico_oc_arquetipos.csv", sep=";")
    piv = d.pivot_table(index=["curso", "oc"], columns="arquetipo",
                        values="prob_acerto_media")
    cron = piv[(piv < 0.50).all(axis=1)].reset_index()[["curso", "oc"]]
    cron["oc_short"] = cron["oc"].str.strip()
    ko["oc_short"] = ko["oc_principal"].str.strip()
    m = cron.merge(ko.drop(columns=["oc_principal"]),
                   on=["curso", "oc_short"], how="left")
    m["sobrevive"] = m["p_tentou"] < 0.50
    print("\nOCs cronicamente deficientes sob a codificação 'só quem tentou':")
    print(m[["curso", "oc_short", "p_inep", "p_tentou", "taxa_branco",
             "sobrevive"]].round(3).to_string(index=False))
    print(f"\nsobreviventes: {int(m['sobrevive'].sum())} de {len(m)}")

    ce = item[item["area_prova"] == "CE"].dropna(subset=["posicao"])
    print()
    for curso, g in ce.groupby("curso"):
        r = np.corrcoef(g["posicao"].astype(float), g["taxa_branco"])[0, 1]
        print(f"posição x taxa de branco ({curso}): r = {r:.3f} | "
              f"branco médio = {g['taxa_branco'].mean():.3f}")


if __name__ == "__main__":
    main()
