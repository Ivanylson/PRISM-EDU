"""
================================================================================
FIGURA 5 (CORRIGIDA): OCs CRONICAMENTE DEFICIENTES POR CURSO
================================================================================

A versão anterior da figura usava um filtro diferente do critério da Tabela 1
(prob < 0,50 nos DOIS arquétipos), o que produzia contagens divergentes
(12/2/11 na figura contra 11/2/4 na tabela). Esta versão parte do arquivo
canônico diagnostico_oc_arquetipos.csv e aplica exatamente o critério da
tabela, garantindo consistência entre figura, tabela e texto.

Entrada:  diagnostico_oc_arquetipos.csv (mesma pasta)
Saída:    figura5_corrigida.jpg (300 dpi para a camera-ready)
================================================================================
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = Path(__file__).resolve().parent

TRAD = {
 "I - Administração e Economia aplicadas à Engenharia Civil;":
     "Administration and Economics applied to Eng.",
 "II - Estado, sociedade e trabalho": "State, society, and work",
 "II - Informática, algoritmos e programação;":
     "Computing, algorithms, and programming",
 "IV - Ciência dos materiais;": "Materials science",
 "VI - Eletricidade aplicada à Engenharia Civil;":
     "Applied electricity to Civil Engineering",
 "VII - Expressão gráfica e desenho universal;":
     "Graphic expression and universal design",
 "X - Topografia e geoprocessamento;": "Topography and geoprocessing",
 "XI - Construção civil;": "Civil construction",
 "XIII - Geotecnia;": "Geotechnics",
 "XIV - Recursos hídricos e saneamento;": "Water resources and sanitation",
 "XV - Transportes;": "Transportation",
 "I - Clínica Médica;": "Clinical Medicine",
 "V - Medicina de Família e Comunidade;": "Family and Community Medicine",
 "IX - História da enfermagem e legislação":
     "History of nursing and legislation",
 "XII - Desigualdades estruturais econômicas, étnico-raciais e de gênero":
     "Structural economic and ethnic inequalities",
 "XVIII - Cuidados de enfermagem em situações de urgência e emergência":
     "Nursing care in urgent and emergency situations",
}

AZUL, VERM = "#3d6b9e", "#c0504d"
PAINEIS = [("Engenharia Civil", "Civil Engineering"),
           ("Medicina", "Medicine"), ("Enfermagem", "Nursing")]


def main():
    d = pd.read_csv(PASTA / "diagnostico_oc_arquetipos.csv", sep=";")
    piv = d.pivot_table(index=["curso", "oc"], columns="arquetipo",
                        values="prob_acerto_media")
    piv.columns = ["a", "b"]
    cron = piv[(piv < 0.50).all(axis=1)].reset_index()
    cron["label"] = cron["oc"].str.strip().map(lambda s: TRAD.get(s, s))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.0))
    for ax, (curso, titulo) in zip(axes, PAINEIS):
        g = (cron[cron["curso"] == curso]
             .sort_values("a", ascending=True).reset_index(drop=True))
        y = range(len(g)); h = 0.38
        ax.barh([i + h/2 for i in y], g["a"], height=h, color=AZUL,
                label="Arch. α")
        ax.barh([i - h/2 for i in y], g["b"], height=h, color=VERM,
                label="Arch. β")
        ax.set_yticks(list(y)); ax.set_yticklabels(g["label"], fontsize=8)
        ax.axvline(0.40, color="#b22222", ls="--", lw=1)
        ax.axvline(0.50, color="#e6a23c", ls="--", lw=1)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Prob. of success", fontsize=9)
        ax.set_title(f"{titulo}\n({len(g)} deficient KOs)",
                     fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Chronically deficient Knowledge Objects "
                 "(average prob. < 0.50 for both archetypes)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PASTA / "figura5_corrigida.jpg", dpi=300, bbox_inches="tight")
    print("contagens:", cron.groupby("curso").size().to_dict())


if __name__ == "__main__":
    main()
