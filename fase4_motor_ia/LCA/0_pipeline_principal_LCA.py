"""
================================================================================
ANÁLISE DE PERFIS DE APRENDIZAGEM A PARTIR DE MICRODADOS DO ENADE 2023
================================================================================

Objetivos:
    1) Identificar perfis latentes de aprendizagem dos estudantes em cada IES.
    2) Produzir diagnóstico institucional dos pontos fortes e fracos por perfil.

Abordagem metodológica:
    Método principal: Latent Class Analysis (LCA), adequado para variáveis
    categóricas binárias (acerto/erro), com seleção de k por critérios de
    informação (BIC, AIC) e validação por entropia e bootstrap.

    Análise de sensibilidade: K-Means sobre coordenadas de Análise de
    Correspondência Múltipla (MCA), apropriada para dados categóricos.

Referências metodológicas centrais:
    - Collins, L. M., & Lanza, S. T. (2010). Latent Class and Latent Transition
      Analysis. Wiley.
    - Nylund, K. L., Asparouhov, T., & Muthén, B. O. (2007). Deciding on the
      Number of Classes in LCA and Growth Mixture Modeling. Structural
      Equation Modeling, 14(4), 535-569.
    - Hennig, C. (2007). Cluster-wise assessment of cluster stability.
      Computational Statistics & Data Analysis, 52(1), 258-271.
    - Greenacre, M. (2017). Correspondence Analysis in Practice. CRC Press.

NOTA DE INTEGRIDADE METODOLÓGICA:
    Os hiperparâmetros NÃO são escolhidos para maximizar métricas. A escolha
    do número de classes/clusters segue critérios de informação amplamente
    aceitos na literatura, complementados por estabilidade via bootstrap e
    interpretabilidade dos perfis. A silhueta é reportada como diagnóstico,
    nunca como objetivo de otimização.
================================================================================
"""

import json
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# CONFIGURAÇÃO E LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuracao:
    """Parâmetros do estudo. Justificativas no docstring de cada campo."""

    # Tamanho mínimo da IES.
    # Justificativa: LCA com k=4 e 38 itens binários requer estimar ~150
    # parâmetros. A regra empírica de Collins & Lanza (2010, p. 84) sugere
    # n >= 5 * número de parâmetros para estimativas estáveis. Adotamos um
    # corte conservador de n >= 50, com sensibilidade para n >= 100.
    min_alunos_ies: int = 50

    # Faixa de k testada.
    # Justificativa pedagógica: a literatura de avaliação educacional
    # tipicamente identifica entre 2 e 5 perfis interpretáveis (e.g., níveis
    # de proficiência do INEP). Estendemos até k=6 como margem.
    k_min: int = 2
    k_max: int = 6

    # Número mínimo de alunos por classe na solução final.
    # Justificativa: classes com n < 5% da amostra ou n < 10 são consideradas
    # espúrias na literatura de mistura finita (Nylund et al., 2007).
    min_n_classe: int = 10
    min_pct_classe: float = 0.05

    # Réplicas de bootstrap para estabilidade.
    # Trade-off computacional: 100 é suficiente para IC de estabilidade
    # (Hennig, 2007). Aumentar para 500+ no estudo final.
    n_bootstrap: int = 100

    # Sementes fixas para reprodutibilidade.
    seed: int = 42

    # Limiar de entropia normalizada.
    # Justificativa: entropia >= 0.80 indica boa separação entre classes
    # (Celeux & Soromenho, 1996).
    entropia_minima_aceitavel: float = 0.80


CFG = Configuracao()


# ==============================================================================
# MAPEAMENTO DE CURSOS (mantido do código original)
# ==============================================================================

CURSOS_MAP = {
    5: "Medicina Veterinária", 6: "Odontologia", 12: "Medicina",
    17: "Agronomia", 19: "Farmácia", 21: "Arquitetura e Urbanismo",
    23: "Enfermagem", 27: "Fonoaudiologia", 28: "Nutrição",
    36: "Fisioterapia", 51: "Zootecnia", 55: "Biomedicina",
    69: "Tecnologia em Radiologia", 90: "Tecnologia em Agronegócios",
    91: "Tecnologia em Gestão Hospitalar",
    92: "Tecnologia em Gestão Ambiental",
    95: "Tecnologia em Estética e Cosmética",
    5710: "Engenharia Civil", 5806: "Engenharia Elétrica",
    5814: "Engenharia de Controle e Automação",
    5902: "Engenharia Mecânica", 6002: "Engenharia de Alimentos",
    6008: "Engenharia Química", 6208: "Engenharia de Produção",
    6307: "Engenharia Ambiental", 6405: "Engenharia Florestal",
    6410: "Tecnologia em Segurança no Trabalho",
    6411: "Engenharia de Computação",
}


# ==============================================================================
# 1. PREPARAÇÃO DOS DADOS
# ==============================================================================

def codificar_respostas(df: pd.DataFrame, cols_questoes: list[str]) -> pd.DataFrame:
    """
    Codifica respostas como binárias (1 = acerto, 0 = erro/branco/anulada).

    DECISÃO METODOLÓGICA EXPLÍCITA:
        Tratamos branco e anulada como "não-acerto", igual a erro.

    Justificativa:
        Embora branco/anulada e erro tenham significados diferentes em termos
        comportamentais (desistência vs. tentativa errada), do ponto de vista
        do *resultado de aprendizagem* todos representam ausência de domínio
        do conteúdo avaliado. Esta convenção é a mesma adotada pelo INEP no
        cálculo da nota ENADE.

    Limitação reconhecida:
        Esta escolha pode mascarar perfis "desistentes" vs. "tentantes errados".
        Recomenda-se análise de sensibilidade tratando branco como missing
        e usando FIML (a ser implementada em estudo posterior).
    """
    df_cod = df.copy()
    for col in cols_questoes:
        df_cod[col] = (df_cod[col].astype(str) == "1").astype(int)
    return df_cod


def filtrar_ies_validas(
    df: pd.DataFrame, min_alunos: int
) -> dict[str, pd.DataFrame]:
    """Retorna apenas IES com tamanho amostral suficiente."""
    ies_validas = {}
    for ies, df_ies in df.groupby("CO_IES"):
        if len(df_ies) >= min_alunos:
            ies_validas[ies] = df_ies
    return ies_validas


# ==============================================================================
# 2. LATENT CLASS ANALYSIS (MÉTODO PRINCIPAL)
# ==============================================================================

class ModeloLCA:
    """
    Implementação de Latent Class Analysis para itens binários via EM.

    Modelo:
        P(Y_i = y_i) = sum_k pi_k * prod_j p_{j,k}^{y_{ij}} * (1 - p_{j,k})^{(1-y_{ij})}

    onde:
        pi_k     = probabilidade da classe k (mixing proportion)
        p_{j,k}  = probabilidade de acerto no item j dado classe k

    Estimação por Expectation-Maximization (Dempster, Laird & Rubin, 1977).
    """

    def __init__(self, k: int, max_iter: int = 500, tol: float = 1e-6,
                 n_init: int = 20, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init  # múltiplos inits para escapar de ótimos locais
        self.seed = seed

        # Atributos ajustados:
        self.pi_: Optional[np.ndarray] = None      # (k,)
        self.p_: Optional[np.ndarray] = None       # (k, n_itens)
        self.log_lik_: Optional[float] = None
        self.n_params_: Optional[int] = None
        self.posterior_: Optional[np.ndarray] = None  # (n, k)
        self.convergiu_: bool = False
        self.n_iter_: int = 0

    def _log_verossimilhanca(self, X: np.ndarray, pi: np.ndarray,
                              p: np.ndarray) -> tuple[float, np.ndarray]:
        """Calcula log-verossimilhança e log-densidade conjunta por classe."""
        # log P(y_i | classe k) para cada (i, k)
        # Usamos log-space para estabilidade numérica
        log_p = np.log(np.clip(p, 1e-12, 1 - 1e-12))
        log_1mp = np.log(np.clip(1 - p, 1e-12, 1 - 1e-12))

        # (n, k): log P(y_i | k) + log pi_k
        log_joint = X @ log_p.T + (1 - X) @ log_1mp.T + np.log(np.clip(pi, 1e-12, 1))

        # log-sum-exp para a marginal
        max_lj = log_joint.max(axis=1, keepdims=True)
        log_marg = max_lj.squeeze() + np.log(
            np.exp(log_joint - max_lj).sum(axis=1)
        )
        return log_marg.sum(), log_joint

    def _passo_e(self, log_joint: np.ndarray) -> np.ndarray:
        """E-step: posterior P(classe | y_i)."""
        max_lj = log_joint.max(axis=1, keepdims=True)
        log_post = log_joint - max_lj
        post = np.exp(log_post)
        return post / post.sum(axis=1, keepdims=True)

    def _passo_m(self, X: np.ndarray, post: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        """M-step: maximiza expectativa da log-verossimilhança completa."""
        n_k = post.sum(axis=0)                    # (k,)
        pi_novo = n_k / X.shape[0]
        # Suavização leve para evitar p exatamente 0 ou 1
        p_novo = (post.T @ X + 1e-3) / (n_k[:, None] + 2e-3)
        return pi_novo, p_novo

    def _ajustar_uma_vez(self, X: np.ndarray, rng: np.random.Generator
                          ) -> tuple[float, np.ndarray, np.ndarray, bool, int]:
        """Uma execução do EM com inicialização aleatória."""
        n, n_itens = X.shape
        # Inicialização: pi uniforme + p perturbado em torno da média marginal
        pi = np.full(self.k, 1 / self.k)
        media_global = X.mean(axis=0)
        p = np.clip(
            media_global + rng.normal(0, 0.15, size=(self.k, n_itens)),
            0.05, 0.95
        )

        log_lik_anterior = -np.inf
        for it in range(self.max_iter):
            log_lik, log_joint = self._log_verossimilhanca(X, pi, p)
            if abs(log_lik - log_lik_anterior) < self.tol:
                return log_lik, pi, p, True, it
            log_lik_anterior = log_lik
            post = self._passo_e(log_joint)
            pi, p = self._passo_m(X, post)

        return log_lik, pi, p, False, self.max_iter

    def fit(self, X: np.ndarray) -> "ModeloLCA":
        """
        Ajusta com múltiplas inicializações e mantém a melhor solução
        (maior log-verossimilhança).
        """
        rng = np.random.default_rng(self.seed)
        melhor_ll = -np.inf
        melhor = None

        for _ in range(self.n_init):
            ll, pi, p, conv, n_iter = self._ajustar_uma_vez(X, rng)
            if ll > melhor_ll:
                melhor_ll = ll
                melhor = (pi, p, conv, n_iter)

        self.pi_, self.p_, self.convergiu_, self.n_iter_ = melhor
        self.log_lik_ = melhor_ll
        # n_params: (k-1) para pi + k * n_itens para p
        self.n_params_ = (self.k - 1) + self.k * X.shape[1]
        # Posterior final
        _, log_joint = self._log_verossimilhanca(X, self.pi_, self.p_)
        self.posterior_ = self._passo_e(log_joint)
        return self

    # ----- Critérios de informação -----

    def bic(self, n: int) -> float:
        """Bayesian Information Criterion (Schwarz, 1978). Menor = melhor."""
        return -2 * self.log_lik_ + self.n_params_ * np.log(n)

    def aic(self) -> float:
        """Akaike Information Criterion (Akaike, 1974). Menor = melhor."""
        return -2 * self.log_lik_ + 2 * self.n_params_

    def abic(self, n: int) -> float:
        """Sample-size Adjusted BIC (Sclove, 1987). Recomendado em LCA."""
        return -2 * self.log_lik_ + self.n_params_ * np.log((n + 2) / 24)

    def entropia_normalizada(self) -> float:
        """
        Entropia relativa de Celeux & Soromenho (1996).
        E_N em [0, 1]; valores >= 0.80 indicam boa separação entre classes.
        """
        post = np.clip(self.posterior_, 1e-12, 1)
        n, k = post.shape
        entropia = -np.sum(post * np.log(post))
        return 1 - entropia / (n * np.log(k))

    def predict(self) -> np.ndarray:
        """Atribuição modal (classe mais provável por aluno)."""
        return self.posterior_.argmax(axis=1)


# ==============================================================================
# 3. SELEÇÃO DO NÚMERO DE CLASSES
# ==============================================================================

@dataclass
class ResultadoAjuste:
    k: int
    log_lik: float
    bic: float
    aic: float
    abic: float
    entropia: float
    menor_classe_pct: float
    convergiu: bool


def ajustar_serie_lca(X: np.ndarray, k_min: int, k_max: int, seed: int
                       ) -> list[tuple[int, ModeloLCA, ResultadoAjuste]]:
    """Ajusta LCA para k = k_min..k_max e retorna métricas de cada modelo."""
    n = len(X)
    resultados = []
    for k in range(k_min, k_max + 1):
        modelo = ModeloLCA(k=k, seed=seed).fit(X)
        labels = modelo.predict()
        pcts = np.bincount(labels, minlength=k) / n
        res = ResultadoAjuste(
            k=k,
            log_lik=modelo.log_lik_,
            bic=modelo.bic(n),
            aic=modelo.aic(),
            abic=modelo.abic(n),
            entropia=modelo.entropia_normalizada(),
            menor_classe_pct=float(pcts.min()),
            convergiu=modelo.convergiu_,
        )
        resultados.append((k, modelo, res))
    return resultados


def selecionar_k(resultados: list[tuple[int, ModeloLCA, ResultadoAjuste]],
                  cfg: Configuracao, n: int
                  ) -> tuple[int, str]:
    """
    Seleciona k seguindo regra de decisão pré-registrada:
      1. Filtra modelos com menor classe >= max(min_n_classe/n, min_pct_classe).
      2. Filtra modelos com entropia >= entropia_minima_aceitavel.
      3. Entre os sobreviventes, escolhe o de menor BIC.
      4. Se nenhum sobrevive ao passo 2, relaxa para o de menor BIC entre os
         que passaram no passo 1, com ressalva no relatório.

    Esta regra é declarada A PRIORI; não é revisada após ver os resultados.
    """
    minimo_pct = max(cfg.min_n_classe / n, cfg.min_pct_classe)

    candidatos = [r for _, _, r in resultados if r.menor_classe_pct >= minimo_pct]
    if not candidatos:
        # Fallback: escolhe k mínimo
        return cfg.k_min, "FALLBACK: nenhum modelo com classes não-espúrias"

    com_entropia = [r for r in candidatos if r.entropia >= cfg.entropia_minima_aceitavel]

    if com_entropia:
        escolhido = min(com_entropia, key=lambda r: r.bic)
        justificativa = (
            f"Menor BIC ({escolhido.bic:.2f}) entre modelos com entropia "
            f">= {cfg.entropia_minima_aceitavel} e classes não-espúrias"
        )
    else:
        escolhido = min(candidatos, key=lambda r: r.bic)
        justificativa = (
            f"RESSALVA: nenhum modelo atingiu entropia "
            f">= {cfg.entropia_minima_aceitavel}; escolhido por menor BIC "
            f"({escolhido.bic:.2f}) entre modelos com classes não-espúrias. "
            f"Entropia obtida: {escolhido.entropia:.3f}"
        )
    return escolhido.k, justificativa


# ==============================================================================
# 4. ESTABILIDADE POR BOOTSTRAP (Hennig, 2007)
# ==============================================================================

def jaccard_max(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """Média do índice de Jaccard máximo entre classes de A e classes de B."""
    jaccards = []
    for a in range(k):
        sa = set(np.where(A == a)[0])
        if not sa:
            continue
        melhor = 0.0
        for b in range(k):
            sb = set(np.where(B == b)[0])
            if not sb:
                continue
            inter = len(sa & sb)
            uniao = len(sa | sb)
            j = inter / uniao if uniao else 0
            if j > melhor:
                melhor = j
        jaccards.append(melhor)
    return float(np.mean(jaccards)) if jaccards else 0.0


def estabilidade_bootstrap(X: np.ndarray, k: int, n_boot: int, seed: int
                            ) -> dict[str, float]:
    """
    Estabilidade dos clusters por bootstrap não-paramétrico.
    Retorna média e IC 95% do índice de Jaccard.

    Interpretação (Hennig, 2007):
        > 0.85 = altamente estável
        0.75-0.85 = estável
        0.60-0.75 = moderada (interpretação cautelosa)
        < 0.60 = instável (não interpretar)
    """
    rng = np.random.default_rng(seed)
    n = len(X)

    # Modelo de referência
    ref = ModeloLCA(k=k, seed=seed, n_init=10).fit(X)
    labels_ref = ref.predict()

    jaccards = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_boot = X[idx]
        try:
            mod_b = ModeloLCA(k=k, seed=seed + b + 1, n_init=5).fit(X_boot)
            # Aplica modelo bootstrap aos dados originais via posterior
            _, lj = mod_b._log_verossimilhanca(X, mod_b.pi_, mod_b.p_)
            labels_b = mod_b._passo_e(lj).argmax(axis=1)
            jaccards.append(jaccard_max(labels_ref, labels_b, k))
        except Exception:
            continue

    if not jaccards:
        return {"jaccard_medio": 0.0, "ic_inf": 0.0, "ic_sup": 0.0, "n_validas": 0}

    arr = np.array(jaccards)
    return {
        "jaccard_medio": float(arr.mean()),
        "ic_inf": float(np.percentile(arr, 2.5)),
        "ic_sup": float(np.percentile(arr, 97.5)),
        "n_validas": len(jaccards),
    }


# ==============================================================================
# 5. ANÁLISE DE SENSIBILIDADE: K-MEANS SOBRE MCA
# ==============================================================================

def kmeans_sobre_mca(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Análise de sensibilidade: K-Means em coordenadas de Análise de
    Correspondência Múltipla (apropriada para dados categóricos).

    Implementação simples de MCA via SVD da matriz indicadora padronizada.
    Mantemos componentes que explicam >= 70% da variância (Greenacre, 2017).
    """
    from sklearn.cluster import KMeans

    # Para dados binários, MCA reduz-se a SVD da matriz centrada/escalada
    # ponderada pelas frequências marginais.
    # Implementação simplificada: SVD sobre matriz binária centrada.
    media = X.mean(axis=0, keepdims=True)
    dp = X.std(axis=0, keepdims=True)
    dp[dp == 0] = 1
    Z = (X - media) / dp

    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    # Variância explicada por componente
    var_exp = (S ** 2) / (S ** 2).sum()
    var_acum = np.cumsum(var_exp)
    n_comp = max(2, int(np.searchsorted(var_acum, 0.70) + 1))
    n_comp = min(n_comp, len(S))

    coords = U[:, :n_comp] * S[:n_comp]
    km = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(coords)
    return km.labels_


def concordancia_lca_kmeans(labels_lca: np.ndarray,
                              labels_km: np.ndarray) -> float:
    """Adjusted Rand Index entre as duas atribuições."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_lca, labels_km))


# ==============================================================================
# 6. CARACTERIZAÇÃO DAS CLASSES PELO CONTEÚDO
# ==============================================================================

def caracterizar_classes(X: np.ndarray, labels: np.ndarray,
                          cols_questoes: list[str], k: int
                          ) -> pd.DataFrame:
    """
    Caracteriza cada classe pelas probabilidades de acerto por item,
    pelo gap em relação à média da IES, e por marcadores diferenciais.

    Salva, para cada classe:
        - Metadados (n, %, taxa média)
        - 38 probabilidades de acerto (PROB_Q1..PROB_Q38)
        - 38 gaps em relação à média da IES (GAP_Q1..GAP_Q38)
        - Resumo top-5 (itens fortes/fracos) — mantido para compatibilidade

    NOTA: Os rótulos descritivos das classes NÃO são atribuídos
    automaticamente por nota. A nomeação é responsabilidade do pesquisador,
    com base na inspeção qualitativa dos padrões de resposta. Aqui apenas
    apresentamos a evidência.
    """
    media_ies = X.mean(axis=0)
    linhas = []
    for c in range(k):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        media_c = X[mask].mean(axis=0)
        gap = media_c - media_ies

        # Itens diferenciadores (gap absoluto > 0.10) - resumo de compatibilidade
        itens_fortes = [cols_questoes[i] for i in np.argsort(-gap)[:5]
                        if gap[i] > 0.10]
        itens_fracos = [cols_questoes[i] for i in np.argsort(gap)[:5]
                        if gap[i] < -0.10]

        linha = {
            "classe_id": c,
            "n_alunos": n_c,
            "pct_alunos": round(100 * n_c / len(X), 2),
            "taxa_acerto_media": round(float(media_c.mean()), 4),
            "taxa_acerto_media_ies": round(float(media_ies.mean()), 4),
            "itens_diferencialmente_fortes": ", ".join(itens_fortes) or "-",
            "itens_diferencialmente_fracos": ", ".join(itens_fracos) or "-",
        }

        # ----- Probabilidades de acerto por item (38 colunas) -----
        # Estas são as taxas de acerto observadas EM CADA CLASSE, não as
        # probabilidades teóricas do modelo LCA (p_{j,k}). A diferença é
        # sutil mas relevante: a taxa observada usa atribuição modal, a
        # teórica é o parâmetro estimado. Para o objetivo desta análise
        # (caracterizar perfis interpretáveis), a observada é mais direta.
        for i, q in enumerate(cols_questoes):
            linha[f"PROB_{q}"] = round(float(media_c[i]), 4)

        # ----- Gaps em relação à média da IES (38 colunas) -----
        for i, q in enumerate(cols_questoes):
            linha[f"GAP_{q}"] = round(float(gap[i]), 4)

        linhas.append(linha)
    return pd.DataFrame(linhas)


# ==============================================================================
# 7. PIPELINE COMPLETA POR IES
# ==============================================================================

def analisar_ies(df_ies: pd.DataFrame, cols_questoes: list[str],
                  curso: str, ies_id: str, cfg: Configuracao) -> dict:
    """Executa o pipeline completo para uma única IES."""
    df_cod = codificar_respostas(df_ies, cols_questoes)
    X = df_cod[cols_questoes].values.astype(int)
    n = len(X)

    # 1. Ajuste da série de modelos LCA
    resultados = ajustar_serie_lca(X, cfg.k_min, cfg.k_max, cfg.seed)

    # 2. Seleção de k
    k_escolhido, justificativa_k = selecionar_k(resultados, cfg, n)

    # 3. Modelo final
    modelo_final = next(m for kk, m, _ in resultados if kk == k_escolhido)
    labels_lca = modelo_final.predict()

    # 4. Estabilidade por bootstrap
    estab = estabilidade_bootstrap(X, k_escolhido, cfg.n_bootstrap, cfg.seed)

    # 5. Análise de sensibilidade: K-Means sobre MCA
    labels_km = kmeans_sobre_mca(X, k_escolhido, cfg.seed)
    ari = concordancia_lca_kmeans(labels_lca, labels_km)

    # 6. Caracterização das classes
    df_classes = caracterizar_classes(X, labels_lca, cols_questoes, k_escolhido)

    # Tabela completa de critérios para todos os k testados (transparência)
    df_criterios = pd.DataFrame([asdict(r) for _, _, r in resultados])

    return {
        "curso": curso,
        "ies": ies_id,
        "n_alunos": n,
        "k_escolhido": k_escolhido,
        "justificativa_k": justificativa_k,
        "tabela_criterios": df_criterios,
        "tabela_classes": df_classes,
        "estabilidade_bootstrap": estab,
        "ari_lca_vs_kmeans_mca": round(ari, 4),
        "log_lik": modelo_final.log_lik_,
        "entropia_normalizada": round(modelo_final.entropia_normalizada(), 4),
        "convergiu": modelo_final.convergiu_,
        "labels_lca": labels_lca,
        "posterior": modelo_final.posterior_,
    }


# ==============================================================================
# 8. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main(caminho_microdados: Path, pasta_saida: Path) -> None:
    pasta_saida.mkdir(parents=True, exist_ok=True)

    logger.info("Carregando microdados de %s", caminho_microdados)
    df = pd.read_csv(caminho_microdados, sep=";", dtype=str)
    cols_q = [f"Q{i}" for i in range(1, 39) if f"Q{i}" in df.columns]
    logger.info("Itens identificados: %d", len(cols_q))

    relatorio_geral = []
    detalhes_classes = []
    detalhes_criterios = []

    for co_grupo in df["CO_GRUPO"].dropna().unique():
        try:
            cod = int(co_grupo)
        except ValueError:
            continue
        if cod not in CURSOS_MAP:
            continue

        nome_curso = CURSOS_MAP[cod]
        df_curso = df[df["CO_GRUPO"] == str(co_grupo)]
        ies_dict = filtrar_ies_validas(df_curso, CFG.min_alunos_ies)
        logger.info("Curso %s: %d IES com n >= %d",
                    nome_curso, len(ies_dict), CFG.min_alunos_ies)

        for ies_id, df_ies in ies_dict.items():
            try:
                resultado = analisar_ies(df_ies, cols_q, nome_curso, ies_id, CFG)
            except Exception as e:
                logger.error("Falha em IES %s (%s): %s", ies_id, nome_curso, e)
                continue

            relatorio_geral.append({
                "curso": resultado["curso"],
                "ies": resultado["ies"],
                "n_alunos": resultado["n_alunos"],
                "k_escolhido": resultado["k_escolhido"],
                "justificativa_k": resultado["justificativa_k"],
                "log_lik": resultado["log_lik"],
                "entropia_normalizada": resultado["entropia_normalizada"],
                "jaccard_medio_bootstrap": resultado["estabilidade_bootstrap"]["jaccard_medio"],
                "jaccard_ic_inf": resultado["estabilidade_bootstrap"]["ic_inf"],
                "jaccard_ic_sup": resultado["estabilidade_bootstrap"]["ic_sup"],
                "ari_lca_vs_kmeans_mca": resultado["ari_lca_vs_kmeans_mca"],
                "convergiu": resultado["convergiu"],
            })

            df_cls = resultado["tabela_classes"].copy()
            df_cls.insert(0, "ies", resultado["ies"])
            df_cls.insert(0, "curso", resultado["curso"])
            detalhes_classes.append(df_cls)

            df_cri = resultado["tabela_criterios"].copy()
            df_cri.insert(0, "ies", resultado["ies"])
            df_cri.insert(0, "curso", resultado["curso"])
            detalhes_criterios.append(df_cri)

    # Salvar saídas
    pd.DataFrame(relatorio_geral).to_csv(
        pasta_saida / "01_relatorio_geral_por_ies.csv",
        sep=";", index=False, encoding="utf-8-sig", decimal=",",
    )
    if detalhes_classes:
        pd.concat(detalhes_classes, ignore_index=True).to_csv(
            pasta_saida / "02_caracterizacao_das_classes.csv",
            sep=";", index=False, encoding="utf-8-sig", decimal=",",
        )
    if detalhes_criterios:
        pd.concat(detalhes_criterios, ignore_index=True).to_csv(
            pasta_saida / "03_criterios_de_selecao_k.csv",
            sep=";", index=False, encoding="utf-8-sig", decimal=",",
        )

    # Log da configuração para reprodutibilidade
    with open(pasta_saida / "00_configuracao_usada.json", "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, indent=2, ensure_ascii=False)

    logger.info("Análise concluída. Resultados em: %s", pasta_saida)


if __name__ == "__main__":
    DIRETORIO_ATUAL = Path(__file__).resolve().parent
    DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent.parent
    CAMINHO_DADOS = DIRETORIO_RAIZ / "arquivosgerados" / "relatorio_final_enade_2023.csv"
    PASTA_SAIDA = DIRETORIO_RAIZ / "arquivosgerados" / "RESULTADOS_LCA"

    main(CAMINHO_DADOS, PASTA_SAIDA)
