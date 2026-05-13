"""
================================================================================
SEÇÃO 5 - DIAGNÓSTICO POR OBJETOS DE CONHECIMENTO
================================================================================

Pré-requisitos:
    1. Pipeline principal `analise_perfis_enade.py` rodado com sucesso.
       (gera 02_caracterizacao_das_classes.csv com PROB_Q1..Q38 e GAP_Q1..Q38)
    2. Pasta com os arquivos Excel do Anexo IX do INEP por curso
       (ex.: engenharia_civil_anexoIX.xlsx, medicina_anexoIX.xlsx, etc.)

O que este script faz:
    1. Lê os Anexos IX e constrói uma tabela mestre Q × OC para cada curso.
    2. Para cada curso (entre os de foco), agrupa as classes em 2 arquétipos
       nacionais via clustering hierárquico (Ward) sobre gaps padronizados.
    3. Para cada arquétipo, calcula a probabilidade média de acerto em cada
       item Q1-Q38 e agrega por OC (média das probabilidades das questões
       que avaliam aquele OC).
    4. Classifica cada (curso, arquétipo, OC) em quatro zonas pedagógicas:
       crítico (<0,40), deficitário (<0,50), intermediário (<0,65), domínio.
    5. Identifica OCs cronicamente deficitários (prob < 0,50 em ambos os
       arquétipos) — lacunas estruturais nacionais.
    6. Identifica OCs marcadores de perfil (diferença ≥ 0,20 entre os dois
       arquétipos) — conteúdos que distinguem fortemente os perfis.
    7. Gera as figuras 6 e 7 do artigo (figOC_diagnostico, figOC_cronicos).
================================================================================
"""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

PASTA_ANEXOS_INEP = Path("/caminho/para/anexos_INEP")  # AJUSTAR
PASTA_DADOS_PIPELINE = Path("/caminho/para/saida/pipeline")  # AJUSTAR
ARQUIVO_CLASSES = PASTA_DADOS_PIPELINE / "02_caracterizacao_das_classes.csv"

PASTA_SAIDA = Path("./resultados_secao_5")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Cursos analisados em profundidade no artigo
CURSOS_FOCO = ["Engenharia Civil", "Medicina", "Enfermagem"]


# ==============================================================================
# 1. CONSTRUÇÃO DA TABELA MESTRE Q × OC
# ==============================================================================

# Mapeamento de nome de arquivo INEP para nome de curso nos microdados
MAPA_CURSO = {
    'agronomia': 'Agronomia',
    'arquitetura_e_urbanismo': 'Arquitetura e Urbanismo',
    'biomedicina': 'Biomedicina',
    'enfermagem': 'Enfermagem',
    'engenharia_ambiental': 'Engenharia Ambiental',
    'engenharia_civil': 'Engenharia Civil',
    'engenharia_de_alimentos': 'Engenharia de Alimentos',
    'engenharia_de_computacao': 'Engenharia de Computação',
    'engenharia_de_controle_e_automacao': 'Engenharia de Controle e Automação',
    'engenharia_de_producao': 'Engenharia de Produção',
    'engenharia_eletrica': 'Engenharia Elétrica',
    'engenharia_florestal': 'Engenharia Florestal',
    'engenharia_mecanica': 'Engenharia Mecânica',
    'engenharia_quimica': 'Engenharia Química',
    'farmacia': 'Farmácia',
    'fisioterapia': 'Fisioterapia',
    'fonoaudiologia': 'Fonoaudiologia',
    'medicina': 'Medicina',
    'medicina_veterinaria': 'Medicina Veterinária',
    'nutricao': 'Nutrição',
    'odontologia': 'Odontologia',
    'tecnologia_em_agronegocios': 'Tecnologia em Agronegócios',
    'tecnologia_em_estetica_e_cosmetica': 'Tecnologia em Estética e Cosmética',
    'tecnologia_em_gestao_ambiental': 'Tecnologia em Gestão Ambiental',
    'tecnologia_em_gestao_hospitalar': 'Tecnologia em Gestão Hospitalar',
    'tecnologia_em_radiologia': 'Tecnologia em Radiologia',
    'tecnologia_em_seguranca_no_trabalho': 'Tecnologia em Segurança no Trabalho',
    'zootecnia': 'Zootecnia',
}


def limpar_texto(s):
    if pd.isna(s):
        return None
    s = str(s).replace('\n', ' ')
    return re.sub(r'\s+', ' ', s).strip()


def construir_tabela_mestre():
    """
    Lê os Anexos IX do INEP e constrói tabela única Q × OC.
    
    Estrutura dos Anexos IX:
        - Coluna POSIÇÃO: D1 (discursiva), 1-9 (FG objetivas), D2, 10-38 (CE)
        - Posições 1-9: idênticas em todos os cursos (Formação Geral)
        - Posições 10-38: específicas do curso (Componente Específico)
        - OC1, OC2: objetos de conhecimento avaliados pelo item
    """
    linhas = []
    for arq, curso_nome in MAPA_CURSO.items():
        fpath = PASTA_ANEXOS_INEP / f'{arq}_anexoIX.xlsx'
        if not fpath.exists():
            print(f"AVISO: arquivo não encontrado: {fpath.name}")
            continue
        df = pd.read_excel(fpath, sheet_name=0)
        objetivas = df[df['POSIÇÃO'].apply(lambda x: isinstance(x, int))]
        for _, r in objetivas.iterrows():
            pos = int(r['POSIÇÃO'])
            linhas.append({
                'curso': curso_nome,
                'questao': f'Q{pos}',
                'posicao': pos,
                'area_prova': 'FG' if pos <= 9 else 'CE',
                'perfil_egresso': limpar_texto(r['PERFIL']),
                'competencia': limpar_texto(r['COMPETÊNCIAS']),
                'oc_principal': limpar_texto(r['OC1']),
                'oc_secundario': limpar_texto(r['OC2']),
            })
    return pd.DataFrame(linhas)


# ==============================================================================
# 2. DIAGNÓSTICO DE OCs POR ARQUÉTIPO NACIONAL
# ==============================================================================

def identificar_arquetipos_nacionais(df_classes_curso):
    """
    Aplica clustering hierárquico de Ward sobre as classes do curso, no
    espaço dos gaps padronizados das questões Q10-Q38 (componente específico).
    Retorna labels 1 ou 2 para cada classe.
    """
    cols_gap = [f'GAP_Q{i}' for i in range(10, 39)]
    X = df_classes_curso[cols_gap].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    D = pdist(X_norm, metric='euclidean')
    Z = linkage(D, method='ward')
    return fcluster(Z, t=2, criterion='maxclust')


def diagnosticar_ocs(curso, df_classes, mestre):
    """
    Para um curso, identifica os arquétipos e produz o diagnóstico de OCs.
    """
    sub = df_classes[df_classes['curso'] == curso].copy().reset_index(drop=True)
    sub['arq'] = identificar_arquetipos_nacionais(sub)
    
    mestre_curso = mestre[mestre['curso'] == curso].set_index('questao')
    prob_cols = [f'PROB_Q{i}' for i in range(1, 39)]
    
    resultados = []
    for arq in [1, 2]:
        sub_arq = sub[sub['arq'] == arq]
        prob_media = sub_arq[prob_cols].mean()
        
        # Agregar por OC: média das probabilidades das questões que avaliam o OC
        oc_prob, oc_questoes = {}, {}
        for q in [f'Q{i}' for i in range(1, 39)]:
            if q not in mestre_curso.index:
                continue
            oc = mestre_curso.loc[q, 'oc_principal']
            if pd.isna(oc):
                continue
            oc_prob.setdefault(oc, []).append(prob_media[f'PROB_{q}'])
            oc_questoes.setdefault(oc, []).append(q)
        
        for oc, probs in oc_prob.items():
            prob_oc = float(np.mean(probs))
            resultados.append({
                'curso': curso,
                'arquetipo': arq,
                'oc': oc,
                'n_questoes': len(probs),
                'questoes': ', '.join(oc_questoes[oc]),
                'prob_acerto_media': round(prob_oc, 3),
                'classificacao': classificar_zona(prob_oc),
                'n_classes_arq': len(sub_arq),
                'n_ies_arq': sub_arq['ies'].nunique(),
            })
    return pd.DataFrame(resultados)


def classificar_zona(prob):
    if prob < 0.40:
        return 'CRÍTICO'
    if prob < 0.50:
        return 'DEFICITÁRIO'
    if prob < 0.65:
        return 'INTERMEDIÁRIO'
    return 'DOMÍNIO'


# ==============================================================================
# 3. ANÁLISES DERIVADAS (LACUNAS NACIONAIS, MARCADORES DE PERFIL)
# ==============================================================================

def identificar_lacunas_e_marcadores(df_diag):
    """
    Para cada curso, identifica:
        - OCs cronicamente deficitários: prob < 0,50 em ambos arquétipos
        - OCs marcadores de perfil: diferença ≥ 0,20 entre arquétipos
    """
    resumos = []
    for curso in df_diag['curso'].unique():
        sub = df_diag[df_diag['curso'] == curso]
        pv = sub.pivot_table(
            index='oc', columns='arquetipo', values='prob_acerto_media'
        ).reset_index()
        pv.columns = ['oc', 'arq1', 'arq2']
        pv['media'] = (pv['arq1'] + pv['arq2']) / 2
        pv['gap_abs'] = (pv['arq1'] - pv['arq2']).abs()
        pv['curso'] = curso
        pv['cronicamente_deficitario'] = (pv['arq1'] < 0.50) & (pv['arq2'] < 0.50)
        pv['marcador_de_perfil'] = pv['gap_abs'] >= 0.20
        resumos.append(pv)
    return pd.concat(resumos, ignore_index=True)


# ==============================================================================
# 4. VISUALIZAÇÕES
# ==============================================================================

def encurtar_oc(s, n=42):
    s = re.sub(r'^[IVXLCD]+\s*-\s*', '', str(s))
    return s.rstrip(';').strip()[:n]


def figura_diagnostico_completo(df_diag, cursos, arq_saida):
    """Figura 6 do artigo: probabilidade por OC × arquétipo."""
    mpl.rcParams.update({'font.family': 'sans-serif', 'savefig.dpi': 300})
    fig, axes = plt.subplots(1, len(cursos), figsize=(13, 7))
    if len(cursos) == 1:
        axes = [axes]
    
    for idx, curso in enumerate(cursos):
        sub = df_diag[df_diag['curso'] == curso]
        pv = sub.pivot_table(index='oc', columns='arquetipo',
                              values='prob_acerto_media')
        pv['oc_curto'] = pv.index.map(encurtar_oc)
        pv['min'] = pv[[1, 2]].min(axis=1)
        pv = pv.sort_values('min')
        
        ax = axes[idx]
        y = np.arange(len(pv))
        ax.barh(y - 0.2, pv[1], height=0.4, color='#2E5984', alpha=0.85,
                edgecolor='black', linewidth=0.4, label='Arq. α')
        ax.barh(y + 0.2, pv[2], height=0.4, color='#C44536', alpha=0.85,
                edgecolor='black', linewidth=0.4, label='Arq. β')
        ax.axvline(0.40, color='#8B0000', linestyle='--', alpha=0.6)
        ax.axvline(0.50, color='#E07B00', linestyle='--', alpha=0.6)
        ax.axvline(0.65, color='#27AE60', linestyle='--', alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(pv['oc_curto'], fontsize=8)
        ax.set_xlabel('Probabilidade de acerto')
        ax.set_xlim(0, 1.05)
        ax.set_title(curso, fontweight='bold')
        if idx == 0:
            ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(arq_saida, dpi=300, bbox_inches='tight')
    plt.close()


def figura_cronicos(df_resumo, cursos, arq_saida):
    """Figura 7 do artigo: OCs cronicamente deficitários."""
    fig, axes = plt.subplots(1, len(cursos), figsize=(14, 6))
    if len(cursos) == 1:
        axes = [axes]
    
    for idx, curso in enumerate(cursos):
        sub = df_resumo[(df_resumo['curso'] == curso) &
                        (df_resumo['cronicamente_deficitario'])].copy()
        sub = sub.sort_values('media').head(12)
        sub['oc_curto'] = sub['oc'].apply(encurtar_oc)
        
        ax = axes[idx]
        y = np.arange(len(sub))
        ax.barh(y - 0.2, sub['arq1'], height=0.4, color='#2E5984',
                alpha=0.85, edgecolor='black', linewidth=0.4, label='Arq. α')
        ax.barh(y + 0.2, sub['arq2'], height=0.4, color='#C44536',
                alpha=0.85, edgecolor='black', linewidth=0.4, label='Arq. β')
        ax.axvline(0.40, color='#8B0000', linestyle='--', alpha=0.6)
        ax.axvline(0.50, color='#E07B00', linestyle='--', alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(sub['oc_curto'], fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Prob. de acerto')
        ax.set_title(f'{curso}\n({len(sub)} OCs deficitários)',
                      fontweight='bold')
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(arq_saida, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    print("Construindo tabela mestre Q × OC a partir dos Anexos IX...")
    mestre = construir_tabela_mestre()
    mestre.to_csv(PASTA_SAIDA / "mestre_questao_oc.csv",
                   sep=';', index=False, encoding='utf-8-sig')
    print(f"Tabela mestre salva: {len(mestre)} linhas, "
          f"{mestre['curso'].nunique()} cursos.")
    
    print("\nCarregando classes do pipeline LCA...")
    df_classes = pd.read_csv(ARQUIVO_CLASSES, sep=';', decimal=',')
    
    print("\nProduzindo diagnóstico por OC × arquétipo...")
    diagnosticos = []
    for curso in CURSOS_FOCO:
        if curso not in df_classes['curso'].unique():
            print(f"  AVISO: '{curso}' não encontrado no df_classes.")
            continue
        diagnosticos.append(diagnosticar_ocs(curso, df_classes, mestre))
    
    df_diag = pd.concat(diagnosticos, ignore_index=True)
    df_diag.to_csv(PASTA_SAIDA / "diagnostico_oc_arquetipos.csv",
                    sep=';', index=False, encoding='utf-8-sig')
    
    print("Identificando lacunas estruturais e marcadores de perfil...")
    df_resumo = identificar_lacunas_e_marcadores(df_diag)
    df_resumo.to_csv(PASTA_SAIDA / "resumo_lacunas_e_marcadores.csv",
                      sep=';', index=False, encoding='utf-8-sig')
    
    print("\nGerando figuras...")
    figura_diagnostico_completo(df_diag, CURSOS_FOCO,
                                 PASTA_SAIDA / "fig_diagnostico_oc.png")
    figura_cronicos(df_resumo, CURSOS_FOCO,
                     PASTA_SAIDA / "fig_ocs_cronicos.png")
    
    # Resumo numérico
    print("\nResumo por curso:")
    for curso in CURSOS_FOCO:
        sub = df_resumo[df_resumo['curso'] == curso]
        n_total = len(sub)
        n_cron = sub['cronicamente_deficitario'].sum()
        n_marc = sub['marcador_de_perfil'].sum()
        prob_med = sub['media'].mean()
        print(f"  {curso}: {n_total} OCs | {n_cron} cronicamente deficitários | "
              f"{n_marc} marcadores | prob. média {prob_med:.2f}")
    
    print(f"\n[OK] Resultados em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
