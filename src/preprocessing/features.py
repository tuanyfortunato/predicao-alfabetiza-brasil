"""Quais colunas entram no modelo, por regime, e a guarda contra vazamento."""
import pandas as pd

from src import config

CATEGORICAS = ["rede_nome", "sigla_uf", "regiao"]
DIAGNOSTICO = ["taxa_escola_loo", "prof_media_escola_loo", "n_alunos_escola", "taxa_participacao_escola_loo"]
NAO_FEATURES = frozenset(config.COLUNAS_PROIBIDAS | {"ano", "nome_municipio"})
REGIMES = ("producao", "diagnostico")
ALVO = "alfabetizado"


def verificar_leakage(colunas) -> None:
    vazadas = sorted(set(colunas) & config.COLUNAS_PROIBIDAS)
    if vazadas:
        raise ValueError(f"colunas proibidas na matriz de features: {vazadas}")


def colunas_por_regime(df: pd.DataFrame, regime: str) -> tuple[list[str], list[str]]:
    if regime not in REGIMES:
        raise ValueError(f"regime deve ser um de {REGIMES}, veio {regime!r}")
    cat = [c for c in CATEGORICAS if c in df.columns]
    num = []
    for c in df.columns:
        if c in NAO_FEATURES or c in cat:
            continue
        if regime == "producao" and c in DIAGNOSTICO:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            num.append(c)
    return num, cat


def separar_xy(df: pd.DataFrame, regime: str):
    num, cat = colunas_por_regime(df, regime)
    verificar_leakage(num + cat)
    X = df[num + cat].copy()
    X[num] = X[num].astype("float64")
    return X, df[ALVO].astype(int), df["id_escola"], df["peso_aluno"]
