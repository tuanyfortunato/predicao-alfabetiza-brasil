"""Quais colunas entram no modelo, por regime, e a guarda contra vazamento."""
import numpy as np
import pandas as pd

from src import config

CATEGORICAS = ["rede_nome", "sigla_uf", "regiao"]
DIAGNOSTICO = ["taxa_escola_loo", "prof_media_escola_loo", "n_alunos_escola", "taxa_participacao_escola_loo"]
NAO_FEATURES = frozenset(config.COLUNAS_PROIBIDAS | {"ano", "nome_municipio"})
REGIMES = ("producao", "diagnostico")
ALVO = "alfabetizado"


def verificar_leakage(colunas) -> None:
    # confere se alguma coluna proibida (tipo a nota ou o alvo) foi parar
    # sem querer na base de features -- se achar, quebra na hora
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
            continue  # essas colunas só existem calculadas no mesmo ano da prova, produção não pode ver isso
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            num.append(c)
    return num, cat


def separar_xy(df: pd.DataFrame, regime: str):
    num, cat = colunas_por_regime(df, regime)
    verificar_leakage(num + cat)  # cinto de segurança: mesmo que colunas_por_regime já filtre, confere de novo
    X = df[num + cat].copy()
    X[num] = X[num].astype("float64")
    # devolve grupo (id_escola, pro split por escola) e peso (peso_aluno, nunca vira feature)
    return X, df[ALVO].astype(int), df["id_escola"], df["peso_aluno"]


def correlacao_alta(df: pd.DataFrame, limite: float = 0.95) -> pd.DataFrame:
    corr = df.corr(numeric_only=True).abs()
    # triu com k=1 pega só o triângulo acima da diagonal -- sem isso cada par apareceria
    # duas vezes (a-b e b-a) e a diagonal (correlação de uma coluna com ela mesma) também entraria
    pares = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    out = pares[pares > limite].reset_index()
    out.columns = ["a", "b", "rho"]
    return out.sort_values("rho", ascending=False).reset_index(drop=True)


def calcular_vif(df: pd.DataFrame) -> pd.DataFrame:
    # colunas quase vazias (ex.: pct_va_* e projecao_ideb_ai, 100% NaN no alvo 2024) derrubariam todas as linhas no dropna
    X = df.select_dtypes("number")
    X = X.loc[:, X.notna().mean() > 0.5].dropna()
    X = X.loc[:, X.std() > 0]  # coluna constante também quebraria a regressão (variância zero)
    linhas = []
    for c in X.columns:
        # VIF na mão via mínimos quadrados (sem puxar statsmodels): regride a coluna c
        # contra todas as outras e vê quanto da variância dela é "explicada" pelas demais
        y = X[c].to_numpy()
        A = np.column_stack([np.ones(len(X)), X.drop(columns=c).to_numpy()])
        residuo = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        r2 = 1 - residuo.var() / y.var()
        linhas.append({"feature": c, "vif": float(1 / max(1 - r2, 1e-12))})  # trava o r2=1 pra não dar divisão por zero
    return pd.DataFrame(linhas).sort_values("vif", ascending=False).reset_index(drop=True)
