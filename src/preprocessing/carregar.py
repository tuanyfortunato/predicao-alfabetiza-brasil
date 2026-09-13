"""Leitura das bases de entrada (Gold/Silver da Fase 2 e fontes externas)."""
import pandas as pd

from src import config


def carregar_gold(nome: str) -> pd.DataFrame:
    return pd.read_parquet(config.GOLD / f"{nome}.parquet")


def carregar_metas() -> pd.DataFrame:
    return pd.read_parquet(config.SILVER / "metas.parquet")


def carregar_externa(nome: str) -> pd.DataFrame:
    caminho = config.EXTERNAL / f"{nome}.parquet"
    if not caminho.exists():
        raise FileNotFoundError(f"fonte externa {nome} não encontrada em {caminho}; rode scripts/extrair_externas.py")
    return pd.read_parquet(caminho)


def carregar_alunos(ano: int | None = None, apenas_com_nota: bool = True) -> pd.DataFrame:
    filtros = [("ano", "==", ano)] if ano is not None else None
    df = pd.read_parquet(config.SILVER / "alunos", filters=filtros)
    # a partição vem como category; int evita surpresa em groupby e merge
    df["ano"] = df["ano"].astype("int64")
    if apenas_com_nota:
        df = df[df["presente"] & ~df["sem_nota"]]
    return df.reset_index(drop=True)
