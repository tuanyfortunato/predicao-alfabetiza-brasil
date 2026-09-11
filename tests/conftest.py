"""Fixtures compartilhadas. Nenhum teste lê data/ de verdade."""
import pandas as pd
import pytest

from src import config


@pytest.fixture
def lake_tmp(tmp_path, monkeypatch):
    """Redireciona todos os caminhos de dados para uma pasta temporária."""
    for nome in ["DATA", "GOLD", "SILVER", "EXTERNAL", "PROCESSED", "MODELS", "REPORTS", "IMAGES"]:
        destino = tmp_path / nome.lower()
        destino.mkdir()
        monkeypatch.setattr(config, nome, destino)
    return tmp_path


def montar_alunos(linhas: list[dict]) -> pd.DataFrame:
    """Linhas com as colunas da Silver de alunos; ausentes vêm sem proficiencia."""
    base = {
        "ano": 2024, "id_municipio": 3550308, "id_escola": 60000001, "sigla_uf": "SP",
        "rede_nome": "municipal", "caderno": "1", "serie": 2, "rede": 3,
        "presenca": 1, "preenchimento_caderno": 1, "peso_aluno": 1.0,
    }
    df = pd.DataFrame([{**base, **l} for l in linhas])
    df["id_aluno"] = range(1, len(df) + 1)
    df["presente"] = df["presenca"] == 1
    df["sem_nota"] = df["proficiencia"].isna()
    df["alfabetizado"] = (df["proficiencia"] >= config.CORTE_ALFABETIZACAO).fillna(False).astype(int)
    return df
