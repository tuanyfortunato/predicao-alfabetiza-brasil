import numpy as np
import pandas as pd
import pytest

from src.modeling import split


def _grupos(n_escolas=100, por_escola=10):
    return pd.Series(np.repeat(np.arange(n_escolas), por_escola))


def test_nenhuma_escola_em_duas_partes_e_proporcoes():
    g = _grupos()
    partes = split.dividir_por_escola(g)
    assert set(partes) == {"treino", "validacao", "teste"}
    total = sum(len(v) for v in partes.values())
    assert total == len(g) and len(np.intersect1d(partes["treino"], partes["teste"])) == 0
    split.conferir_sem_vazamento(g, partes)                      # não levanta
    assert 0.6 <= len(partes["treino"]) / total <= 0.8
    assert 0.1 <= len(partes["teste"]) / total <= 0.2


def test_split_e_deterministico():
    g = _grupos()
    a, b = split.dividir_por_escola(g, seed=42), split.dividir_por_escola(g, seed=42)
    assert np.array_equal(a["teste"], b["teste"])
    assert not np.array_equal(a["teste"], split.dividir_por_escola(g, seed=7)["teste"])


def test_conferir_detecta_vazamento():
    g = _grupos(n_escolas=2, por_escola=2)
    with pytest.raises(AssertionError, match="escola"):
        split.conferir_sem_vazamento(g, {"treino": np.array([0, 1, 2]), "teste": np.array([3])})


def test_cv_por_grupo_nao_mistura_escolas():
    g = _grupos(n_escolas=20, por_escola=5)
    y = (np.arange(len(g)) % 3 == 0).astype(int)
    X = np.zeros((len(g), 1))
    for tr, te in split.cv_por_grupo(n_splits=4).split(X, y, g):
        assert not set(g.iloc[tr]) & set(g.iloc[te])
