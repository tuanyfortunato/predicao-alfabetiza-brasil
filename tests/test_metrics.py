import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics


def test_predicao_perfeita_e_aleatoria():
    y = np.array([1, 1, 1, 0, 0, 0])
    m = metrics.calcular_metricas(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]))
    assert m["roc_auc"] == 1.0 and m["f1_nao_alf"] == 1.0 and m["recall_nao_alf"] == 1.0
    assert m["matriz"] == [[3, 0], [0, 3]] and m["n"] == 6 and m["prevalencia_nao_alf"] == 0.5
    m2 = metrics.calcular_metricas(y, np.full(6, 0.5))
    assert m2["roc_auc"] == 0.5 and m2["brier"] == pytest.approx(0.25)


def test_pesos_mudam_a_metrica():
    y = np.array([1, 0, 0, 1])
    proba = np.array([0.9, 0.6, 0.2, 0.4])
    sem = metrics.calcular_metricas(y, proba)
    com = metrics.calcular_metricas(y, proba, pesos=np.array([1, 10, 1, 1]))
    assert sem["acuracia"] == 0.5 and com["acuracia"] < 0.5


def test_escolher_limiar_garante_recall_minimo():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.6 * y + rng.normal(0, 0.3, 2000), 0, 1)
    limiar = metrics.escolher_limiar(y, proba, recall_minimo=0.8)
    m = metrics.calcular_metricas(y, proba, limiar)
    assert m["recall_nao_alf"] >= 0.8
    assert metrics.calcular_metricas(y, proba, limiar - 0.02)["recall_nao_alf"] < m["recall_nao_alf"]


def test_recorte_e_calibracao():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1200)
    proba = np.clip(y * 0.5 + rng.random(1200) * 0.5, 0, 1)
    uf = pd.Series(np.where(np.arange(1200) < 900, "SP", "AC"))
    r = metrics.metricas_por_recorte(uf, y, proba, 0.5, minimo=500)
    assert r["recorte"].tolist() == ["SP"] and r.loc[0, "n"] == 900
    cal = metrics.tabela_calibracao(y, proba, n_bins=5)
    assert len(cal) <= 5 and cal["n"].sum() == 1200 and cal["taxa_observada"].between(0, 1).all()


def test_recorte_sem_grupo_valido_retorna_dataframe_vazio():
    recorte = ["A"] * 10 + ["B"] * 10
    y = np.array([1, 0] * 10)
    proba = np.random.default_rng(0).random(20)
    r = metrics.metricas_por_recorte(recorte, y, proba, 0.5, minimo=500)
    assert len(r) == 0
    assert "recorte" in r.columns and "roc_auc" in r.columns and "matriz" not in r.columns


def test_escolher_limiar_nao_ultrapassa_recall_com_probabilidades_empatadas():
    proba0 = np.array([0.1] * 50 + [0.2] * 50 + [0.3] * 50 + [0.4] * 50)
    y0 = np.zeros_like(proba0, dtype=int)
    limiar = metrics.escolher_limiar(y0, proba0, recall_minimo=0.5)
    recall = (proba0 < limiar).mean()
    assert recall == pytest.approx(0.5)
