import numpy as np
import pandas as pd
import pytest

from src.evaluation import interpret
from src.preprocessing import features, pipeline
from tests.conftest import montar_base_sintetica


@pytest.fixture(scope="module")
def ajustado():
    base = montar_base_sintetica(n_escolas=30, alunos_por_escola=20)
    X, y, _, pesos = features.separar_xy(base, "producao")
    num, cat = features.colunas_por_regime(base, "producao")
    hgb = pipeline.build_pipeline("hgb", num, cat).fit(X, y)
    log = pipeline.build_pipeline("logistica", num, cat).fit(X, y)
    return base, X, y, pesos, hgb, log


def test_nomes_features_sem_prefixo(ajustado):
    _, X, _, _, hgb, _ = ajustado
    nomes = interpret.nomes_features(hgb)
    assert "log_pib_per_capita" in nomes and "sigla_uf_SP" in nomes and "faltante_tdi_ai" in nomes
    assert not any(n.startswith(("num__", "cat__")) for n in nomes)
    assert len(nomes) == hgb.named_steps["prep"].transform(X.head(1)).shape[1]


def test_permutacao_uma_linha_por_coluna_crua(ajustado):
    _, X, y, pesos, hgb, _ = ajustado
    imp = interpret.importancia_permutacao(hgb, X, y, n_repeats=2, pesos=pesos, n_amostra=300)
    assert set(imp["feature"]) == set(X.columns)
    assert imp["importancia_media"].iloc[0] >= imp["importancia_media"].iloc[-1]


def test_shap_para_arvore_e_linear(ajustado):
    _, X, _, _, hgb, log = ajustado
    for pipe in (hgb, log):
        exp = interpret.explicar_shap(pipe, X, n_amostra=100)
        assert exp.values.shape == (100, len(interpret.nomes_features(pipe)))
        resumo = interpret.resumo_shap(exp)
        assert resumo["shap_medio_abs"].ge(0).all() and len(resumo) == exp.values.shape[1]
    grupo = interpret.shap_por_grupo(exp, X["regiao"].head(100))
    assert set(grupo.index) <= {"Norte", "Nordeste", "Sudeste"}


def test_coeficientes_logistica(ajustado):
    *_, log = ajustado
    coef = interpret.coeficientes_logistica(log)
    assert "coeficiente" in coef.columns and "taxa_alfabetizacao_mun_t1" in coef["feature"].values
    with pytest.raises(TypeError):
        interpret.coeficientes_logistica(ajustado[4])
