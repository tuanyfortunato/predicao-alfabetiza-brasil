import json

import numpy as np

from src import config
from src.modeling import risco_municipio as rm
from tests.conftest import montar_base_municipio_sintetica


def test_preparar_treino_sem_alvo_nem_colunas_instaveis_nas_features():
    base = montar_base_municipio_sintetica()
    X, y_reg, y_clf, ids = rm.preparar_treino(base)
    assert len(X) == 200 and not set(X.columns) & rm.NAO_FEATURES_B
    assert "meta_prox" in X.columns and "taxa_alfabetizacao" in X.columns
    assert "pct_familias_bolsa_familia" not in X.columns and "pct_va_servicos" not in X.columns
    assert y_clf.isna().sum() == 10                            # sem_meta fica fora da classificação


def test_avaliar_bate_a_persistencia_e_treina_final():
    base = montar_base_municipio_sintetica()
    X, y_reg, y_clf, _ = rm.preparar_treino(base)
    res = rm.avaliar(X, y_reg, y_clf)
    assert {"persistencia", "ridge", "hgb_reg", "logistica", "hgb_clf"} <= set(res["modelo"])
    mae = res[(res.metrica == "mae")].set_index("modelo")["media"]
    assert mae["hgb_reg"] < mae["persistencia"] or mae["ridge"] < mae["persistencia"]
    reg, clf = rm.treinar_final(X, y_reg, y_clf)
    assert reg.predict(X.head(2)).shape == (2,) and clf.predict_proba(X.head(2)).shape == (2, 2)


def test_ranking_2025():
    base = montar_base_municipio_sintetica()
    X, y_reg, y_clf, _ = rm.preparar_treino(base)
    reg, clf = rm.treinar_final(X, y_reg, y_clf)
    rank = rm.gerar_ranking(base, reg, clf)
    assert len(rank) == 200 and rank["id_municipio"].is_unique
    assert rank["prob_nao_atingir_2025"].between(0, 1).all()
    assert rank["taxa_prevista_2025"].between(0, 100).all()
    assert rank["prioridade"].is_monotonic_decreasing
    assert np.allclose(rank["gap_previsto_2025"], rank["taxa_prevista_2025"] - rank["meta_2025"])
    assert rank["meta_2025"].eq(0.65).all()


def test_cli(lake_tmp):
    config.PROCESSED.mkdir(exist_ok=True)
    montar_base_municipio_sintetica().to_parquet(config.PROCESSED / "base_modelagem_municipio.parquet")
    rm.main([])
    assert (config.REPORTS / "ranking_risco_municipios.csv").exists()
    assert (config.MODELS / "modelo_b_regressao.joblib").exists()
    m = json.loads((config.REPORTS / "metricas_modelo_b.json").read_text())
    assert "avaliacao" in m and m["n_treino"] == 200 and m["n_com_meta"] == 190
