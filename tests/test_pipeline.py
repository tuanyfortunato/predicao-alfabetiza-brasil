import numpy as np
import pandas as pd
import pytest

from src.preprocessing import features, pipeline
from tests.conftest import montar_base_sintetica


@pytest.mark.parametrize("modelo", ["dummy", "logistica", "hgb"])
def test_pipeline_treina_com_nan_e_categoria_nova(modelo):
    base = montar_base_sintetica(n_escolas=20, alunos_por_escola=15)
    X, y, _, _ = features.separar_xy(base, "producao")
    num, cat = features.colunas_por_regime(base, "producao")
    pipe = pipeline.build_pipeline(modelo, num, cat)
    pipe.fit(X, y)
    novo = X.head(3).copy()
    novo.loc[novo.index[0], "sigla_uf"] = "ZZ"                    # categoria nunca vista
    novo.loc[novo.index[1], "log_pib_per_capita"] = np.nan
    proba = pipe.predict_proba(novo)[:, 1]
    assert proba.shape == (3,) and np.all((proba >= 0) & (proba <= 1))
    nomes = pipe.named_steps["prep"].get_feature_names_out()
    assert any("missingindicator" in n for n in nomes)            # add_indicator ligado
    assert not any(n.startswith("remainder") for n in nomes)      # nada passa sem transformar


def test_params_com_prefixo_clf_chegam_no_modelo():
    pipe = pipeline.build_pipeline("hgb", ["a"], [], params={"clf__max_depth": 3, "clf__learning_rate": 0.2})
    assert pipe.named_steps["clf"].max_depth == 3
    assert pipe.named_steps["clf"].random_state == 42


def test_modelo_desconhecido():
    with pytest.raises(ValueError, match="modelo"):
        pipeline.build_pipeline("xgboost", ["a"], [])
