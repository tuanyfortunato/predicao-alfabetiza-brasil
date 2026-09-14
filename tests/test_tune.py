import json

import pandas as pd

from src.modeling import tune
from tests.conftest import montar_base_sintetica


def test_amostra_por_escola_nao_parte_escolas():
    base = montar_base_sintetica(n_escolas=40, alunos_por_escola=10)
    am = tune.amostrar_por_escola(base, n_alunos=100)
    assert 100 <= len(am) < 130
    contagem = base.groupby("id_escola").size()
    assert (am.groupby("id_escola").size() == contagem.loc[am["id_escola"].unique()]).all()


def test_buscar_devolve_params_serializaveis_e_resultados():
    base = montar_base_sintetica(n_escolas=40, alunos_por_escola=15)
    params, res = tune.buscar(base, "producao", modelo="logistica", n_amostra=600, n_candidatos=3)
    assert params and all(k.startswith("clf__") for k in params)
    json.dumps(params)
    assert isinstance(res, pd.DataFrame) and "mean_test_score" in res.columns and len(res) >= 3
