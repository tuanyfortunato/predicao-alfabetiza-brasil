import json

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import config
from src.modeling import predict, train
from tests.conftest import montar_base_sintetica


@pytest.fixture
def treinado(lake_tmp):
    base = montar_base_sintetica(n_escolas=30, alunos_por_escola=15)
    r = train.treinar(base, "producao", modelo="logistica")
    train.salvar(r, base)
    config.PROCESSED.mkdir(exist_ok=True)
    base.to_parquet(config.PROCESSED / "base_modelagem_aluno.parquet")
    return base, r


def test_pontuar_reproduz_a_metrica_gravada(treinado):
    base, r = treinado
    pipe = predict.carregar_modelo(regime="producao", modelo="logistica")
    teste = base.iloc[r["partes"]["teste"]]
    pred = predict.pontuar(pipe, teste, limiar=r["metricas"]["limiar"])
    assert list(pred.columns) == ["id_aluno", "id_escola", "id_municipio", "sigla_uf", "prob_alfabetizado", "risco_nao_alf", "classe_prevista"]
    # 4 casas de arredondamento podem criar empates; a tolerância cobre isso numa partição de ~70 linhas
    assert roc_auc_score(teste["alfabetizado"], pred["prob_alfabetizado"]) == pytest.approx(r["metricas"]["teste"]["roc_auc"], abs=5e-3)
    assert set(pred["classe_prevista"]) <= {"alfabetizado", "nao_alfabetizado"}
    assert (pred["prob_alfabetizado"] + pred["risco_nao_alf"]).round(6).eq(1).all()


def test_cli_so_teste_usa_limiar_do_json_e_arredonda(treinado):
    base, r = treinado
    predict.main(["--regime", "producao", "--modelo", "logistica", "--so-teste"])
    saida = pd.read_csv(config.REPORTS / "predicoes_producao_logistica.csv")
    assert len(saida) == len(r["partes"]["teste"])
    limiar = json.loads((config.REPORTS / "metricas_producao_logistica.json").read_text())["limiar"]
    assert ((saida["prob_alfabetizado"] < limiar) == (saida["classe_prevista"] == "nao_alfabetizado")).all()
    assert (saida["prob_alfabetizado"] * 1e4).round(6).eq((saida["prob_alfabetizado"] * 1e4).round()).all()
