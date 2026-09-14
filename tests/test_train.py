import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.modeling import train
from tests.conftest import montar_base_sintetica


def test_treinar_logistica_ponta_a_ponta():
    base = montar_base_sintetica(n_escolas=40, alunos_por_escola=15)
    r = train.treinar(base, "producao", modelo="logistica", recall_minimo=0.7, cv_municipio=True)
    m = r["metricas"]
    assert m["regime"] == "producao" and m["modelo"] == "logistica"
    assert m["n_treino"] + m["n_validacao"] + m["n_teste"] == 600
    assert 0.5 < m["teste"]["roc_auc"] <= 1.0                     # há sinal na base sintética
    assert m["validacao"]["recall_nao_alf"] >= 0.7
    assert set(m["teste_ponderado"]) == set(m["teste"])
    assert isinstance(m["por_uf"], list) and isinstance(m["calibracao"], list)
    assert m["cv_municipio"] is not None and "roc_auc_media" in m["cv_municipio"]
    json.dumps(m)                                                 # serializável
    escolas_teste = set(base.iloc[r["partes"]["teste"]]["id_escola"])
    assert not escolas_teste & set(base.iloc[r["partes"]["treino"]]["id_escola"])


def test_diagnostico_usa_loo_e_producao_nao():
    base = montar_base_sintetica(n_escolas=20, alunos_por_escola=10)
    prod = train.treinar(base, "producao", modelo="dummy")
    diag = train.treinar(base, "diagnostico", modelo="dummy")
    assert "taxa_escola_loo" not in prod["colunas"]["numericas"]
    assert "taxa_escola_loo" in diag["colunas"]["numericas"]


def test_salvar_grava_modelo_particao_e_metricas(lake_tmp):
    base = montar_base_sintetica(n_escolas=20, alunos_por_escola=10)
    r = train.treinar(base, "producao", modelo="dummy")
    caminhos = train.salvar(r, base)
    assert caminhos["modelo"] == config.MODELS / "modelo_aluno_producao_dummy.joblib"
    assert joblib.load(caminhos["modelo"]).predict_proba(base.head(2)[r["colunas"]["numericas"] + r["colunas"]["categoricas"]]).shape == (2, 2)
    part = pd.read_parquet(caminhos["particao"])
    assert set(part["parte"]) == {"treino", "validacao", "teste"} and len(part) == 200
    assert json.loads(caminhos["metricas"].read_text())["modelo"] == "dummy"


def test_cli_com_amostra(lake_tmp, monkeypatch):
    base = montar_base_sintetica(n_escolas=30, alunos_por_escola=10)
    config.PROCESSED.mkdir(exist_ok=True)
    base.to_parquet(config.PROCESSED / "base_modelagem_aluno.parquet")
    train.main(["--regime", "producao", "--modelo", "dummy", "--amostra-escolas", "10"])
    m = json.loads((config.REPORTS / "metricas_producao_dummy.json").read_text())
    assert m["n_treino"] + m["n_validacao"] + m["n_teste"] == 100
