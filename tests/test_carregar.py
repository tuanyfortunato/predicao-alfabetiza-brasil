import pandas as pd
import pytest

from src import config
from src.preprocessing import carregar
from tests.conftest import montar_alunos


def _grava_alunos(silver, ano, linhas):
    p = silver / "alunos" / f"ano={ano}"
    p.mkdir(parents=True)
    montar_alunos(linhas).drop(columns="ano").to_parquet(p / "parte.parquet")


def test_carregar_alunos_filtra_ano_e_presentes_com_nota(lake_tmp):
    _grava_alunos(config.SILVER, 2023, [{"proficiencia": 700.0}])
    _grava_alunos(config.SILVER, 2024, [
        {"proficiencia": 760.0},
        {"proficiencia": None, "presenca": 0, "preenchimento_caderno": 0, "peso_aluno": None},
    ])
    df = carregar.carregar_alunos(ano=2024)
    assert len(df) == 1
    assert df["ano"].dtype.kind == "i"
    assert df["ano"].iloc[0] == 2024
    assert df.index.tolist() == [0]

    todos = carregar.carregar_alunos(ano=2024, apenas_com_nota=False)
    assert len(todos) == 2


def test_carregar_alunos_sem_ano_le_todas_as_particoes(lake_tmp):
    _grava_alunos(config.SILVER, 2023, [{"proficiencia": 700.0}])
    _grava_alunos(config.SILVER, 2024, [{"proficiencia": 760.0}])
    assert sorted(carregar.carregar_alunos()["ano"].unique()) == [2023, 2024]


def test_carregar_gold_e_metas_e_externa_por_nome(lake_tmp):
    pd.DataFrame({"ano": [2023], "id_municipio": [1]}).to_parquet(config.GOLD / "indicador_municipio.parquet")
    pd.DataFrame({"ano": [2024], "meta_alfabetizacao_2025": [60.0]}).to_parquet(config.SILVER / "metas.parquet")
    pd.DataFrame({"id_municipio": [1], "populacao": [10]}).to_parquet(config.EXTERNAL / "censo2022_municipio.parquet")
    assert list(carregar.carregar_gold("indicador_municipio").columns) == ["ano", "id_municipio"]
    assert "meta_alfabetizacao_2025" in carregar.carregar_metas().columns
    assert carregar.carregar_externa("censo2022_municipio")["populacao"].iloc[0] == 10


def test_externa_inexistente_da_erro_claro(lake_tmp):
    with pytest.raises(FileNotFoundError, match="nao_existe"):
        carregar.carregar_externa("nao_existe")
