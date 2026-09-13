import json

import pandas as pd
import pytest

from scripts.extrair_externas import CONSULTAS, NOMES, extrair, registrar_metadados, validar


class _ClienteFalso:
    """Imita client.query(sql).result().to_dataframe() e job.total_bytes_processed."""
    total_bytes_processed = 1234

    def __init__(self, df):
        self._df = df

    def query(self, sql):
        return self

    def result(self, timeout=None):
        return self

    def to_dataframe(self):
        return self._df


def test_sete_fontes_com_cast_do_id_municipio():
    assert sorted(CONSULTAS) == sorted(NOMES) and len(NOMES) == 7
    for nome, c in CONSULTAS.items():
        assert "CAST(id_municipio AS INT64)" in c["sql"], nome
        assert "id_municipio" in c["chave"], nome


def test_extrair_grava_parquet_e_resume(tmp_path):
    df = pd.DataFrame({"id_municipio": [3550308, 3304557], "ano": [2023, 2023], "x": [1.0, 2.0]})
    resumo = extrair(_ClienteFalso(df), "pib_municipio", tmp_path)
    assert (tmp_path / "pib_municipio.parquet").exists()
    assert resumo == {"linhas": 2, "bytes_processados": 1234}


def test_validar_rejeita_id_curto_chave_duplicada_e_vazio():
    with pytest.raises(ValueError, match="7 dígitos"):
        validar(pd.DataFrame({"id_municipio": [355030]}), ["id_municipio"])
    with pytest.raises(ValueError, match="duplicad"):
        validar(pd.DataFrame({"id_municipio": [3550308, 3550308], "ano": [2023, 2023]}), ["id_municipio", "ano"])
    with pytest.raises(ValueError, match="vazia"):
        validar(pd.DataFrame({"id_municipio": []}), ["id_municipio"])


def test_registrar_metadados_acumula_por_fonte(tmp_path):
    registrar_metadados(tmp_path, "a", {"linhas": 1})
    registrar_metadados(tmp_path, "b", {"linhas": 2})
    registrar_metadados(tmp_path, "a", {"linhas": 3})
    meta = json.loads((tmp_path / "_metadados.json").read_text(encoding="utf-8"))
    assert meta["a"]["linhas"] == 3 and meta["b"]["linhas"] == 2
    assert "extraido_em" in meta["a"]
