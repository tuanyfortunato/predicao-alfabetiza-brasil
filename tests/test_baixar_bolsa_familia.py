import pandas as pd
import pytest

from scripts.baixar_bolsa_familia import COLUNAS, mapa_codigos, processar


def _diretorios():
    return pd.DataFrame({"id_municipio": [3550308, 1100015, 5101837], "nome_municipio": ["São Paulo", "Alta Floresta D'Oeste", "Boa Esperança do Norte"]})


def _bruto():
    linhas = []
    for anomes in (202311, 202312):
        linhas += [
            {"codigo_ibge": 355030, "anomes_s": anomes, "qtd_ben_brc": 100 + anomes % 100, "qtd_ben_bf": 30, "qtd_ben_bpi": 5, "qtd_ben_sus": None},
            {"codigo_ibge": 110001, "anomes_s": anomes, "qtd_ben_brc": 10, "qtd_ben_bf": 3, "qtd_ben_bpi": 1, "qtd_ben_sus": 0},
            {"codigo_ibge": 999999, "anomes_s": anomes, "qtd_ben_brc": 1, "qtd_ben_bf": 1, "qtd_ben_bpi": 0, "qtd_ben_sus": 0},
            {"codigo_ibge": 510183, "anomes_s": anomes, "qtd_ben_brc": None, "qtd_ben_bf": None, "qtd_ben_bpi": None, "qtd_ben_sus": None},
        ]
    return pd.DataFrame(linhas)


def test_mapa_codigos_descarta_o_digito_verificador():
    assert mapa_codigos(_diretorios()) == {355030: 3550308, 110001: 1100015, 510183: 5101837}


def test_processar_usa_dezembro_converte_id_e_renomeia():
    out = processar(_bruto(), 2023, mapa_codigos(_diretorios()))
    assert list(out.columns) == ["id_municipio", "ano", *COLUNAS.values()]
    assert sorted(out.id_municipio) == [1100015, 3550308]          # 999999 não tem 7 dígitos: fora
    sp = out[out.id_municipio == 3550308].iloc[0]
    assert sp.pessoas_brc == 112 and sp.familias_bf == 30 and sp.ben_primeira_infancia == 5
    assert (out.ano == 2023).all() and out.id_municipio.dtype.kind == "i"


def test_processar_falha_claro_sem_dezembro():
    bruto = _bruto()
    with pytest.raises(ValueError, match="dezembro de 2024"):
        processar(bruto, 2024, mapa_codigos(_diretorios()))


def test_processar_descarta_municipio_sem_nenhum_valor_e_mantem_inteiros():
    out = processar(_bruto(), 2023, mapa_codigos(_diretorios()))
    assert 5101837 not in set(out.id_municipio)
    assert all(str(out[c].dtype) == "int64" for c in COLUNAS.values())
