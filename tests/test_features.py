import pytest

from src import config
from src.preprocessing import features
from tests.conftest import montar_base_sintetica


def test_verificar_leakage_barra_colunas_proibidas():
    with pytest.raises(ValueError, match="proficiencia"):
        features.verificar_leakage(["log_pib_per_capita", "proficiencia"])
    features.verificar_leakage(["log_pib_per_capita"])          # não levanta


def test_colunas_por_regime():
    base = montar_base_sintetica(n_escolas=5, alunos_por_escola=4)
    num, cat = features.colunas_por_regime(base, "producao")
    assert cat == ["rede_nome", "sigla_uf", "regiao"]
    assert "sem_historico" in num and "taxa_alfabetizacao_mun_t1" in num and "matriculas_ai" in num
    assert not set(num) & config.COLUNAS_PROIBIDAS and not set(num) & set(features.DIAGNOSTICO)
    assert "nome_municipio" not in num and "ano" not in num
    num_d, _ = features.colunas_por_regime(base, "diagnostico")
    assert set(features.DIAGNOSTICO) <= set(num_d)
    with pytest.raises(ValueError):
        features.colunas_por_regime(base, "outro")


def test_separar_xy_devolve_x_limpo_grupos_e_pesos():
    base = montar_base_sintetica(n_escolas=5, alunos_por_escola=4)
    X, y, grupos, pesos = features.separar_xy(base, "producao")
    assert not set(X.columns) & config.COLUNAS_PROIBIDAS
    assert y.name == "alfabetizado" and set(y.unique()) <= {0, 1}
    assert grupos.name == "id_escola" and pesos.name == "peso_aluno"
    assert X["sem_historico"].dtype == "float64" and X["matriculas_ai"].dtype == "float64"
    assert len(X) == len(y) == len(grupos) == len(pesos) == 20
