import pandas as pd
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


def test_correlacao_alta_e_vif():
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.01, size=200), "c": rng.normal(size=200)})
    pares = features.correlacao_alta(df, limite=0.95)
    assert list(pares.columns) == ["a", "b", "rho"] and len(pares) == 1 and set(pares.iloc[0][["a", "b"]]) == {"a", "b"}
    vif = features.calcular_vif(df)
    assert list(vif.columns) == ["feature", "vif"] and vif.set_index("feature").loc["a", "vif"] > 10 > vif.set_index("feature").loc["c", "vif"]
