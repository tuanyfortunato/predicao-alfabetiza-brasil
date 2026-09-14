import numpy as np
import pandas as pd
import pytest

from src.preprocessing import contexto
from tests.conftest import montar_alunos


def test_loo_exclui_o_proprio_aluno_e_da_nan_para_escola_solitaria():
    alunos = montar_alunos([
        {"id_escola": 1, "proficiencia": 700.0},
        {"id_escola": 1, "proficiencia": 750.0},
        {"id_escola": 1, "proficiencia": 800.0},
        {"id_escola": 2, "proficiencia": 760.0},
    ])
    loo = contexto.contexto_escola_loo(alunos)
    assert list(loo.columns) == ["taxa_escola_loo", "prof_media_escola_loo", "n_alunos_escola"]
    assert loo.loc[0, "prof_media_escola_loo"] == pytest.approx(775.0)
    assert loo.loc[0, "taxa_escola_loo"] == pytest.approx(1.0)      # 750 e 800 >= 743
    assert loo.loc[1, "taxa_escola_loo"] == pytest.approx(0.5)      # 700 e 800
    assert loo.loc[0, "n_alunos_escola"] == 2
    assert np.isnan(loo.loc[3, "prof_media_escola_loo"]) and loo.loc[3, "n_alunos_escola"] == 0


def test_participacao_loo_desconta_o_aluno():
    alunos = montar_alunos([{"id_escola": 1, "proficiencia": 750.0}, {"id_escola": 1, "proficiencia": 700.0}])
    perfil = pd.DataFrame({"ano": [2024], "id_escola": [1], "alunos_avaliados": [11], "alunos_presentes": [10]})
    p = contexto.participacao_escola_loo(alunos, perfil)
    assert p.tolist() == pytest.approx([0.9, 0.9])
    assert p.name == "taxa_participacao_escola_loo"


def test_contexto_municipal_usa_apenas_o_ano_anterior():
    indicador = pd.DataFrame({
        "ano": [2023, 2024, 2024], "id_municipio": [1, 1, 2], "sigla_uf": ["SP"] * 3,
        "alunos_avaliados": [100, 120, 50], "alunos_presentes": [90, 110, 45], "alunos_com_nota": [90, 110, 45],
        "taxa_participacao": [0.9, 0.92, 0.9], "taxa_alfabetizacao": [0.6, 0.7, 0.5], "ic95": [0.05, 0.04, 0.1],
        "taxa_limite_inferior": [0.55, 0.66, 0.4], "taxa_limite_superior": [0.65, 0.74, 0.6],
        "proficiencia_media": [750.0, 760.0, 740.0], "criancas_nao_alfabetizadas": [40.0, 36.0, 25.0],
        "alerta_participacao": [False, False, True],
    })
    dist = pd.DataFrame({
        "ano": [2023, 2023, 2024], "nivel": ["municipio"] * 3, "rede": ["total", "municipal", "total"],
        "sigla_uf": ["SP"] * 3, "id_municipio": [1, 1, 1], "alunos_com_nota": [90, 80, 110],
        **{f"pct_nivel_{i}": [0.1, 0.2, 0.3] for i in range(9)},
        "pct_critico": [0.2, 0.3, 0.1], "pct_atencao": [0.2, 0.2, 0.2], "pct_alfabetizado": [0.6, 0.5, 0.7], "pct_quase_la": [0.1, 0.1, 0.1],
    })
    ctx = contexto.contexto_municipal_defasado(2024, indicador, dist)
    assert ctx["id_municipio"].tolist() == [1]                      # município 2 não tem 2023
    assert ctx.loc[0, "taxa_alfabetizacao_mun_t1"] == 0.6
    assert ctx.loc[0, "pct_critico_mun_t1"] == 0.2                  # rede total, não municipal
    assert "ano" not in ctx.columns and "sigla_uf" not in ctx.columns
    assert all(c.endswith("_mun_t1") or c == "id_municipio" for c in ctx.columns)


def test_meta_pactuada_do_ano_alvo():
    metas = pd.DataFrame({
        "ano": [2024, 2024, 2024], "nivel": ["municipio", "municipio", "uf"], "rede_padronizada": ["municipal"] * 3,
        "id_municipio": [1, 2, None], "meta_alfabetizacao_2024": [0.55, None, 0.6], "meta_alfabetizacao_2025": [0.6, 0.62, 0.65],
    })
    m = contexto.meta_pactuada(2024, metas).set_index("id_municipio")["meta_alvo"]
    assert len(m) == 2 and m[1] == 0.55 and np.isnan(m[2])
    assert contexto.meta_pactuada(2025, metas).set_index("id_municipio").loc[2, "meta_alvo"] == 0.62


def test_decompor_variancia_isola_o_componente_de_escola():
    # dois municípios com a mesma média; dentro de cada um, duas escolas homogêneas e diferentes entre si
    alunos = montar_alunos([
        {"id_municipio": 1, "id_escola": 1, "proficiencia": 700.0}, {"id_municipio": 1, "id_escola": 1, "proficiencia": 700.0},
        {"id_municipio": 1, "id_escola": 2, "proficiencia": 800.0}, {"id_municipio": 1, "id_escola": 2, "proficiencia": 800.0},
        {"id_municipio": 2, "id_escola": 3, "proficiencia": 700.0}, {"id_municipio": 2, "id_escola": 3, "proficiencia": 700.0},
        {"id_municipio": 2, "id_escola": 4, "proficiencia": 800.0}, {"id_municipio": 2, "id_escola": 4, "proficiencia": 800.0},
    ])
    d = contexto.decompor_variancia(alunos).set_index("componente")
    assert list(d.index) == ["entre_municipios", "entre_escolas", "intra_escola"]
    assert d["participacao_pct"].sum() == pytest.approx(100.0)
    assert d.loc["entre_municipios", "variancia"] == pytest.approx(0.0)
    assert d.loc["intra_escola", "variancia"] == pytest.approx(0.0)
    assert d.loc["entre_escolas", "participacao_pct"] == pytest.approx(100.0)
