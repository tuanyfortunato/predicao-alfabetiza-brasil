import numpy as np
import pandas as pd
import pytest

from src.preprocessing import externas


def _fontes():
    ids = [1100015, 1100023]
    return {
        "diretorios_municipio": pd.DataFrame({"id_municipio": pd.array(ids, dtype="Int64"), "nome_municipio": ["A", "B"], "sigla_uf": ["RO", "RO"],
                                              "regiao": ["Norte", "Norte"], "capital_uf": pd.array([0, 1], dtype="Int64"), "amazonia_legal": pd.array([1, 1], dtype="Int64"),
                                              "latitude": [-11.9, -12.7], "longitude": [-61.9, -60.1]}),
        "censo2022_municipio": pd.DataFrame({"id_municipio": pd.array(ids, dtype="Int64"), "populacao": pd.array([20000, 80000], dtype="Int64"), "domicilios": pd.array([7000, 30000], dtype="Int64"),
                                             "area": pd.array([4000, 8000], dtype="Int64"), "taxa_alfabetizacao_adultos": [0.90, 0.95],
                                             "idade_mediana": [30.0, 32.0], "indice_envelhecimento": [40.0, 50.0],
                                             "razao_sexo": [100.0, 98.0], "populacao_indigena": pd.array([200, 0], dtype="Int64"),
                                             "populacao_quilombola": pd.array([0, 800], dtype="Int64")}),
        "pib_municipio": pd.DataFrame({"id_municipio": pd.array(ids * 3, dtype="Int64"), "ano": pd.array([2021, 2021, 2022, 2022, 2023, 2023], dtype="Int64"),
                                       "pib": pd.array([100, 400, 110, 440, 120, 480], dtype="Int64"), "impostos_liquidos": pd.array([10] * 6, dtype="Int64"), "va": pd.array([100, 400, 100, 400, 100, 400], dtype="Int64"),
                                       "va_agropecuaria": pd.array([50, 40, 50, 40, 50, 40], dtype="Int64"), "va_industria": pd.array([10, 100, 10, 100, 10, 100], dtype="Int64"),
                                       "va_servicos": pd.array([20, 200, 20, 200, 20, 200], dtype="Int64"), "va_adespss": pd.array([20, 60, 20, 60, 20, 60], dtype="Int64")}),
        "populacao_municipio": pd.DataFrame({"id_municipio": pd.array(ids * 3, dtype="Int64"), "ano": pd.array([2021, 2021, 2022, 2022, 2023, 2023], dtype="Int64"),
                                             "populacao": pd.array([20000, 80000, 20500, 80500, 21000, 81000], dtype="Int64")}),
        "ideb_municipio": pd.DataFrame({"id_municipio": pd.array([1100015, 1100015], dtype="Int64"), "ano": pd.array([2021, 2023], dtype="Int64"), "ideb_ai": [5.0, 5.5],
                                        "taxa_aprovacao_ideb_ai": [90.0, 95.0], "rendimento_ideb_ai": [0.9, 0.95],
                                        "nota_saeb_lp_ai": [200.0, 210.0], "nota_saeb_mat_ai": [205.0, 215.0], "projecao_ideb_ai": [5.2, 5.6]}),
        "indicadores_municipio": pd.DataFrame({"id_municipio": pd.array(ids * 2, dtype="Int64"), "ano": pd.array([2022, 2022, 2023, 2023], dtype="Int64"), "atu_ai": [20.0, 25.0, 21.0, 26.0],
                                               "had_ai": [4.5] * 4, "tdi_ai": [10.0, 5.0, 9.0, 4.0], "taxa_aprovacao_ai": [90.0] * 4,
                                               "taxa_reprovacao_ai": [5.0] * 4, "taxa_abandono_ai": [1.0] * 4, "dsu_ai": [80.0] * 4,
                                               "afd_ai_grupo1": [60.0] * 4, "ied_ai_nivel1": [10.0] * 4, "ird_baixa": [5.0] * 4}),
        "censo_escolar_municipio": pd.DataFrame({"ano": pd.array([2022, 2022, 2023, 2023], dtype="Int64"), "id_municipio": pd.array(ids * 2, dtype="Int64"), "n_escolas_ai": pd.array([10, 30, 11, 31], dtype="Int64"),
                                                 "pct_escolas_rurais": [0.5, 0.1] * 2, "pct_escolas_internet": [0.6, 0.9] * 2,
                                                 "pct_escolas_biblioteca": [0.3, 0.7] * 2, "pct_escolas_esgoto_rede": [0.2, 0.8] * 2,
                                                 "pct_escolas_agua_potavel": [0.9, 1.0] * 2, "pct_escolas_energia_rede": [1.0, 1.0] * 2,
                                                 "pct_escolas_lab_informatica": [0.2, 0.5] * 2, "pct_escolas_quadra": [0.3, 0.6] * 2,
                                                 "pct_escolas_alimentacao": [1.0, 1.0] * 2, "matriculas_ai": pd.array([1000, 4000, 1100, 4100], dtype="Int64"),
                                                 "docentes_ai": pd.array([50, 200, 55, 205], dtype="Int64"), "pct_matriculas_integral_ai": [0.1, 0.3] * 2,
                                                 "alunos_por_turma_ai": [22.0, 26.0] * 2}),
        "bolsa_familia_municipio": pd.DataFrame({"id_municipio": ids * 2, "ano": [2023, 2023, 2024, 2024],
                                                 "familias_bf": pd.array([1400, 3000, 1500, 3100], dtype="Int64"),
                                                 "pessoas_brc": pd.array([4000, 9000, 4200, 9300], dtype="Int64"),
                                                 "ben_primeira_infancia": pd.array([300, 500, 320, 520], dtype="Int64")}),
    }


def test_regra_temporal_por_fonte():
    f = _fontes()
    assert externas.ano_referencia("pib_municipio", 2024, f["pib_municipio"]["ano"]) == 2022
    assert externas.ano_referencia("ideb_municipio", 2024, f["ideb_municipio"]["ano"]) == 2023
    assert externas.ano_referencia("ideb_municipio", 2023, f["ideb_municipio"]["ano"]) == 2021
    assert externas.ano_referencia("indicadores_municipio", 2024, f["indicadores_municipio"]["ano"]) == 2023
    assert externas.ano_referencia("bolsa_familia_municipio", 2024, f["bolsa_familia_municipio"]["ano"]) == 2023
    with pytest.raises(ValueError):
        externas.ano_referencia("pib_municipio", 2020, f["pib_municipio"]["ano"])


def test_uma_linha_por_municipio_com_derivadas_e_faltantes():
    ctx = externas.montar_contexto_externo(2024, _fontes())
    assert len(ctx) == 2 and ctx["id_municipio"].is_unique
    a = ctx.set_index("id_municipio").loc[1100015]
    assert a["pib_per_capita"] == pytest.approx(110 / 20500)        # pib 2022 / população 2022
    assert a["log_pib_per_capita"] == pytest.approx(np.log1p(110 / 20500))
    assert a["densidade_demografica"] == pytest.approx(20000 / 4000)
    assert a["pct_va_agropecuaria"] == pytest.approx(0.5)
    assert a["pct_pop_indigena"] == pytest.approx(0.01)
    assert a["ideb_ai"] == 5.5 and a["tdi_ai"] == 9.0 and a["n_escolas_ai"] == 11
    assert a["alunos_por_docente_ai"] == pytest.approx(1100 / 55)
    assert a["log_populacao"] == pytest.approx(np.log1p(21000))     # população 2023
    assert a["pct_familias_bolsa_familia"] == pytest.approx(1400 / 7000)   # dez/2023 ÷ domicílios 2022
    b = ctx.set_index("id_municipio").loc[1100023]
    assert np.isnan(b["ideb_ai"])                                   # sem IDEB: NaN, linha permanece
    assert "ano" not in ctx.columns and "nome_municipio" in ctx.columns
    assert not {"familias_bf", "pessoas_brc", "ben_primeira_infancia"} & set(ctx.columns)
    assert str(ctx["pct_familias_bolsa_familia"].dtype) == "float64"


def test_anos_referencia_para_o_dicionario():
    assert externas.anos_referencia(2024, _fontes()) == {
        "pib_municipio": 2022, "populacao_municipio": 2023, "ideb_municipio": 2023,
        "indicadores_municipio": 2023, "censo_escolar_municipio": 2023, "bolsa_familia_municipio": 2023,
    }


def test_bolsa_familia_sem_historico_suficiente_vira_nan_sem_quebrar():
    # bolsa_familia só tem 2023/2024 commitados; para ano=2023 (defasagem 1) precisaria de <= 2022,
    # que não existe ainda - a função deve degradar só essa coluna, não propagar o ValueError
    f = _fontes()
    ctx = externas.montar_contexto_externo(2023, f)
    assert len(ctx) == 2 and ctx["id_municipio"].is_unique
    assert ctx["pct_familias_bolsa_familia"].isna().all()
    assert ctx["ideb_ai"].notna().any()                      # demais fontes seguem populadas normalmente
    assert not ctx["pib_per_capita"].isna().all()
