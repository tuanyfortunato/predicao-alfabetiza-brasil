import numpy as np
import pandas as pd

from src import config
from src.preprocessing import feature_store


def test_base_aluno_tem_uma_linha_por_aluno_com_nota_e_flags(lake_minimo):
    base = feature_store.montar_base_aluno(2024)
    assert len(base) == 30                                   # 3 municípios × 2 escolas × 5 com nota; o privado fica fora
    assert "privada" not in set(base["rede_nome"])
    assert base["id_aluno"].is_unique
    assert base["sem_historico"].sum() == 10                 # município 3 não tem 2023
    assert base.loc[base["sem_historico"], "taxa_alfabetizacao_mun_t1"].isna().all()
    assert base.loc[~base["sem_historico"], "taxa_alfabetizacao_mun_t1"].isin([0.5, 0.7]).all()
    assert base["meta_alvo"].eq(0.6).all()
    assert base["regiao"].eq("Norte").all()
    assert np.allclose(base["taxa_participacao_escola_loo"], 4 / 5)
    assert base["n_alunos_escola"].eq(4).all()
    for c in ["taxa_escola_loo", "prof_media_escola_loo", "log_pib_per_capita", "tdi_ai", "pct_escolas_rurais",
              "pct_familias_bolsa_familia", "sigla_uf"]:
        assert c in base.columns
    assert not [c for c in base.columns if c.endswith("_x") or c.endswith("_y")]


def test_base_aluno_nao_carrega_gold_do_proprio_ano(lake_minimo):
    base = feature_store.montar_base_aluno(2024)
    assert "taxa_alfabetizacao" not in base.columns
    assert "pct_critico" not in base.columns                 # só a versão _mun_t1


def test_base_municipio_alvo_do_ano_seguinte(lake_minimo):
    b = feature_store.montar_base_municipio()
    assert sorted(b["ano"].unique()) == [2023, 2024]
    assert len(b) == 5 and not b.duplicated(["ano", "id_municipio"]).any()
    l23 = b[(b.ano == 2023) & (b.id_municipio == 1100015)].iloc[0]
    l24 = b[(b.ano == 2024) & (b.id_municipio == 1100015)].iloc[0]
    assert l23["taxa_prox"] == l24["taxa_alfabetizacao"]
    assert l23["meta_prox"] == 0.6 and l24["meta_prox"] == 0.65
    assert np.isnan(l24["taxa_prox"]) and pd.isna(l24["situacao_meta_prox"]) and np.isnan(l24["nao_atingiu_prox"])
    assert l23["nao_atingiu_prox"] in (0.0, 1.0)
    assert "pct_nivel_0" in b.columns and "log_pib_per_capita" in b.columns


def test_cli_grava_as_duas_bases(lake_minimo):
    feature_store.main(["--ano", "2024"])
    assert (config.PROCESSED / "base_modelagem_aluno.parquet").exists()
    assert (config.PROCESSED / "base_modelagem_municipio.parquet").exists()
