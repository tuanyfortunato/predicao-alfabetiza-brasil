import pandas as pd
import pytest

from scripts.baixar_dados import TABELAS_GOLD, copiar_lake


def _lake_fase2(raiz):
    for t in TABELAS_GOLD:
        (raiz / "gold" / t).mkdir(parents=True)
        pd.DataFrame({"ano": [2024], "x": [1.0]}).to_parquet(raiz / "gold" / t / "data.parquet")
    (raiz / "silver" / "metas").mkdir(parents=True)
    pd.DataFrame({"ano": [2024]}).to_parquet(raiz / "silver" / "metas" / "data.parquet")
    (raiz / "silver" / "resultados" / "municipio").mkdir(parents=True)
    pd.DataFrame({"ano": [2024]}).to_parquet(raiz / "silver" / "resultados" / "municipio" / "data.parquet")
    for ano in (2023, 2024):
        p = raiz / "silver" / "alunos" / f"ano={ano}"
        p.mkdir(parents=True)
        pd.DataFrame({"id_aluno": [1, 2]}).to_parquet(p / "parte-0.parquet")
        (p / "parte-0.parquet.crc").write_text("lixo")
    return raiz


def test_copia_gold_silver_e_alunos_particionado(tmp_path):
    origem = _lake_fase2(tmp_path / "fase2")
    destino = tmp_path / "novo"
    resumo = copiar_lake(origem, destino)

    assert sorted(p.name for p in (destino / "gold").glob("*.parquet")) == sorted(f"{t}.parquet" for t in TABELAS_GOLD)
    assert (destino / "silver" / "metas.parquet").exists()
    assert (destino / "silver" / "resultados_municipio.parquet").exists()
    assert (destino / "silver" / "alunos" / "ano=2023" / "parte-0.parquet").exists()
    assert not (destino / "silver" / "alunos" / "ano=2023" / "parte-0.parquet.crc").exists()
    assert resumo["alunos"] == 4
    assert resumo["indicador_municipio"] == 1


def test_recopia_sobrescreve_sem_duplicar_alunos(tmp_path):
    origem = _lake_fase2(tmp_path / "fase2")
    destino = tmp_path / "novo"
    copiar_lake(origem, destino)
    resumo = copiar_lake(origem, destino)
    assert resumo["alunos"] == 4


def test_falha_claro_se_origem_nao_tem_gold(tmp_path):
    with pytest.raises(FileNotFoundError, match="gold"):
        copiar_lake(tmp_path / "vazio", tmp_path / "novo")
