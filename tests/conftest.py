"""Fixtures compartilhadas. Nenhum teste lê data/ de verdade."""
import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture
def lake_tmp(tmp_path, monkeypatch):
    """Redireciona todos os caminhos de dados para uma pasta temporária."""
    for nome in ["DATA", "GOLD", "SILVER", "EXTERNAL", "PROCESSED", "MODELS", "REPORTS", "IMAGES"]:
        destino = tmp_path / nome.lower()
        destino.mkdir()
        monkeypatch.setattr(config, nome, destino)
    return tmp_path


def montar_alunos(linhas: list[dict]) -> pd.DataFrame:
    """Linhas com as colunas da Silver de alunos; ausentes vêm sem proficiencia."""
    base = {
        "ano": 2024, "id_municipio": 3550308, "id_escola": 60000001, "sigla_uf": "SP",
        "rede_nome": "municipal", "caderno": "1", "serie": 2, "rede": 3,
        "presenca": 1, "preenchimento_caderno": 1, "peso_aluno": 1.0,
    }
    df = pd.DataFrame([{**base, **l} for l in linhas])
    df["id_aluno"] = range(1, len(df) + 1)
    df["presente"] = df["presenca"] == 1
    df["sem_nota"] = df["proficiencia"].isna()
    df["alfabetizado"] = (df["proficiencia"] >= config.CORTE_ALFABETIZACAO).fillna(False).astype(int)
    return df


def _gravar(df: pd.DataFrame, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(caminho, index=False)


@pytest.fixture
def lake_minimo(lake_tmp):
    """Gold, Silver e externas com 3 municípios (1 e 2 com 2023; 3 só em 2024), 2 escolas por município."""
    from tests.test_externas import _fontes
    anos = [2023, 2024]
    muns = [1100015, 1100023, 1100031]

    def linha_ind(ano, m, taxa):
        return {"ano": ano, "id_municipio": m, "sigla_uf": "RO", "alunos_avaliados": 40, "alunos_presentes": 36,
                "alunos_com_nota": 36, "taxa_participacao": 0.9, "taxa_alfabetizacao": taxa, "ic95": 0.08,
                "taxa_limite_inferior": taxa - 0.08, "taxa_limite_superior": taxa + 0.08, "proficiencia_media": 700 + 100 * taxa,
                "criancas_nao_alfabetizadas": 36 * (1 - taxa), "alerta_participacao": False}
    ind = [linha_ind(2023, m, t) for m, t in zip(muns[:2], [0.5, 0.7])] + [linha_ind(2024, m, t) for m, t in zip(muns, [0.55, 0.72, 0.4])]
    _gravar(pd.DataFrame(ind), config.GOLD / "indicador_municipio.parquet")

    dist, mvr, evo = [], [], []
    for ano in anos:
        for m in (muns if ano == 2024 else muns[:2]):
            for rede in ["total", "municipal"]:
                dist.append({"ano": ano, "nivel": "municipio", "rede": rede, "sigla_uf": "RO", "id_municipio": m, "alunos_com_nota": 36,
                             **{f"pct_nivel_{i}": 1 / 9 for i in range(9)}, "pct_critico": 0.2, "pct_atencao": 0.2,
                             "pct_alfabetizado": 0.6, "pct_quase_la": 0.1})
            taxa = 0.5 + 0.1 * (ano - 2023) + 0.05 * muns.index(m)
            mvr.append({"ano": ano, "nivel": "municipio", "rede": "municipal", "sigla_uf": "RO", "id_municipio": m, "alunos_com_nota": 36,
                        "taxa_alfabetizacao": taxa, "ic95": 0.08, "meta_ano": 0.6 if ano == 2024 else None, "gap": taxa - 0.6,
                        "atingiu_meta": taxa >= 0.6, "situacao_meta": "sem_meta" if ano == 2023 else ("atingiu" if taxa >= 0.6 else "nao_atingiu")})
            evo.append({"ano": ano, "nivel": "municipio", "rede": "municipal", "sigla_uf": "RO", "id_municipio": m, "alunos_avaliados": 40,
                        "alunos_presentes": 36, "alunos_com_nota": 36, "taxa_participacao": 0.9, "taxa_alfabetizacao": taxa, "ic95": 0.08,
                        "proficiencia_media": 750.0, "criancas_nao_alfabetizadas": 36 * (1 - taxa)})
    _gravar(pd.DataFrame(dist), config.GOLD / "distribuicao_proficiencia.parquet")
    _gravar(pd.DataFrame(mvr), config.GOLD / "meta_vs_resultado.parquet")
    _gravar(pd.DataFrame(evo), config.GOLD / "evolucao_temporal.parquet")

    escolas = {m: [m * 10 + 1, m * 10 + 2] for m in muns}
    perfil = [{"ano": ano, "id_escola": e, "id_municipio": m, "sigla_uf": "RO", "rede": "municipal", "alunos_avaliados": 6,
               "alunos_presentes": 5, "alunos_com_nota": 5, "taxa_participacao": 5 / 6, "taxa_alfabetizacao": 0.6, "ic95": 0.2,
               "proficiencia_media": 750.0, "taxa_municipio": 0.6, "residuo": 0.0}
              for ano in anos for m in (muns if ano == 2024 else muns[:2]) for e in escolas[m]]
    _gravar(pd.DataFrame(perfil), config.GOLD / "perfil_escola.parquet")

    metas = [{"ano": 2024, "nivel": "municipio", "rede_padronizada": "municipal", "id_municipio": m, "sigla_uf": None,
              "meta_alfabetizacao_2024": 0.6, "meta_alfabetizacao_2025": 0.65, "meta_alfabetizacao_2030": 0.8} for m in muns]
    _gravar(pd.DataFrame(metas), config.SILVER / "metas.parquet")

    rng = np.random.default_rng(0)
    for ano in anos:
        linhas = []
        for m in (muns if ano == 2024 else muns[:2]):
            for e in escolas[m]:
                for _ in range(5):
                    linhas.append({"id_municipio": m, "id_escola": e, "sigla_uf": "RO", "proficiencia": float(rng.normal(750, 40))})
                linhas.append({"id_municipio": m, "id_escola": e, "sigla_uf": "RO", "proficiencia": None,
                               "presenca": 0, "preenchimento_caderno": 0, "peso_aluno": None})
        if ano == 2024:   # a Silver tem rede privada (24 alunos em 2024), a Gold não: precisa ficar fora da base
            linhas.append({"id_municipio": muns[0], "id_escola": 99, "sigla_uf": "RO", "rede_nome": "privada", "rede": 1, "proficiencia": 760.0})
        _gravar(montar_alunos(linhas).drop(columns="ano"), config.SILVER / "alunos" / f"ano={ano}" / "parte.parquet")

    for nome, df in _fontes().items():
        extra = df[df["id_municipio"] == 1100015].assign(id_municipio=1100031)   # terceiro município copia o primeiro
        _gravar(pd.concat([df, extra], ignore_index=True), config.EXTERNAL / f"{nome}.parquet")
    return lake_tmp
