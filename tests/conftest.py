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


def montar_base_sintetica(n_escolas: int = 30, alunos_por_escola: int = 20, seed: int = 0) -> pd.DataFrame:
    """Imita base_modelagem_aluno: contexto com sinal de verdade, colunas proibidas presentes, NaN nas externas."""
    rng = np.random.default_rng(seed)
    muns = [1100015, 1100023, 1100031, 2900108, 3550308]
    ufs = {1100015: "RO", 1100023: "RO", 1100031: "RO", 2900108: "BA", 3550308: "SP"}
    regioes = {"RO": "Norte", "BA": "Nordeste", "SP": "Sudeste"}
    taxa_mun = {m: t for m, t in zip(muns, [0.35, 0.45, np.nan, 0.55, 0.75])}
    linhas = []
    for e in range(n_escolas):
        m = muns[e % len(muns)]
        efeito_escola = rng.normal(0, 0.8)
        for _ in range(alunos_por_escola):
            base_logit = 3 * ((taxa_mun[m] if not np.isnan(taxa_mun[m]) else 0.5) - 0.5) + efeito_escola
            p = 1 / (1 + np.exp(-(base_logit + rng.normal(0, 1))))
            alf = int(rng.random() < p)
            linhas.append({
                "id_municipio": m, "id_escola": 60000 + e, "sigla_uf": ufs[m], "regiao": regioes[ufs[m]],
                "rede_nome": "municipal" if e % 4 else "estadual", "rede": 3 if e % 4 else 2,
                "taxa_alfabetizacao_mun_t1": taxa_mun[m], "ic95_mun_t1": 0.05, "sem_historico": np.isnan(taxa_mun[m]),
                "meta_alvo": 0.6, "log_pib_per_capita": rng.normal(3, 0.5), "tdi_ai": rng.normal(8, 3) if rng.random() > 0.1 else np.nan,
                "pct_escolas_rurais": rng.random(), "matriculas_ai": int(rng.integers(100, 5000)),
                "taxa_escola_loo": np.nan, "prof_media_escola_loo": np.nan, "n_alunos_escola": alunos_por_escola - 1,
                "taxa_participacao_escola_loo": 0.9,
                "alfabetizado": alf, "proficiencia": 743 + (1 if alf else -1) * abs(rng.normal(30, 20)),
                "peso_aluno": float(rng.uniform(0.5, 2.0)), "presente": True, "sem_nota": False, "presenca": 1,
                "preenchimento_caderno": 1, "caderno": "1", "serie": 2, "ano": 2024, "presenca_nome": "presente",
                "_row_hash": 0, "_ingestion_ts": "", "_source": "", "nome_municipio": "x",
            })
    df = pd.DataFrame(linhas)
    df["id_aluno"] = range(1, len(df) + 1)
    df["matriculas_ai"] = df["matriculas_ai"].astype("Int64")   # dtype nullable como vem do parquet
    g = df.groupby("id_escola")
    n = g["alfabetizado"].transform("count")
    df["taxa_escola_loo"] = (g["alfabetizado"].transform("sum") - df["alfabetizado"]) / (n - 1)
    df["prof_media_escola_loo"] = (g["proficiencia"].transform("sum") - df["proficiencia"]) / (n - 1)
    return df


def montar_base_municipio_sintetica(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Imita base_modelagem_municipio: 2023 com alvo de 2024, 2024 sem alvo; taxa_prox depende da taxa e do PIB."""
    rng = np.random.default_rng(seed)
    ids = 1100000 + np.arange(n)
    ufs = rng.choice(["RO", "BA", "SP"], n)
    regioes = {"RO": "Norte", "BA": "Nordeste", "SP": "Sudeste"}
    linhas = []
    for ano in (2023, 2024):
        taxa = np.clip(rng.normal(0.55, 0.15, n), 0.05, 0.98)
        pib = rng.normal(3, 0.5, n)
        for i in range(n):
            linhas.append({
                "ano": ano, "id_municipio": ids[i], "nome_municipio": f"m{i}", "sigla_uf": ufs[i], "regiao": regioes[ufs[i]],
                "alunos_com_nota": 200, "taxa_alfabetizacao": taxa[i], "ic95": 0.06, "meta_ano": np.nan if ano == 2023 else 0.6,
                "gap": np.nan, "atingiu_meta": None, "situacao_meta": "sem_meta" if ano == 2023 else "atingiu",
                "taxa_participacao": 0.9, "proficiencia_media": 700 + 100 * taxa[i], "criancas_nao_alfabetizadas": 200 * (1 - taxa[i]),
                **{f"pct_nivel_{k}": 1 / 9 for k in range(9)}, "pct_critico": 0.2, "pct_atencao": 0.2, "pct_quase_la": 0.1,
                "log_pib_per_capita": pib[i], "tdi_ai": rng.normal(8, 3) if rng.random() > 0.1 else np.nan,
                "taxa_alfabetizacao_adultos": 0.90, "pct_escolas_rurais": rng.random(), "log_populacao": rng.normal(9, 1),
                "densidade_demografica": rng.lognormal(3, 1), "meta_prox": 0.6 if ano == 2023 else 0.65,
                "pct_familias_bolsa_familia": np.nan if ano == 2023 else rng.random(),
                "pct_va_servicos": rng.random() if ano == 2023 else np.nan,
            })
    df = pd.DataFrame(linhas)
    e23 = df["ano"] == 2023
    prox = np.clip(df.loc[e23, "taxa_alfabetizacao"] * 0.8 + 0.05 * df.loc[e23, "log_pib_per_capita"] + rng.normal(0, 0.05, e23.sum()), 0, 1)
    df.loc[e23, "taxa_prox"] = prox.to_numpy()
    df.loc[e23, "situacao_meta_prox"] = np.where(prox < 0.6 - 0.06, "nao_atingiu", np.where(prox < 0.6 + 0.06, "indistinguivel", "atingiu"))
    df.loc[e23 & (df["id_municipio"] < 1100000 + 10), "situacao_meta_prox"] = "sem_meta"
    df.loc[df["situacao_meta_prox"] == "sem_meta", "nao_atingiu_prox"] = np.nan
    ok = e23 & (df["situacao_meta_prox"] != "sem_meta")
    df.loc[ok, "nao_atingiu_prox"] = (df.loc[ok, "situacao_meta_prox"] == "nao_atingiu").astype(float)
    # replica o caso real: ~100 municípios sem meta pactuada para o ano seguinte na aplicação de 2024
    # (aqui, os 10 primeiros ids) — devem continuar no ranking com gap/acima_da_margem não computáveis.
    sem_meta_2025 = (df["ano"] == 2024) & (df["id_municipio"] < 1100000 + 10)
    df.loc[sem_meta_2025, "meta_prox"] = np.nan
    return df
