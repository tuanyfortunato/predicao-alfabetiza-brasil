"""Contexto que o aluno recebe sem vazar o alvo: município em t-1 e escola com leave-one-out."""
import numpy as np
import pandas as pd

COLS_INDICADOR = [
    "taxa_alfabetizacao", "ic95", "taxa_participacao", "proficiencia_media", "alunos_avaliados",
    "criancas_nao_alfabetizadas", "taxa_limite_inferior", "taxa_limite_superior", "alerta_participacao",
]
COLS_DISTRIBUICAO = [f"pct_nivel_{i}" for i in range(9)] + ["pct_critico", "pct_atencao", "pct_quase_la"]


def contexto_municipal_defasado(ano_alvo: int, indicador: pd.DataFrame, distribuicao: pd.DataFrame,
                                rede: str = "total") -> pd.DataFrame:
    t1 = ano_alvo - 1
    ind = indicador.loc[indicador["ano"] == t1, ["id_municipio"] + COLS_INDICADOR]
    dist = distribuicao.loc[
        (distribuicao["ano"] == t1) & (distribuicao["nivel"] == "municipio") & (distribuicao["rede"] == rede),
        ["id_municipio"] + COLS_DISTRIBUICAO,
    ]
    ctx = ind.merge(dist, on="id_municipio", how="left")
    ctx["alerta_participacao"] = ctx["alerta_participacao"].astype(float)
    ctx = ctx.rename(columns={c: f"{c}_mun_t1" for c in ctx.columns if c != "id_municipio"})
    assert ctx["id_municipio"].is_unique
    return ctx.reset_index(drop=True)


def meta_pactuada(ano_alvo: int, metas: pd.DataFrame) -> pd.DataFrame:
    col = f"meta_alfabetizacao_{ano_alvo}"
    m = metas[(metas["nivel"] == "municipio") & (metas["rede_padronizada"] == "municipal")]
    # a meta é pactuada uma vez; qualquer linha do município serve, a mais recente por garantia
    m = m.sort_values("ano").groupby("id_municipio", as_index=False)[col].last()
    m["id_municipio"] = m["id_municipio"].astype(int)
    return m.rename(columns={col: "meta_alvo"})


def contexto_escola_loo(alunos: pd.DataFrame) -> pd.DataFrame:
    g = alunos.groupby("id_escola")
    n = g["proficiencia"].transform("count")
    soma_prof = g["proficiencia"].transform("sum")
    soma_alf = g["alfabetizado"].transform("sum")
    n_outros = n - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "taxa_escola_loo": (soma_alf - alunos["alfabetizado"]) / n_outros,
            "prof_media_escola_loo": (soma_prof - alunos["proficiencia"]) / n_outros,
            "n_alunos_escola": n_outros,
        }, index=alunos.index)
    out.loc[n_outros == 0, ["taxa_escola_loo", "prof_media_escola_loo"]] = np.nan
    return out


def participacao_escola_loo(alunos: pd.DataFrame, perfil_escola: pd.DataFrame) -> pd.Series:
    ano = int(alunos["ano"].iloc[0])
    p = perfil_escola.loc[perfil_escola["ano"] == ano].set_index("id_escola")
    aval = alunos["id_escola"].map(p["alunos_avaliados"]) - 1
    pres = alunos["id_escola"].map(p["alunos_presentes"]) - 1
    return (pres / aval.replace(0, np.nan)).rename("taxa_participacao_escola_loo")


def decompor_variancia(alunos: pd.DataFrame) -> pd.DataFrame:
    """Mesmo cálculo do laboratório da Fase 2: média simples, sem peso amostral."""
    media_mun = alunos.groupby("id_municipio")["proficiencia"].transform("mean")
    media_esc = alunos.groupby("id_escola")["proficiencia"].transform("mean")
    variancia = pd.Series({
        "entre_municipios": media_mun.var(),
        "entre_escolas": (media_esc - media_mun).var(),
        "intra_escola": (alunos["proficiencia"] - media_esc).var(),
    })
    out = pd.DataFrame({"componente": variancia.index, "variancia": variancia.to_numpy()})
    out["participacao_pct"] = 100 * out["variancia"] / out["variancia"].sum()
    return out
