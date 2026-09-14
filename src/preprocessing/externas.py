"""Contexto externo por município (IBGE, INEP, MDS, diretórios), com regra temporal por fonte.

Feature do ano t usa o último ano disponível <= t - defasagem. Censo 2022 e
diretórios são estruturais (sem ano). data/external tem linhas do ano alvo e
além: a regra é o que impede vazamento temporal.
"""
import numpy as np
import pandas as pd

from src.preprocessing.carregar import carregar_externa

ESTRUTURAIS = ["diretorios_municipio", "censo2022_municipio"]
DEFASAGEM = {
    "pib_municipio": 2,
    "populacao_municipio": 1,
    "ideb_municipio": 1,
    "indicadores_municipio": 1,
    "censo_escolar_municipio": 1,
    "bolsa_familia_municipio": 1,
}
NOMES = ESTRUTURAIS + list(DEFASAGEM)
SETORES = ["agropecuaria", "industria", "servicos", "adespss"]


def ano_referencia(fonte: str, ano: int, anos_disponiveis) -> int:
    limite = ano - DEFASAGEM[fonte]
    candidatos = [int(a) for a in pd.unique(pd.Series(anos_disponiveis)) if a <= limite]
    if not candidatos:
        raise ValueError(f"{fonte}: nenhum ano <= {limite} disponível")
    return max(candidatos)


def _carregar_todas() -> dict[str, pd.DataFrame]:
    return {n: carregar_externa(n) for n in NOMES}


def anos_referencia(ano: int, fontes: dict | None = None) -> dict[str, int]:
    fontes = fontes or _carregar_todas()
    return {f: ano_referencia(f, ano, fontes[f]["ano"]) for f in DEFASAGEM}


def _no_ano(fontes, nome, ano):
    df = fontes[nome]
    ref = ano_referencia(nome, ano, df["ano"])
    return df[df["ano"] == ref].drop(columns="ano").copy()


def montar_contexto_externo(ano: int, fontes: dict | None = None) -> pd.DataFrame:
    fontes = fontes or _carregar_todas()

    ctx = fontes["diretorios_municipio"].copy()

    c22 = fontes["censo2022_municipio"].rename(columns={"populacao": "pop_2022", "domicilios": "domicilios_2022", "area": "area_km2"})
    c22["densidade_demografica"] = c22["pop_2022"] / c22["area_km2"]
    c22["pct_pop_indigena"] = c22["populacao_indigena"] / c22["pop_2022"]
    c22["pct_pop_quilombola"] = c22["populacao_quilombola"] / c22["pop_2022"]
    ctx = ctx.merge(c22.drop(columns=["populacao_indigena", "populacao_quilombola"]), on="id_municipio", how="left")

    # PIB per capita usa a população do mesmo ano do PIB (t-2), não a de t-1
    pib = _no_ano(fontes, "pib_municipio", ano)
    ano_pib = ano_referencia("pib_municipio", ano, fontes["pib_municipio"]["ano"])
    pop_pib = fontes["populacao_municipio"].query("ano == @ano_pib")[["id_municipio", "populacao"]]
    pib = pib.merge(pop_pib, on="id_municipio", how="left")
    pib["pib_per_capita"] = pib["pib"] / pib["populacao"]
    pib["log_pib_per_capita"] = np.log1p(pib["pib_per_capita"])
    for s in SETORES:
        pib[f"pct_va_{s}"] = pib[f"va_{s}"] / pib["va"]
    ctx = ctx.merge(pib[["id_municipio", "pib_per_capita", "log_pib_per_capita"] + [f"pct_va_{s}" for s in SETORES]],
                    on="id_municipio", how="left")

    pop = _no_ano(fontes, "populacao_municipio", ano)
    pop["log_populacao"] = np.log1p(pop["populacao"])
    ctx = ctx.merge(pop, on="id_municipio", how="left")

    ctx = ctx.merge(_no_ano(fontes, "ideb_municipio", ano), on="id_municipio", how="left")
    ctx = ctx.merge(_no_ano(fontes, "indicadores_municipio", ano), on="id_municipio", how="left")

    ce = _no_ano(fontes, "censo_escolar_municipio", ano)
    ce["alunos_por_docente_ai"] = ce["matriculas_ai"] / ce["docentes_ai"]
    ctx = ctx.merge(ce, on="id_municipio", how="left")

    # famílias no Bolsa Família em dezembro de t-1 sobre os domicílios do Censo 2022;
    # as contagens brutas são volume (já coberto por log_populacao) e ficam de fora.
    # a fonte só tem 2023/2024 commitados (adicionada depois das demais, adendo 11): para anos em
    # que t-1 fica antes de 2023 ainda não há histórico — degrada para NaN em vez de quebrar a função.
    try:
        bf = _no_ano(fontes, "bolsa_familia_municipio", ano)[["id_municipio", "familias_bf"]]
        ctx = ctx.merge(bf, on="id_municipio", how="left")
        ctx["pct_familias_bolsa_familia"] = ctx["familias_bf"].astype("float64") / ctx["domicilios_2022"]
        ctx = ctx.drop(columns="familias_bf")
    except ValueError:
        print(f"aviso: bolsa_familia_municipio sem ano <= {ano - DEFASAGEM['bolsa_familia_municipio']} disponível — pct_familias_bolsa_familia fica NaN para ano={ano}")
        ctx["pct_familias_bolsa_familia"] = float("nan")

    # converter dtypes nullable de volta para numpy nativo
    # evita que pandas nullable Int64/Float64/boolean contaminem a saída
    for col in ctx.columns:
        dtype_str = str(ctx[col].dtype)
        if dtype_str == "Float64":
            ctx[col] = ctx[col].astype("float64")
        elif dtype_str == "Int64":
            # Int64 com NaN → float64; Int64 sem NaN → int64
            if ctx[col].isna().any():
                ctx[col] = ctx[col].astype("float64")
            else:
                ctx[col] = ctx[col].astype("int64")
        elif dtype_str == "boolean":
            # boolean → bool (mas mantém NaN como float se houver)
            if ctx[col].isna().any():
                ctx[col] = ctx[col].astype("float64")
            else:
                ctx[col] = ctx[col].astype("bool")

    assert ctx["id_municipio"].is_unique
    return ctx.reset_index(drop=True)
