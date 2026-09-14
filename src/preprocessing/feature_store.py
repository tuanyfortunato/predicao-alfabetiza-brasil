"""Monta as bases de modelagem (feature store) a partir da Gold/Silver da Fase 2 e das externas.

    python -m src.preprocessing.feature_store            # grava as duas bases em data/processed
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.preprocessing.carregar import carregar_alunos, carregar_gold, carregar_metas
from src.preprocessing.contexto import (COLS_DISTRIBUICAO, contexto_escola_loo,
                                        contexto_municipal_defasado, meta_pactuada,
                                        participacao_escola_loo)
from src.preprocessing.externas import montar_contexto_externo

# a Silver e a Gold já trazem sigla_uf; tirar da externa evita sufixo _x/_y no merge
COLS_EXTERNAS_REPETIDAS = ["sigla_uf"]


REDES_PUBLICAS = ("municipal", "estadual")


def montar_base_aluno(ano_alvo: int = config.ANO_ALVO) -> pd.DataFrame:
    # a Gold e o perfil_escola são só rede pública; a Silver traz 24 alunos privados em 2024
    alunos = carregar_alunos(ano_alvo)
    alunos = alunos[alunos["rede_nome"].isin(REDES_PUBLICAS)].reset_index(drop=True)
    perfil = carregar_gold("perfil_escola")

    base = pd.concat([alunos, contexto_escola_loo(alunos), participacao_escola_loo(alunos, perfil)], axis=1)

    ctx = contexto_municipal_defasado(ano_alvo, carregar_gold("indicador_municipio"), carregar_gold("distribuicao_proficiencia"))
    base = base.merge(ctx, on="id_municipio", how="left")
    base["sem_historico"] = base["taxa_alfabetizacao_mun_t1"].isna()

    base = base.merge(meta_pactuada(ano_alvo, carregar_metas()), on="id_municipio", how="left")

    ext = montar_contexto_externo(ano_alvo).drop(columns=COLS_EXTERNAS_REPETIDAS)
    base = base.merge(ext, on="id_municipio", how="left")
    return base.reset_index(drop=True)


def _municipio_no_ano(ano: int) -> pd.DataFrame:
    filtro = "ano == @ano and nivel == 'municipio' and rede == 'municipal'"
    mvr = carregar_gold("meta_vs_resultado").query(filtro)
    evo = carregar_gold("evolucao_temporal").query(filtro)
    dist = carregar_gold("distribuicao_proficiencia").query(filtro)

    df = mvr[["id_municipio", "sigla_uf", "alunos_com_nota", "taxa_alfabetizacao", "ic95", "meta_ano", "gap", "situacao_meta"]]
    df = df.merge(evo[["id_municipio", "taxa_participacao", "proficiencia_media", "criancas_nao_alfabetizadas"]], on="id_municipio", how="left")
    df = df.merge(dist[["id_municipio"] + COLS_DISTRIBUICAO], on="id_municipio", how="left")
    # externa referenciada a ano+1 (o ano que este registro tenta prever, como meta_prox abaixo):
    # dá a mesma vintage que montar_base_aluno(ano+1) usaria e evita pedir defasagem além do que a fonte cobre
    # (bolsa_familia só tem 2023/2024 commitados; ano puro quebraria a linha de 2023 por faltar 2022)
    df = df.merge(montar_contexto_externo(ano + 1).drop(columns=COLS_EXTERNAS_REPETIDAS), on="id_municipio", how="left")
    df = df.merge(meta_pactuada(ano + 1, carregar_metas()).rename(columns={"meta_alvo": "meta_prox"}), on="id_municipio", how="left")
    df.insert(0, "ano", ano)
    return df


def montar_base_municipio(anos=(2023, 2024)) -> pd.DataFrame:
    base = pd.concat([_municipio_no_ano(a) for a in anos], ignore_index=True)
    prox = base[["ano", "id_municipio", "taxa_alfabetizacao", "situacao_meta"]].copy()
    prox["ano"] = prox["ano"] - 1
    prox = prox.rename(columns={"taxa_alfabetizacao": "taxa_prox", "situacao_meta": "situacao_meta_prox"})
    base = base.merge(prox, on=["ano", "id_municipio"], how="left")
    base["nao_atingiu_prox"] = np.where(base["situacao_meta_prox"].isna(), np.nan,
                                        (base["situacao_meta_prox"] == "nao_atingiu").astype(float))
    return base


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano", type=int, default=config.ANO_ALVO)
    args = ap.parse_args(argv)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    aluno = montar_base_aluno(args.ano)
    aluno.to_parquet(config.PROCESSED / "base_modelagem_aluno.parquet", index=False)
    print(f"base_modelagem_aluno     {aluno.shape[0]:>10,} x {aluno.shape[1]}  sem_historico={aluno['sem_historico'].mean():.1%}")

    mun = montar_base_municipio()
    mun.to_parquet(config.PROCESSED / "base_modelagem_municipio.parquet", index=False)
    print(f"base_modelagem_municipio {mun.shape[0]:>10,} x {mun.shape[1]}")


if __name__ == "__main__":
    main()
