"""Bolsa Família (dados.gov.br / MDS) agregado por município: dezembro de cada ano.

Os CSVs anuais são baixados manualmente do catálogo "Bolsa Família - Benefícios
Básicos e Variáveis" (um arquivo por ano, todos os meses) e ficam fora do repo.
O parquet resultante (poucos kB) é versionado em data/external.

Uso:
    python -m scripts.baixar_bolsa_familia                       # anos 2023 e 2024, origem do .env
    python -m scripts.baixar_bolsa_familia --anos 2023 2024 2025 --origem D:/mds
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from scripts.extrair_externas import registrar_metadados, validar

ANOS_PADRAO = (2023, 2024)
ARQUIVO = "aplicacoes.mds.gov.br.{ano}.txt"
COLUNAS = {
    "qtd_ben_bf": "familias_bf",
    "qtd_ben_brc": "pessoas_brc",
    "qtd_ben_bpi": "ben_primeira_infancia",
}
MINIMO_MUNICIPIOS = 5500


def mapa_codigos(diretorios: pd.DataFrame) -> dict[int, int]:
    # o 7º dígito do código IBGE é verificador; os dados do MDS vêm sem ele
    ids = diretorios["id_municipio"].astype(int)
    return dict(zip(ids // 10, ids))


def processar(bruto: pd.DataFrame, ano: int, mapa: dict[int, int]) -> pd.DataFrame:
    dez = bruto[bruto["anomes_s"] == ano * 100 + 12]
    if dez.empty:
        raise ValueError(f"não há dezembro de {ano} no arquivo")
    out = pd.DataFrame({
        "id_municipio": dez["codigo_ibge"].map(mapa),
        "ano": ano,
        **{novo: dez[antigo].values for antigo, novo in COLUNAS.items()},
    })
    sem_match = out["id_municipio"].isna()
    if sem_match.any():
        print(f"  aviso: {int(sem_match.sum())} código(s) sem correspondência de 7 dígitos descartado(s): "
              f"{sorted(dez.loc[sem_match.values, 'codigo_ibge'].tolist())}")
    out = out[~sem_match]
    sem_valor = out[list(COLUNAS.values())].isna().all(axis=1)
    if sem_valor.any():
        print(f"  aviso: {int(sem_valor.sum())} município(s) sem nenhum valor descartado(s): "
              f"{sorted(out.loc[sem_valor, 'id_municipio'].astype(int).tolist())}")
    out = out[~sem_valor]
    out = out.astype({"id_municipio": "int64", "ano": "int64", **{c: "int64" for c in COLUNAS.values()}})
    return out.reset_index(drop=True)


def main(argv=None) -> None:
    load_dotenv()
    raiz = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default=os.environ.get("BOLSA_FAMILIA_PATH", "."))
    ap.add_argument("--anos", nargs="+", type=int, default=list(ANOS_PADRAO))
    ap.add_argument("--destino", default=raiz / "data" / "external")
    args = ap.parse_args(argv)

    destino = Path(args.destino)
    mapa = mapa_codigos(pd.read_parquet(destino / "diretorios_municipio.parquet"))
    partes = []
    for ano in args.anos:
        bruto = pd.read_csv(Path(args.origem) / ARQUIVO.format(ano=ano))
        parte = processar(bruto, ano, mapa)
        if len(parte) < MINIMO_MUNICIPIOS:
            raise ValueError(f"{ano}: só {len(parte)} municípios em dezembro, esperava >= {MINIMO_MUNICIPIOS}")
        partes.append(parte)
        print(f"{ano}: {len(parte):,} municípios")
    df = pd.concat(partes, ignore_index=True)
    validar(df, ["id_municipio", "ano"])
    df.to_parquet(destino / "bolsa_familia_municipio.parquet", index=False)
    registrar_metadados(destino, "bolsa_familia_municipio", {
        "linhas": int(len(df)), "anos": args.anos, "mes": 12,
        "fonte": "dados.gov.br - MDS/SAGICAD, Bolsa Família - Benefícios Básicos e Variáveis (download manual)",
    })


if __name__ == "__main__":
    main()
