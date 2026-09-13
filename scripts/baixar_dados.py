"""Copia a Gold e a Silver do pipeline da Fase 2 para data/.

Uso:
    python -m scripts.baixar_dados                    # usa FASE2_LAKE_PATH do .env
    python -m scripts.baixar_dados --origem D:/lake    # pasta data/ do pipeline
"""
import argparse
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv

TABELAS_GOLD = [
    "indicador_municipio", "meta_vs_resultado", "evolucao_temporal",
    "perfil_escola", "distribuicao_proficiencia",
]
TABELAS_SILVER = {
    "metas": Path("silver/metas/data.parquet"),
    "resultados_municipio": Path("silver/resultados/municipio/data.parquet"),
}


def _linhas(caminho: Path) -> int:
    return pq.read_metadata(caminho).num_rows


def copiar_lake(origem: Path, destino: Path) -> dict[str, int]:
    origem, destino = Path(origem), Path(destino)
    if not (origem / "gold").is_dir():
        raise FileNotFoundError(f"não achei a pasta gold em {origem}")

    resumo = {}
    (destino / "gold").mkdir(parents=True, exist_ok=True)
    for t in TABELAS_GOLD:
        alvo = destino / "gold" / f"{t}.parquet"
        shutil.copy(origem / "gold" / t / "data.parquet", alvo)
        resumo[t] = _linhas(alvo)

    (destino / "silver").mkdir(exist_ok=True)
    for nome, rel in TABELAS_SILVER.items():
        alvo = destino / "silver" / f"{nome}.parquet"
        shutil.copy(origem / rel, alvo)
        resumo[nome] = _linhas(alvo)

    alunos_dst = destino / "silver" / "alunos"
    if alunos_dst.exists():
        shutil.rmtree(alunos_dst)
    shutil.copytree(origem / "silver" / "alunos", alunos_dst,
                    ignore=shutil.ignore_patterns("*.crc", "_*"))
    resumo["alunos"] = sum(_linhas(p) for p in alunos_dst.rglob("*.parquet"))
    return resumo


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default=os.environ.get("FASE2_LAKE_PATH", "../pipeline-dados-alfabetiza-brasil/data"))
    ap.add_argument("--destino", default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()
    for nome, n in copiar_lake(args.origem, args.destino).items():
        print(f"{nome:<28}{n:>12,} linhas")


if __name__ == "__main__":
    main()
