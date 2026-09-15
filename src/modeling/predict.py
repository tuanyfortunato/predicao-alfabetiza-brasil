"""Pontua uma base nova com o pipeline salvo.

    python -m src.modeling.predict --regime producao --modelo hgb --so-teste
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config

ID_COLS = ["id_aluno", "id_escola", "id_municipio", "sigla_uf"]
CASAS = 4


def carregar_modelo(caminho: Path | None = None, regime: str = "producao", modelo: str = "hgb"):
    return joblib.load(caminho or config.MODELS / f"modelo_aluno_{regime}_{modelo}.joblib")


def pontuar(pipeline, base: pd.DataFrame, limiar: float = 0.5) -> pd.DataFrame:
    # feature_names_in_ garante que pega as colunas na mesma ordem/conjunto que o pipeline
    # foi treinado, mesmo que "base" tenha colunas extras (ids, alvo etc)
    X = base[list(pipeline.feature_names_in_)]
    proba = pipeline.predict_proba(X)[:, 1].round(CASAS)
    out = base[ID_COLS].reset_index(drop=True).copy()
    out["prob_alfabetizado"] = proba
    out["risco_nao_alf"] = (1 - proba).round(CASAS)
    # limiar vem do train.py (escolhido pra bater o recall mínimo), não é o 0.5 padrão do sklearn
    out["classe_prevista"] = np.where(proba >= limiar, "alfabetizado", "nao_alfabetizado")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="producao")
    ap.add_argument("--modelo", default="hgb")
    ap.add_argument("--caminho-modelo", type=Path)
    ap.add_argument("--entrada", type=Path, default=None)
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--limiar", type=float, default=None)
    ap.add_argument("--so-teste", action="store_true", help="pontua só a partição de teste gravada pelo train.py")
    args = ap.parse_args(argv)

    base = pd.read_parquet(args.entrada or config.PROCESSED / "base_modelagem_aluno.parquet")
    if args.so_teste:
        # usa a partição salva pelo train.py em vez de dividir de novo -- assim pontua exatamente
        # os mesmos alunos que ficaram de fora do treino daquele modelo
        part = pd.read_parquet(config.MODELS / f"particao_{args.regime}.parquet")
        base = base[base["id_aluno"].isin(part.loc[part["parte"] == "teste", "id_aluno"])]
    limiar = args.limiar
    if limiar is None:
        # sem --limiar explícito, usa o mesmo limiar que o train.py calculou e salvou nas métricas
        limiar = json.loads((config.REPORTS / f"metricas_{args.regime}_{args.modelo}.json").read_text())["limiar"]

    pred = pontuar(carregar_modelo(args.caminho_modelo, args.regime, args.modelo), base, limiar)
    saida = args.saida or config.REPORTS / f"predicoes_{args.regime}_{args.modelo}.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(saida, index=False)
    print(f"{len(pred):,} alunos pontuados (limiar {limiar:.3f}); "
          f"{(pred['classe_prevista'] == 'nao_alfabetizado').mean():.1%} previstos não alfabetizados -> {saida}")


if __name__ == "__main__":
    main()
