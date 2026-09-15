"""Busca de hiperparâmetros em amostra por escola; o refit na base completa é o train.py com --params.

    python -m src.modeling.tune --regime producao --modelo hgb
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from src import config
from src.modeling.split import cv_por_grupo
from src.preprocessing.features import colunas_por_regime, separar_xy
from src.preprocessing.pipeline import build_pipeline

ESPACOS = {
    "hgb": {
        "clf__learning_rate": loguniform(0.02, 0.3),
        "clf__max_leaf_nodes": randint(15, 128),
        "clf__min_samples_leaf": randint(20, 500),
        "clf__l2_regularization": loguniform(1e-3, 10),
        "clf__max_depth": [None, 4, 6, 8, 12],
    },
    "logistica": {"clf__C": loguniform(1e-3, 10)},
}


def amostrar_por_escola(base: pd.DataFrame, n_alunos: int, seed: int = config.SEED) -> pd.DataFrame:
    if n_alunos >= len(base):
        return base
    # embaralha as escolas e vai empilhando até chegar perto de n_alunos -- de novo, escola
    # inteira ou nada, então o resultado passa um pouco de n_alunos (a busca de hiperparâmetro
    # não precisa ser no tamanho exato, só rápida o bastante)
    rng = np.random.default_rng(seed)
    tamanhos = base.groupby("id_escola").size()
    ordem = rng.permutation(tamanhos.index.to_numpy())
    acumulado = tamanhos.loc[ordem].cumsum()
    escolhidas = acumulado.index[acumulado <= n_alunos].tolist()
    if len(escolhidas) < len(ordem):
        escolhidas.append(ordem[len(escolhidas)])          # cruza o alvo com a próxima escola inteira
    return base[base["id_escola"].isin(escolhidas)].reset_index(drop=True)


def _nativo(v):
    # best_params_ vem com tipos numpy (np.float64 etc) que o json.dumps não serializa -- converte pro tipo python puro
    return v.item() if isinstance(v, np.generic) else v


def buscar(base: pd.DataFrame, regime: str, modelo: str = "hgb", n_amostra: int = 300_000,
           n_candidatos: int = 40, seed: int = config.SEED) -> tuple[dict, pd.DataFrame]:
    amostra = amostrar_por_escola(base, n_amostra, seed)
    X, y, grupos, _ = separar_xy(amostra, regime)
    num, cat = colunas_por_regime(amostra, regime)
    busca = HalvingRandomSearchCV(
        build_pipeline(modelo, num, cat, seed=seed), ESPACOS[modelo],
        # refit=False porque quem treina o modelo final na base inteira é o train.py com --params;
        # aqui só interessa achar os melhores hiperparâmetros, não guardar esse pipeline treinado na amostra
        n_candidates=n_candidatos, factor=3, cv=cv_por_grupo(5, seed), scoring="roc_auc",
        random_state=seed, n_jobs=-1, refit=False, verbose=1,
    )
    busca.fit(X, y, groups=grupos)
    melhores = {k: _nativo(v) for k, v in busca.best_params_.items()}
    return melhores, pd.DataFrame(busca.cv_results_)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["producao", "diagnostico"], default="producao")
    ap.add_argument("--modelo", choices=list(ESPACOS), default="hgb")
    ap.add_argument("--n-amostra", type=int, default=300_000)
    ap.add_argument("--n-candidatos", type=int, default=40)
    args = ap.parse_args(argv)

    base = pd.read_parquet(config.PROCESSED / "base_modelagem_aluno.parquet")
    melhores, res = buscar(base, args.regime, args.modelo, args.n_amostra, args.n_candidatos)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    nome = f"{args.regime}_{args.modelo}"
    (config.REPORTS / f"melhores_params_{nome}.json").write_text(json.dumps(melhores, indent=2), encoding="utf-8")
    cols = ["iter", "n_resources", "mean_test_score", "std_test_score", "rank_test_score"] + [c for c in res.columns if c.startswith("param_")]
    res[cols].sort_values(["iter", "rank_test_score"]).to_csv(config.REPORTS / f"busca_{nome}.csv", index=False)
    print(json.dumps(melhores, indent=2))
    print(f"melhor roc_auc (cv por escola, amostra): {res['mean_test_score'].max():.4f}")


if __name__ == "__main__":
    main()
