"""Treina, avalia e grava o modelo de aluno de ponta a ponta.

    python -m src.modeling.train --regime producao --modelo hgb
    python -m src.modeling.train --regime diagnostico --modelo hgb --params reports/melhores_params_diagnostico_hgb.json
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score

from src import config
from src.evaluation.metrics import calcular_metricas, escolher_limiar, metricas_por_recorte, tabela_calibracao
from src.modeling.split import dividir_por_escola
from src.preprocessing.features import colunas_por_regime, separar_xy
from src.preprocessing.pipeline import build_pipeline


def treinar(base: pd.DataFrame, regime: str, modelo: str = "hgb", seed: int = config.SEED,
            params: dict | None = None, recall_minimo: float = 0.8, cv_municipio: bool = False) -> dict:
    X, y, grupos, pesos = separar_xy(base, regime)
    num, cat = colunas_por_regime(base, regime)
    partes = dividir_por_escola(grupos, seed=seed)
    tr, va, te = partes["treino"], partes["validacao"], partes["teste"]

    pipe = build_pipeline(modelo, num, cat, seed=seed, params=params)
    pipe.fit(X.iloc[tr], y.iloc[tr])

    proba_va = pipe.predict_proba(X.iloc[va])[:, 1]
    limiar = escolher_limiar(y.iloc[va], proba_va, recall_minimo)

    proba_te = pipe.predict_proba(X.iloc[te])[:, 1]
    y_te, p_te = y.iloc[te].to_numpy(), pesos.iloc[te].to_numpy()

    cv = None
    if cv_municipio:
        scores = cross_val_score(build_pipeline(modelo, num, cat, seed=seed, params=params), X.iloc[tr], y.iloc[tr],
                                 groups=base["id_municipio"].iloc[tr], cv=GroupKFold(n_splits=5), scoring="roc_auc")
        cv = {"roc_auc_media": float(scores.mean()), "roc_auc_dp": float(scores.std())}

    metricas = {
        "regime": regime, "modelo": modelo, "seed": seed, "params": params or {},
        "n_treino": int(len(tr)), "n_validacao": int(len(va)), "n_teste": int(len(te)),
        "n_escolas_teste": int(grupos.iloc[te].nunique()), "limiar": limiar,
        "validacao": calcular_metricas(y.iloc[va], proba_va, limiar),
        "teste": calcular_metricas(y_te, proba_te, limiar),
        "teste_ponderado": calcular_metricas(y_te, proba_te, limiar, pesos=p_te),
        "por_uf": metricas_por_recorte(base["sigla_uf"].iloc[te], y_te, proba_te, limiar).to_dict("records"),
        "por_rede": metricas_por_recorte(base["rede_nome"].iloc[te], y_te, proba_te, limiar).to_dict("records"),
        "calibracao": tabela_calibracao(y_te, proba_te).assign(faixa=lambda d: d["faixa"].astype(str)).to_dict("records"),
        "cv_municipio": cv,
    }
    return {"pipeline": pipe, "partes": partes, "colunas": {"numericas": num, "categoricas": cat}, "metricas": metricas}


def salvar(resultado: dict, base: pd.DataFrame) -> dict[str, Path]:
    m = resultado["metricas"]
    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    nome = f"{m['regime']}_{m['modelo']}"

    caminhos = {
        "modelo": config.MODELS / f"modelo_aluno_{nome}.joblib",
        "particao": config.MODELS / f"particao_{m['regime']}.parquet",
        "metricas": config.REPORTS / f"metricas_{nome}.json",
    }
    joblib.dump(resultado["pipeline"], caminhos["modelo"])
    partes = pd.concat([pd.DataFrame({"id_aluno": base["id_aluno"].iloc[pos].to_numpy(), "parte": nome_parte})
                        for nome_parte, pos in resultado["partes"].items()], ignore_index=True)
    partes.to_parquet(caminhos["particao"], index=False)
    caminhos["metricas"].write_text(json.dumps({**m, "colunas": resultado["colunas"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    return caminhos


def _amostrar_escolas(base: pd.DataFrame, n_escolas: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    escolhidas = rng.choice(base["id_escola"].unique(), size=n_escolas, replace=False)
    return base[base["id_escola"].isin(escolhidas)].reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["producao", "diagnostico"], default="producao")
    ap.add_argument("--modelo", choices=["dummy", "logistica", "hgb"], default="hgb")
    ap.add_argument("--params", type=Path, help="json com hiperparâmetros (saída de tune.py)")
    ap.add_argument("--amostra-escolas", type=int, help="treina numa amostra de escolas (execuções rápidas)")
    ap.add_argument("--recall-minimo", type=float, default=0.8)
    ap.add_argument("--cv-municipio", action="store_true")
    args = ap.parse_args(argv)

    base = pd.read_parquet(config.PROCESSED / "base_modelagem_aluno.parquet")
    if args.amostra_escolas:
        base = _amostrar_escolas(base, args.amostra_escolas, config.SEED)
    params = json.loads(args.params.read_text()) if args.params else None

    r = treinar(base, args.regime, args.modelo, params=params, recall_minimo=args.recall_minimo, cv_municipio=args.cv_municipio)
    caminhos = salvar(r, base)
    t = r["metricas"]["teste"]
    print(f"{args.regime}/{args.modelo}: roc_auc={t['roc_auc']:.4f} pr_auc_nao_alf={t['pr_auc_nao_alf']:.4f} "
          f"recall_nao_alf={t['recall_nao_alf']:.3f} brier={t['brier']:.4f} limiar={r['metricas']['limiar']:.3f}")
    for k, p in caminhos.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
