"""Modelo B: features do município em t -> taxa e risco de não atingir a meta em t+1.

Treina em 2023->2024 e aplica em 2024->2025 (meta_alfabetizacao_2025).
    python -m src.modeling.risco_municipio
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing.pipeline import build_preprocessor

# alvos, identificadores, colunas que são NaN/sem_meta em 2023 inteiro (meta_ano, gap, atingiu_meta,
# situacao_meta) e colunas 100% NaN em um dos dois anos (bolsa família só em 2024, pct_va_* só em 2023):
# todas mudariam de distribuição entre treino (2023) e aplicação (2024)
NAO_FEATURES_B = frozenset({
    "taxa_prox", "situacao_meta_prox", "nao_atingiu_prox", "ano", "id_municipio", "nome_municipio",
    "meta_ano", "gap", "atingiu_meta", "situacao_meta",
    "pct_familias_bolsa_familia", "pct_va_agropecuaria", "pct_va_industria", "pct_va_servicos", "pct_va_adespss",
})
CATEGORICAS_B = ["sigla_uf", "regiao"]
METRICAS_REG = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
METRICAS_CLF = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}


def colunas_b(base: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in CATEGORICAS_B if c in base.columns]
    num = [c for c in base.columns if c not in NAO_FEATURES_B and c not in cat
           and (pd.api.types.is_numeric_dtype(base[c]) or pd.api.types.is_bool_dtype(base[c]))]
    return num, cat


def preparar_treino(base: pd.DataFrame, ano: int = 2023):
    # só treina com município que tem os dois lados: resultado do ano base e o do ano seguinte (o alvo)
    t = base[(base["ano"] == ano) & base["taxa_prox"].notna() & base["taxa_alfabetizacao"].notna()].reset_index(drop=True)
    num, cat = colunas_b(t)
    X = t[num + cat].copy()
    X[num] = X[num].astype("float64")
    return X, t["taxa_prox"].astype(float), t["nao_atingiu_prox"].astype(float), t["id_municipio"]


def _pipe(estimador, X):
    num = [c for c in X.columns if c not in CATEGORICAS_B]
    cat = [c for c in X.columns if c in CATEGORICAS_B]
    return Pipeline([("prep", build_preprocessor(num, cat)), ("clf", estimador)])


def _resumir(nome, tarefa, cv, mapa):
    return [{"modelo": nome, "tarefa": tarefa, "metrica": m, "media": float(abs(cv[f"test_{m}"]).mean()),
             "dp": float(cv[f"test_{m}"].std())} for m in mapa]


def avaliar(X: pd.DataFrame, y_reg: pd.Series, y_clf: pd.Series, seed: int = config.SEED) -> pd.DataFrame:
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    # "persistência" é o baseline ingênuo: chuta que a taxa do ano seguinte vai ser igual à de agora.
    # se o modelo não bater isso, não vale o esforço
    linhas = [{"modelo": "persistencia", "tarefa": "regressao", "metrica": "mae",
               "media": float(mean_absolute_error(y_reg, X["taxa_alfabetizacao"])), "dp": 0.0}]
    for nome, est in [("ridge", Ridge(alpha=1.0)),
                      ("hgb_reg", HistGradientBoostingRegressor(random_state=seed, max_iter=300, learning_rate=0.05))]:
        r = cross_validate(_pipe(est, X), X, y_reg, cv=cv, scoring=METRICAS_REG, n_jobs=-1)
        linhas += _resumir(nome, "regressao", r, METRICAS_REG)
    com_meta = y_clf.notna()  # nem todo município tem meta pactuada -- classificação só roda em quem tem
    for nome, est in [("logistica", LogisticRegression(max_iter=1000)),
                      ("hgb_clf", HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05))]:
        r = cross_validate(_pipe(est, X), X[com_meta], y_clf[com_meta].astype(int), cv=cv, scoring=METRICAS_CLF, n_jobs=-1)
        linhas += _resumir(nome, "classificacao", r, METRICAS_CLF)
    return pd.DataFrame(linhas)


def treinar_final(X, y_reg, y_clf, seed: int = config.SEED):
    reg = _pipe(HistGradientBoostingRegressor(random_state=seed, max_iter=300, learning_rate=0.05), X).fit(X, y_reg)
    com_meta = y_clf.notna()
    clf = _pipe(HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05), X)
    clf.fit(X[com_meta], y_clf[com_meta].astype(int))
    return reg, clf


def gerar_ranking(base: pd.DataFrame, pipe_reg, pipe_clf, ano_aplicacao: int = 2024) -> pd.DataFrame:
    t = base[base["ano"] == ano_aplicacao].reset_index(drop=True)
    X = t[list(pipe_reg.feature_names_in_)].copy()
    p = ano_aplicacao + 1
    out = pd.DataFrame({
        "id_municipio": t["id_municipio"], "nome_municipio": t["nome_municipio"], "sigla_uf": t["sigla_uf"], "regiao": t["regiao"],
        f"taxa_{ano_aplicacao}": t["taxa_alfabetizacao"], f"ic95_{ano_aplicacao}": t["ic95"],
        f"criancas_nao_alfabetizadas_{ano_aplicacao}": t["criancas_nao_alfabetizadas"],
        f"meta_{p}": t["meta_prox"],
        f"taxa_prevista_{p}": np.clip(pipe_reg.predict(X), 0, 100),   # pontos percentuais, como na Gold
        f"prob_nao_atingir_{p}": pipe_clf.predict_proba(X)[:, 1],
    })
    out[f"gap_previsto_{p}"] = out[f"taxa_prevista_{p}"] - out[f"meta_{p}"]
    # "acima da margem" = o gap previsto é pior que o intervalo de confiança do resultado atual --
    # ou seja, não é só ruído da medição, dá pra dizer que o município provavelmente vai errar a meta
    out["acima_da_margem"] = out[f"gap_previsto_{p}"] < -out[f"ic95_{ano_aplicacao}"]
    # prioridade pondera risco por volume: um município com prob alta mas poucas crianças
    # não alfabetizadas importa menos pra política pública do que um grande com risco moderado
    out["prioridade"] = out[f"prob_nao_atingir_{p}"] * out[f"criancas_nao_alfabetizadas_{ano_aplicacao}"]
    return out.sort_values("prioridade", ascending=False).reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    base = pd.read_parquet(config.PROCESSED / "base_modelagem_municipio.parquet")
    X, y_reg, y_clf, _ = preparar_treino(base)
    aval = avaliar(X, y_reg, y_clf)
    reg, clf = treinar_final(X, y_reg, y_clf)
    rank = gerar_ranking(base, reg, clf)

    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, config.MODELS / "modelo_b_regressao.joblib")
    joblib.dump(clf, config.MODELS / "modelo_b_classificacao.joblib")
    rank.round(4).to_csv(config.REPORTS / "ranking_risco_municipios.csv", index=False)
    e23 = base["ano"] == 2023
    resumo = {
        "n_treino": int(len(X)), "n_com_meta": int(y_clf.notna().sum()),
        "n_indistinguivel_2024": int((base.loc[e23, "situacao_meta_prox"] == "indistinguivel").sum()),
        "features": list(X.columns), "avaliacao": aval.to_dict("records"),
        "n_aplicacao_2024": int(len(rank)), "n_sem_meta_2025": int(rank["meta_2025"].isna().sum()),
        "n_acima_da_margem": int(rank["acima_da_margem"].sum()),
        "prob_media_nao_atingir_2025": float(rank["prob_nao_atingir_2025"].mean()),
    }
    (config.REPORTS / "metricas_modelo_b.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(aval.pivot_table(index="modelo", columns="metrica", values="media").round(4))
    print(rank.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
