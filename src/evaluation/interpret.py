"""Importância por permutação (colunas cruas) e SHAP (matriz transformada)."""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src import config


def nomes_features(pipeline) -> list[str]:
    nomes = []
    for n in pipeline.named_steps["prep"].get_feature_names_out():
        # o ColumnTransformer prefixa tudo com "num__"/"cat__"; tira isso e deixa
        # o indicador de valor faltante com um nome legível em vez do nome técnico do sklearn
        n = n.split("__", 1)[1]
        nomes.append(n.replace("missingindicator_", "faltante_") if n.startswith("missingindicator_") else n)
    return nomes


def amostrar_posicoes(n_total: int, n_amostra: int | None, seed: int = config.SEED) -> np.ndarray:
    """Mesma amostra que explicar_shap/importancia_permutacao tiram; útil para alinhar grupos (região, UF)."""
    if n_amostra is None or n_amostra >= n_total:
        return np.arange(n_total)
    return np.random.default_rng(seed).choice(n_total, size=n_amostra, replace=False)


def _amostra(X, n, seed, *outros):
    pos = amostrar_posicoes(len(X), n, seed)
    if len(pos) == len(X):
        return (X, *outros)
    return (X.iloc[pos], *[None if o is None else np.asarray(o)[pos] for o in outros])


def importancia_permutacao(pipeline, X, y, n_repeats: int = 5, seed: int = config.SEED,
                           pesos=None, n_amostra: int | None = 100_000) -> pd.DataFrame:
    Xs, ys, ps = _amostra(X, n_amostra, seed, y, pesos)
    r = permutation_importance(pipeline, Xs, ys, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=seed, sample_weight=ps, n_jobs=-1)
    out = pd.DataFrame({"feature": Xs.columns, "importancia_media": r.importances_mean, "importancia_dp": r.importances_std})
    return out.sort_values("importancia_media", ascending=False).reset_index(drop=True)


def explicar_shap(pipeline, X, n_amostra: int | None = 10_000, seed: int = config.SEED) -> shap.Explanation:
    (Xs,) = _amostra(X, n_amostra, seed)
    Xt = pipeline.named_steps["prep"].transform(Xs)
    clf = pipeline.named_steps["clf"]
    nomes = nomes_features(pipeline)
    if isinstance(clf, HistGradientBoostingClassifier):
        explicador = shap.TreeExplainer(clf)
        valores = explicador.shap_values(Xt)
        base = explicador.expected_value
    elif isinstance(clf, LogisticRegression):
        explicador = shap.LinearExplainer(clf, Xt)
        valores = explicador.shap_values(Xt)
        base = explicador.expected_value
    else:
        raise TypeError(f"sem explicador para {type(clf).__name__}")
    # o shap muda o formato de saída dependendo da versão/modelo: às vezes devolve uma lista
    # [valores da classe 0, valores da classe 1], às vezes um array 3D -- os dois casos abaixo
    # normalizam pra sempre pegar a classe 1 (alfabetizado), que é o que a gente quer explicar
    if isinstance(valores, list):                 # algumas versões devolvem [classe0, classe1]
        valores, base = valores[1], np.ravel(base)[-1]
    elif np.ndim(valores) == 3:
        valores, base = valores[:, :, 1], np.ravel(base)[-1]
    return shap.Explanation(values=np.asarray(valores), base_values=np.ravel(base)[0] if np.ndim(base) else base,
                            data=np.asarray(Xt), feature_names=nomes)


def resumo_shap(exp: shap.Explanation) -> pd.DataFrame:
    out = pd.DataFrame({"feature": exp.feature_names, "shap_medio_abs": np.abs(exp.values).mean(axis=0)})
    return out.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)


def shap_por_grupo(exp: shap.Explanation, grupo) -> pd.DataFrame:
    df = pd.DataFrame(np.abs(exp.values), columns=exp.feature_names)
    df["_g"] = np.asarray(grupo)
    return df.groupby("_g").mean().rename_axis(None)


def coeficientes_logistica(pipeline) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    if not isinstance(clf, LogisticRegression):
        raise TypeError("só a regressão logística tem coeficientes")
    out = pd.DataFrame({"feature": nomes_features(pipeline), "coeficiente": clf.coef_[0]})
    return out.reindex(out["coeficiente"].abs().sort_values(ascending=False).index).reset_index(drop=True)
