"""Pré-processamento acoplado ao modelo: um único Pipeline serializado."""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numerico = Pipeline([
        # add_indicator=True cria uma coluna "faltava ou não" pra cada numérica -- o fato de faltar
        # (ex.: município sem histórico) pode ser informativo, não só o valor imputado
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    # handle_unknown=ignore evita quebrar em produção se aparecer uma categoria (UF, rede) nunca vista no treino
    categorico = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("num", numerico, num_cols), ("cat", categorico, cat_cols)],
        remainder="drop", verbose_feature_names_out=True,
    )


def _estimador(modelo: str, seed: int):
    if modelo == "dummy":
        return DummyClassifier(strategy="prior")  # baseline burro: só chuta a classe mais frequente, serve de piso de comparação
    if modelo == "logistica":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if modelo == "hgb":
        # early_stopping evita treinar as 500 árvores à toa quando a validação interna já parou de melhorar
        return HistGradientBoostingClassifier(
            random_state=seed, max_iter=500, learning_rate=0.1,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        )
    raise ValueError(f"modelo desconhecido: {modelo!r} (use dummy, logistica ou hgb)")


def build_pipeline(modelo: str, num_cols: list[str], cat_cols: list[str],
                   seed: int = config.SEED, params: dict | None = None) -> Pipeline:
    pipe = Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("clf", _estimador(modelo, seed))])
    if params:
        pipe.set_params(**params)
    return pipe
