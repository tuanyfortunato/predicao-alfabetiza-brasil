"""Clusters de vulnerabilidade: forma da distribuição de proficiência + contexto socioeconômico.

    python -m src.modeling.clusters --k 4
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.visualization import plots

COLUNAS_FORMA = [f"pct_nivel_{i}" for i in range(9)]
COLUNAS_CONTEXTO = ["taxa_alfabetizacao_adultos", "log_pib_per_capita", "pct_escolas_rurais", "tdi_ai", "log_populacao", "densidade_demografica"]
COLUNAS_ID = ["id_municipio", "nome_municipio", "sigla_uf", "regiao", "taxa_alfabetizacao", "criancas_nao_alfabetizadas"]


def preparar_matriz(base: pd.DataFrame, ano: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = base[base["ano"] == ano].dropna(subset=COLUNAS_FORMA).reset_index(drop=True)
    return t[COLUNAS_FORMA + COLUNAS_CONTEXTO].astype("float64"), t[COLUNAS_ID]


def build_cluster_pipeline(k: int, seed: int = config.SEED) -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("km", KMeans(n_clusters=k, n_init=10, random_state=seed))])


def avaliar_k(X: pd.DataFrame, ks=range(2, 9), seed: int = config.SEED) -> pd.DataFrame:
    prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Xt = prep.fit_transform(X)
    # silhouette_score é O(n²), calcular com todos os municípios ficaria lento à toa --
    # amostra de 5000 já dá uma estimativa estável o suficiente pra escolher o k
    amostra = np.random.default_rng(seed).choice(len(Xt), size=min(5000, len(Xt)), replace=False)
    linhas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xt)
        linhas.append({"k": k, "inercia": float(km.inertia_), "silhueta": float(silhouette_score(Xt[amostra], km.labels_[amostra]))})
    return pd.DataFrame(linhas)


def ajustar(X: pd.DataFrame, k: int, seed: int = config.SEED) -> Pipeline:
    return build_cluster_pipeline(k, seed).fit(X)


def rotulos(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(X)


def projetar_pca(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Xt = pipeline[:-1].transform(X)
    return PCA(n_components=2, random_state=config.SEED).fit_transform(Xt)


def perfis(X: pd.DataFrame, ids: pd.DataFrame, labels) -> pd.DataFrame:
    df = pd.concat([ids.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    df["cluster"] = np.asarray(labels)
    g = df.groupby("cluster")
    out = pd.DataFrame({
        "n": g.size(), "taxa_media": g["taxa_alfabetizacao"].mean(),
        "criancas_nao_alfabetizadas": g["criancas_nao_alfabetizadas"].sum(),
        # região mais comum em cada cluster, só pra ajudar a interpretar o cluster (não entra no modelo)
        "regiao_dominante": g["regiao"].agg(lambda s: s.value_counts().index[0]),
        "pct_regiao_dominante": g["regiao"].agg(lambda s: s.value_counts(normalize=True).iloc[0]),
    })
    return pd.concat([out, g[list(X.columns)].mean()], axis=1).reset_index()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args(argv)
    base = pd.read_parquet(config.PROCESSED / "base_modelagem_municipio.parquet")
    X, ids = preparar_matriz(base)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    aval = avaliar_k(X)
    aval.to_csv(config.REPORTS / "avaliacao_k.csv", index=False)
    k = args.k or int(aval.loc[aval["silhueta"].idxmax(), "k"])  # sem --k, deixa a silhueta escolher o melhor

    fig, (a1, a2) = plots.plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(aval["k"], aval["inercia"], "o-"); a1.set(title="Cotovelo", xlabel="k", ylabel="inércia")
    a2.plot(aval["k"], aval["silhueta"], "o-"); a2.set(title="Silhueta", xlabel="k")
    plots.salvar(fig, "clusters_k")

    pipe = ajustar(X, k)
    lab = rotulos(pipe, X)
    perfis(X, ids, lab).round(4).to_csv(config.REPORTS / "perfis_clusters.csv", index=False)
    ids.assign(cluster=lab).to_csv(config.REPORTS / "clusters_municipios.csv", index=False)
    plots.salvar(plots.plot_clusters_pca(projetar_pca(pipe, X), lab), "clusters_pca")
    print(f"k={k}")
    print(perfis(X, ids, lab)[["cluster", "n", "taxa_media", "criancas_nao_alfabetizadas", "regiao_dominante"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
