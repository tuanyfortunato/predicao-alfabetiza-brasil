import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from src import config
from src.modeling import clusters
from tests.conftest import montar_base_municipio_sintetica


def test_avaliar_k_acha_tres_blobs():
    Xb, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    X = pd.DataFrame(Xb, columns=["a", "b"])
    res = clusters.avaliar_k(X, ks=range(2, 7))
    assert list(res.columns) == ["k", "inercia", "silhueta"] and res["inercia"].is_monotonic_decreasing
    assert int(res.loc[res["silhueta"].idxmax(), "k"]) == 3


def test_preparar_perfis_e_pca():
    base = montar_base_municipio_sintetica()
    X, ids = clusters.preparar_matriz(base, ano=2024)
    assert len(X) == 200 and set(clusters.COLUNAS_FORMA) <= set(X.columns) and "tdi_ai" in X.columns
    pipe = clusters.ajustar(X, k=3)
    lab = clusters.rotulos(pipe, X)
    assert set(lab) == {0, 1, 2}
    p = clusters.perfis(X, ids, lab)
    assert len(p) == 3 and p["n"].sum() == 200 and {"taxa_media", "regiao_dominante", "log_pib_per_capita"} <= set(p.columns)
    assert clusters.projetar_pca(pipe, X).shape == (200, 2)


def test_cli(lake_tmp):
    config.PROCESSED.mkdir(exist_ok=True)
    montar_base_municipio_sintetica().to_parquet(config.PROCESSED / "base_modelagem_municipio.parquet")
    clusters.main(["--k", "3"])
    assert (config.REPORTS / "perfis_clusters.csv").exists() and (config.IMAGES / "clusters_pca.png").exists()
    assert len(pd.read_csv(config.REPORTS / "clusters_municipios.csv")) == 200


def test_cli_sem_k_usa_selecao_automatica(lake_tmp):
    # sem --k, main() deve cair no ramo `k = args.k or int(aval.loc[aval["silhueta"].idxmax(), "k"])`
    config.PROCESSED.mkdir(exist_ok=True)
    montar_base_municipio_sintetica().to_parquet(config.PROCESSED / "base_modelagem_municipio.parquet")
    clusters.main([])
    assert (config.REPORTS / "avaliacao_k.csv").exists()
    perfis = pd.read_csv(config.REPORTS / "perfis_clusters.csv")
    assert len(perfis) >= 2
    assert len(pd.read_csv(config.REPORTS / "clusters_municipios.csv")) == 200
