import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import config
from src.evaluation import metrics
from src.visualization import plots


def test_curvas_calibracao_e_matriz_salvam_png(lake_tmp):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    fig = plots.plot_roc_pr({"hgb": (y, proba), "dummy": (y, np.full(500, 0.5))})
    assert isinstance(fig, Figure)
    caminho = plots.salvar(fig, "teste_roc")
    assert caminho == config.IMAGES / "teste_roc.png" and caminho.stat().st_size > 0
    assert isinstance(plots.plot_calibracao(metrics.tabela_calibracao(y, proba)), Figure)
    assert isinstance(plots.plot_matriz_confusao([[100, 20], [30, 350]], "teste"), Figure)


def test_barras_recorte_comparacao_e_outros():
    imp = pd.DataFrame({"feature": list("abcdef"), "importancia_media": [6, 5, 4, 3, 2, 1]})
    assert isinstance(plots.plot_barras_horizontais(imp, "importancia_media", top=3), Figure)
    rec = pd.DataFrame({"recorte": ["SP", "BA"], "roc_auc": [0.7, 0.65], "n": [1000, 800]})
    assert isinstance(plots.plot_por_recorte(rec), Figure)
    ms = [{"modelo": "dummy", "regime": "producao", "teste": {"roc_auc": 0.5, "pr_auc_nao_alf": 0.4, "recall_nao_alf": 0.8}},
          {"modelo": "hgb", "regime": "producao", "teste": {"roc_auc": 0.7, "pr_auc_nao_alf": 0.6, "recall_nao_alf": 0.8}}]
    assert isinstance(plots.plot_comparacao_modelos(ms), Figure)
    assert isinstance(plots.plot_distribuicao_proficiencia(np.random.default_rng(1).normal(750, 50, 1000)), Figure)
    df = pd.DataFrame({"x": np.arange(10), "y": np.arange(10) * 2.0, "r": ["a", "b"] * 5})
    assert isinstance(plots.plot_dispersao(df, "x", "y", hue="r"), Figure)
    assert isinstance(plots.plot_boxplot_por_grupo(df, "r", "y", "t"), Figure)
    assert isinstance(plots.plot_correlacao(df[["x", "y"]]), Figure)
    assert isinstance(plots.plot_clusters_pca(np.random.default_rng(2).normal(size=(50, 2)), np.arange(50) % 3), Figure)
    rank = pd.DataFrame({"nome_municipio": [f"m{i}" for i in range(30)], "sigla_uf": ["BA"] * 30,
                         "prob_nao_atingir_2025": np.linspace(0.9, 0.3, 30), "criancas_nao_alfabetizadas_2024": np.linspace(5000, 100, 30)})
    assert isinstance(plots.plot_ranking_risco(rank, top=10), Figure)


def test_perfis_niveis_uma_linha_por_grupo():
    df = pd.DataFrame({**{f"pct_nivel_{i}": np.random.default_rng(i).random(6) for i in range(9)}, "cluster": [0, 0, 1, 1, 2, 2]})
    fig = plots.plot_perfis_niveis(df, "cluster")
    assert isinstance(fig, Figure) and len(fig.axes[0].lines) == 3
