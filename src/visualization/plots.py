"""Figuras do README e dos notebooks: um estilo só, sem I/O além de salvar()."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sem isso quebra rodando sem tela (CI, terminal) tentando abrir uma janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

from src import config  # noqa: E402

COR_NAO_ALF = "#c0392b"
COR_ALF = "#2e86ab"
PALETA = ["#2e86ab", "#c0392b", "#f39c12", "#27ae60", "#8e44ad", "#7f8c8d"]
NIVEIS = [f"pct_nivel_{i}" for i in range(9)]


def aplicar_estilo() -> None:
    sns.set_theme(style="whitegrid", palette=PALETA, font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})


aplicar_estilo()


def salvar(fig, nome: str) -> Path:
    config.IMAGES.mkdir(parents=True, exist_ok=True)
    caminho = config.IMAGES / f"{nome}.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)  # fecha a figura pra não acumular memória quando gera muitos gráficos em sequência
    return caminho


def plot_roc_pr(curvas: dict) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for nome, (y, proba) in curvas.items():
        RocCurveDisplay.from_predictions(y, proba, name=nome, ax=a1)
        # inverte y e proba pro precision-recall focar na classe não alfabetizado, igual no metrics.py
        PrecisionRecallDisplay.from_predictions(1 - np.asarray(y), 1 - np.asarray(proba), name=nome, ax=a2)
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a1.set_title("ROC (alfabetizado)")
    a2.set_title("Precision-Recall (classe não alfabetizado)")
    return fig


def plot_calibracao(tabela: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfeita")
    ax.plot(tabela["proba_media"], tabela["taxa_observada"], "o-", color=COR_ALF, label="modelo")
    ax.set(xlabel="probabilidade prevista", ylabel="taxa observada", title="Calibração")
    ax.legend()
    return fig


def plot_matriz_confusao(matriz, titulo: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(np.asarray(matriz), annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["prev. não alf.", "prev. alf."], yticklabels=["não alf.", "alf."])
    ax.set_title(titulo or "Matriz de confusão")
    return fig


def plot_barras_horizontais(df: pd.DataFrame, coluna_valor: str, coluna_nome: str = "feature",
                            top: int = 20, titulo: str = "") -> plt.Figure:
    d = df.nlargest(top, coluna_valor).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1))
    ax.barh(d[coluna_nome].astype(str), d[coluna_valor], color=COR_ALF)
    ax.set_title(titulo)
    return fig


def plot_shap_beeswarm(exp, top: int = 15) -> plt.Figure:
    plt.figure()
    shap.plots.beeswarm(exp, max_display=top, show=False)
    return plt.gcf()


def plot_shap_dependence(exp, feature: str) -> plt.Figure:
    plt.figure()
    shap.plots.scatter(exp[:, feature], show=False)
    return plt.gcf()


def plot_por_recorte(df: pd.DataFrame, metrica: str = "roc_auc") -> plt.Figure:
    d = df.sort_values(metrica)
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(d) + 1))
    ax.barh(d["recorte"].astype(str), d[metrica], color=COR_ALF)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set(title=f"{metrica} por recorte", xlim=(0.4, 1.0))
    return fig


def plot_comparacao_modelos(metricas: list[dict], chaves=("roc_auc", "pr_auc_nao_alf", "recall_nao_alf")) -> plt.Figure:
    d = pd.DataFrame([{"modelo": f"{m['regime']}/{m['modelo']}", **{k: m["teste"][k] for k in chaves}} for m in metricas])
    fig, ax = plt.subplots(figsize=(8, 4))
    d.set_index("modelo")[list(chaves)].plot.bar(ax=ax, rot=0)
    ax.set(ylim=(0, 1), title="Comparação no teste")
    return fig


def plot_distribuicao_proficiencia(prof, corte: float = config.CORTE_ALFABETIZACAO) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(np.asarray(prof), bins=80, ax=ax, color=COR_ALF)
    ax.axvline(corte, color=COR_NAO_ALF, ls="--", label=f"corte {corte:.0f}")
    ax.set(title="Distribuição da proficiência (presentes com nota)", xlabel="proficiência")
    ax.legend()
    return fig


def plot_dispersao(df: pd.DataFrame, x: str, y: str, hue: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=12, alpha=0.6, ax=ax)
    return fig


def plot_boxplot_por_grupo(df: pd.DataFrame, x: str, y: str, titulo: str = "") -> plt.Figure:
    ordem = df.groupby(x, observed=True)[y].median().sort_values().index
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(ordem) + 2), 4.5))
    sns.boxplot(data=df, x=x, y=y, order=ordem, ax=ax, color=COR_ALF, fliersize=1.5)
    ax.set_title(titulo)
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_perfis_niveis(df: pd.DataFrame, coluna_grupo: str, colunas: list[str] | None = None) -> plt.Figure:
    colunas = colunas or NIVEIS
    medias = df.groupby(coluna_grupo)[colunas].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for grupo, linha in medias.iterrows():
        ax.plot(range(len(colunas)), linha.to_numpy(), "o-", label=str(grupo))
    ax.set(xticks=range(len(colunas)), xticklabels=[c.replace("pct_nivel_", "nível ") for c in colunas],
           ylabel="% de alunos", title=f"Perfil da distribuição por {coluna_grupo}")
    ax.legend(title=coluna_grupo)
    return fig


def plot_correlacao(df_num: pd.DataFrame) -> plt.Figure:
    corr = df_num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(0.4 * len(corr) + 3, 0.4 * len(corr) + 2))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax, square=True, cbar_kws={"shrink": 0.6})
    ax.set_title("Correlação (Pearson)")
    return fig


def plot_clusters_pca(coords, labels) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=np.asarray(labels).astype(str), s=14, alpha=0.7, ax=ax)
    ax.set(xlabel="PC1", ylabel="PC2", title="Clusters de vulnerabilidade (PCA)")
    return fig


def plot_ranking_risco(df: pd.DataFrame, top: int = 20) -> plt.Figure:
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(d) + 1))
    ax.barh(d["nome_municipio"] + " (" + d["sigla_uf"] + ")", d["criancas_nao_alfabetizadas_2024"], color=COR_NAO_ALF)
    # escreve a probabilidade de não bater a meta na ponta de cada barra, só pra dar o contexto
    # de risco junto do volume (a barra sozinha só mostra o tamanho, não o risco)
    for i, (n, p) in enumerate(zip(d["criancas_nao_alfabetizadas_2024"], d["prob_nao_atingir_2025"])):
        ax.text(n, i, f" {p:.0%}", va="center", fontsize=8)
    ax.set(title="Maior risco de não atingir a meta 2025 × crianças não alfabetizadas", xlabel="crianças não alfabetizadas (2024)")
    return fig
