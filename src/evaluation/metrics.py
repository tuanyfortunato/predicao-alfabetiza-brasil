"""Métricas do modelo de aluno. y = alfabetizado (1/0); a classe de interesse é 0 (não alfabetizado)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def calcular_metricas(y, proba, limiar: float = 0.5, pesos=None) -> dict:
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    pesos = None if pesos is None else np.asarray(pesos, dtype=float)
    pred = (proba >= limiar).astype(int)
    nao = 1 - y
    return {
        "n": int(len(y)),
        "limiar": float(limiar),
        "prevalencia_nao_alf": float(np.average(nao, weights=pesos)),
        "roc_auc": float(roc_auc_score(y, proba, sample_weight=pesos)),
        "pr_auc_nao_alf": float(average_precision_score(nao, 1 - proba, sample_weight=pesos)),
        "f1_nao_alf": float(f1_score(y, pred, pos_label=0, sample_weight=pesos, zero_division=0)),
        "recall_nao_alf": float(recall_score(y, pred, pos_label=0, sample_weight=pesos, zero_division=0)),
        "precisao_nao_alf": float(precision_score(y, pred, pos_label=0, sample_weight=pesos, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=pesos)),
        "acuracia": float(accuracy_score(y, pred, sample_weight=pesos)),
        "brier": float(brier_score_loss(y, proba, sample_weight=pesos)),
        "matriz": confusion_matrix(y, pred, labels=[0, 1], sample_weight=pesos).round(2).tolist(),
    }


def escolher_limiar(y, proba, recall_minimo: float = 0.8) -> float:
    """Não alfabetizado é previsto quando proba < limiar. O menor limiar que ainda
    captura recall_minimo dos não alfabetizados é o que erra menos alfabetizados.

    Usa uma fórmula exata baseada em rank (em vez de np.quantile) porque com
    probabilidades empatadas — rotina em modelos de árvore, cujas folhas repetem a
    mesma probabilidade para muitas amostras — a interpolação de np.quantile pode
    devolver um limiar acima do necessário, entregando recall maior que o pedido
    às custas de mais alfabetizados classificados como não alfabetizados."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    proba0 = np.sort(proba[y == 0])
    n = len(proba0)
    k = int(np.clip(np.ceil(n * recall_minimo) - 1, 0, n - 1))
    return float(np.nextafter(proba0[k], np.inf))


def metricas_por_recorte(recorte, y, proba, limiar: float, pesos=None, minimo: int = 500) -> pd.DataFrame:
    recorte = pd.Series(np.asarray(recorte))
    y, proba = np.asarray(y), np.asarray(proba)
    pesos = None if pesos is None else np.asarray(pesos)
    linhas = []
    for valor, pos in recorte.groupby(recorte).groups.items():
        pos = np.asarray(pos)
        # exige as duas classes presentes: roc_auc_score não é definido com uma só.
        if len(pos) < minimo or len(np.unique(y[pos])) < 2:
            continue
        m = calcular_metricas(y[pos], proba[pos], limiar, None if pesos is None else pesos[pos])
        m.pop("matriz")
        linhas.append({"recorte": valor, **m})
    if not linhas:
        colunas = ["recorte"] + [k for k in calcular_metricas([0, 1], [0.1, 0.9]) if k != "matriz"]
        return pd.DataFrame(columns=colunas).reset_index(drop=True)
    return pd.DataFrame(linhas).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def tabela_calibracao(y, proba, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "proba": np.asarray(proba)})
    df["faixa"] = pd.cut(df["proba"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = df.groupby("faixa", observed=True).agg(proba_media=("proba", "mean"), taxa_observada=("y", "mean"), n=("y", "size"))
    return out.reset_index()
