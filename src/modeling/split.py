"""Divisão treino/validação/teste e CV sempre por escola: nenhum id_escola em dois lados."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src import config


def dividir_por_escola(grupos, frac_val: float = 0.15, frac_teste: float = 0.15,
                       seed: int = config.SEED) -> dict[str, np.ndarray]:
    grupos = np.asarray(grupos)
    idx = np.arange(len(grupos))
    # separa teste primeiro, depois divide o resto em treino/validação -- por isso a fração
    # de validação precisa ser recalculada em cima do "resto" (que já não tem os 15% do teste)
    resto, teste = next(GroupShuffleSplit(n_splits=1, test_size=frac_teste, random_state=seed).split(idx, groups=grupos))
    frac_val_rel = frac_val / (1 - frac_teste)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=frac_val_rel, random_state=seed).split(resto, groups=grupos[resto]))
    partes = {"treino": resto[tr], "validacao": resto[va], "teste": teste}
    conferir_sem_vazamento(grupos, partes)
    return partes


def cv_por_grupo(n_splits: int = 5, seed: int = config.SEED) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def conferir_sem_vazamento(grupos, partes: dict[str, np.ndarray]) -> None:
    # dupla checagem depois do split: se a mesma escola aparecer em duas partes é bug grave
    # (vaza contexto de escola entre treino e teste), então isso aqui tem que estourar, não só logar
    grupos = pd.Series(np.asarray(grupos))
    vistos: dict = {}
    for nome, pos in partes.items():
        for g in set(grupos.iloc[pos]):
            assert vistos.setdefault(g, nome) == nome, f"escola {g} está em {vistos[g]} e em {nome}"
