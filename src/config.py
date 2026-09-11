"""Caminhos e constantes compartilhados por todo o projeto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
GOLD = DATA / "gold"
SILVER = DATA / "silver"
EXTERNAL = DATA / "external"
PROCESSED = DATA / "processed"
MODELS = RAIZ / "models"
REPORTS = RAIZ / "reports"
IMAGES = RAIZ / "images"

SEED = 42
CORTE_ALFABETIZACAO = 743
ANO_ALVO = 2024

# nunca podem entrar como feature: definem o alvo, a presença na prova,
# são pesos amostrais ou identificadores que vazariam o resultado
COLUNAS_PROIBIDAS = frozenset({
    "proficiencia", "alfabetizado", "presenca", "presente", "sem_nota",
    "preenchimento_caderno", "peso_aluno", "caderno", "serie", "id_aluno",
    "id_escola", "id_municipio", "gap", "atingiu_meta", "situacao_meta",
    "presenca_nome", "_row_hash", "_ingestion_ts", "_source",
})
