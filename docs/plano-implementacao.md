# Plano de Implementação — predicao-alfabetiza-brasil (Tech Challenge Fase 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar, em repositório próprio, o modelo supervisionado de alfabetização (aluno alfabetizado × não), a feature store que o alimenta, o modelo de risco de meta por município, os clusters de vulnerabilidade, os notebooks de EDA/modelagem e a documentação — tudo reproduzível por script e coberto por testes. Vídeo e apresentação ficam fora deste plano.

**Architecture:** O repositório consome a Gold e a Silver da Fase 2 como dados de entrada (Gold commitada, Silver de alunos baixada por script) e extrai as fontes externas do BigQuery da Base dos Dados com consultas já agregadas por município. `src/preprocessing` monta a feature store (contexto municipal defasado t−1, contexto externo, leave-one-out da escola), `src/modeling` treina/avalia com pipeline scikit-learn e split por escola, `src/evaluation` mede e interpreta (SHAP), `src/visualization` gera as figuras que o README e os notebooks usam.

**Tech Stack:** Python 3.11, pandas 2.2, pyarrow 17, scikit-learn 1.5, shap 0.46, matplotlib 3.9, seaborn 0.13, joblib, google-cloud-bigquery 3.x (só na extração), pytest 8.

**Spec:** `docs/planejamento-fase3.md` (cópia da especificação; ler junto com o adendo abaixo).

## Adendo à especificação (decisões tomadas depois dela)

1. **Repositório novo** (`predicao-alfabetiza-brasil`), não o da Fase 2. A Fase 2 não é alterada (no máximo um link no README dela, fora deste plano).
2. **A feature store nasce em `src/preprocessing/`** deste repo, não como tabela Gold no pipeline da Fase 2. As seções 2.2, 4.2 e 6 da spec ficam substituídas pela estrutura de arquivos deste plano.
3. **Dados de entrada:** Gold (5 tabelas) + Silver de metas e resultados são **commitados** (4,7 MB). Silver de alunos (124 MB) fica fora do Git e é copiada por `scripts/baixar_dados.py` a partir do lake local da Fase 2 ou de uma URL (Release no GitHub da Fase 2).
4. **Fontes externas** são extraídas por `scripts/extrair_externas.py`, com agregação feita no próprio BigQuery, e as saídas (poucos MB) são **commitadas** em `data/external/`. Quem clona não precisa de credencial GCP.
5. **FUNDEB fica fora** (a tabela usa códigos de indicador que exigem decodificação; vai para "evoluções futuras"). INSE fora (só 2014–2015).
6. **Nada roda na AWS.** Treino local.
7. **Sem `caderno` como feature.** Testado só na EDA (H10).
8. **`peso_aluno` não entra no treino**; entra nas métricas ponderadas, reportadas ao lado das não ponderadas.
9. **`docs/` é versionado inteiro**, inclusive `planejamento-fase3.md` e este plano. Não existe `docs/teste/`; a nota da spec sobre não versionar o planejamento não vale aqui.
10. **Onde a spec e este plano divergem** (nomes de notebooks na seção 1, `src/03_gold/` na 4.7, branches na 7), **vale o plano**.

## Global Constraints

- Python **3.11** (o `.venv` da Fase 2 é 3.11; criar `.venv` novo aqui).
- Universo do alvo: **presentes com nota** (`presente == True and sem_nota == False`), ano alvo **2024**.
- Alvo: coluna `alfabetizado` da Silver (idêntica a `proficiencia >= 743`).
- **Colunas proibidas na matriz de features:** `proficiencia`, `alfabetizado`, `presenca`, `presente`, `sem_nota`, `preenchimento_caderno`, `peso_aluno`, `caderno`, `serie`, `id_aluno`, `id_escola`, `id_municipio`, `gap`, `atingiu_meta`, `situacao_meta`, e qualquer coluna do ano alvo agregada por município/escola sem leave-one-out.
- Contexto municipal sempre de **t−1** (2023 para o alvo 2024). Externas: regra por fonte na Task 5.
- Split **por grupo `id_escola`** (nenhuma escola em dois lados). Teste tocado uma vez.
- `random_state=42` em tudo que aceita seed. Constante única em `src/config.py`.
- Código versionado em português, comentários mínimos, tom de dev humano. Mensagens de commit curtas, minúsculas, no imperativo ("adiciona", "corrige"), sem trailer de coautoria.
- Testes: `pytest` na raiz, fixtures pequenas montadas à mão em `tests/conftest.py`; nunca ler `data/` real nos testes.
- Nenhum notebook é fonte de verdade: toda lógica reutilizável mora em `src/`, o notebook importa.

---

## Estrutura de arquivos

```
predicao-alfabetiza-brasil/
├── README.md                          # 11 seções (Task 19)
├── requirements.txt
├── .gitignore
├── .env.example                       # GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID, FASE2_LAKE_PATH
├── pytest.ini
├── data/
│   ├── gold/                          # commitado: indicador_municipio.parquet, meta_vs_resultado.parquet,
│   │                                  #   evolucao_temporal.parquet, perfil_escola.parquet, distribuicao_proficiencia.parquet
│   ├── silver/                        # commitado: metas.parquet, resultados_municipio.parquet
│   │   └── alunos/ano=YYYY/*.parquet  # NÃO commitado (124 MB), via scripts/baixar_dados.py
│   ├── external/                      # commitado: censo2022_municipio, pib_municipio, populacao_municipio,
│   │                                  #   ideb_municipio, indicadores_municipio, censo_escolar_municipio, diretorios_municipio (.parquet)
│   └── processed/                     # NÃO commitado: base_modelagem_aluno.parquet, base_modelagem_municipio.parquet
├── scripts/
│   ├── baixar_dados.py                # Fase 2 lake -> data/gold, data/silver
│   └── extrair_externas.py            # BigQuery -> data/external
├── src/
│   ├── __init__.py
│   ├── config.py                      # caminhos, SEED, CORTE, ANO_ALVO, COLUNAS_PROIBIDAS
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── carregar.py                # leitura de gold/silver/external
│   │   ├── externas.py                # contexto externo por município (regra temporal por fonte)
│   │   ├── contexto.py                # contexto municipal t-1 e leave-one-out da escola
│   │   ├── feature_store.py           # monta base_modelagem_aluno / base_modelagem_municipio (CLI)
│   │   ├── features.py                # listas de colunas por regime + guarda de leakage
│   │   └── pipeline.py                # build_preprocessor / build_pipeline
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── split.py                   # split e CV por escola
│   │   ├── train.py                   # CLI ponta a ponta (modelo A)
│   │   ├── tune.py                    # busca de hiperparâmetros (modelo A)
│   │   ├── predict.py                 # pontua base nova com o joblib
│   │   ├── risco_municipio.py         # modelo B + ranking 2025 (CLI)
│   │   └── clusters.py                # clusters de vulnerabilidade (CLI)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # métricas (ponderadas ou não), por recorte, limiar
│   │   └── interpret.py               # permutation importance + SHAP
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                   # estilo único + funções de figura
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modelagem_aluno.ipynb
│   ├── 03_risco_municipio.ipynb
│   └── 04_clusters.ipynb
├── models/                            # NÃO commitado
├── reports/                           # metricas_*.json, melhores_params_*.json, ranking_risco_municipios.csv, perfis_clusters.csv, relatorio_tecnico.md
├── images/                            # figuras (commitadas)
├── docs/                              # commitado inteiro
│   ├── planejamento-fase3.md          # especificação
│   ├── plano-implementacao.md         # este plano
│   └── dicionario_base_modelagem.md   # contrato da feature store (Task 7)
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_baixar_dados.py
    ├── test_carregar.py
    ├── test_extrair_externas.py
    ├── test_externas.py
    ├── test_contexto.py
    ├── test_feature_store.py
    ├── test_features.py
    ├── test_pipeline.py
    ├── test_split.py
    ├── test_metrics.py
    ├── test_train.py
    ├── test_tune.py
    ├── test_interpret.py
    ├── test_plots.py
    ├── test_predict.py
    ├── test_risco_municipio.py
    └── test_clusters.py
```

## Branches e PRs

Trabalhar sempre em branch a partir de `develop`; PR para `develop` com descrição das decisões; ao final, PR `develop` → `main`.

| Branch | Tasks | Entrega do PR |
|---|---|---|
| `feature/setup-e-dados` | 1–3 | esqueleto, dados da Fase 2 no lugar, leitura testada |
| `feature/enriquecimento-externo` | 4–5 | extração do BigQuery e contexto externo por município |
| `feature/feature-store` | 6–7 | contexto t−1, LOO, base de modelagem + dicionário |
| `feature/pipeline-modelo` | 8–12 | pipeline sklearn, split, métricas, treino, tuning |
| `feature/interpretacao` | 13–15 | SHAP, figuras, predict |
| `feature/risco-e-clusters` | 16–17 | modelo B, ranking 2025, clusters |
| `feature/notebooks` | 18 | EDA e notebooks executados |
| `docs/readme` | 19 | README, relatório técnico |

---

### Task 1: Esqueleto do repositório e configuração

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.env.example`, `pytest.ini`, `src/__init__.py`, `src/config.py`, `src/preprocessing/__init__.py`, `src/modeling/__init__.py`, `src/evaluation/__init__.py`, `src/visualization/__init__.py`, `tests/__init__.py` (vazio), `tests/conftest.py`, `tests/test_config.py`
- Modify: `README.md` (placeholder curto; o definitivo é a Task 19)

**Interfaces:**
- Produces: `src.config` com `RAIZ, DATA, GOLD, SILVER, EXTERNAL, PROCESSED, MODELS, REPORTS, IMAGES` (Path), `SEED = 42`, `CORTE_ALFABETIZACAO = 743`, `ANO_ALVO = 2024`, `COLUNAS_PROIBIDAS: frozenset[str]`; fixture `lake_tmp` em `tests/conftest.py` que redireciona os caminhos para `tmp_path`.

- [ ] **Step 1: Criar ambiente e branch**

```bash
cd C:/Users/tcarm/Projetos/predicao-alfabetiza-brasil
git checkout -b feature/setup-e-dados develop
"C:/Users/tcarm/AppData/Local/Programs/Python/Python311/python.exe" -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
```

- [ ] **Step 2: Escrever `requirements.txt`**

```
pandas==2.2.3
pyarrow==17.0.0
numpy==1.26.4
scikit-learn==1.5.2
shap==0.46.0
joblib==1.4.2
matplotlib==3.9.2
seaborn==0.13.2
python-dotenv==1.0.1

# só a extração das fontes externas usa
google-cloud-bigquery==3.42.2
db-dtypes==1.3.0

# desenvolvimento
pytest==8.3.3
ipykernel==6.29.5
nbconvert==7.16.4
```

Instalar: `.venv/Scripts/python.exe -m pip install -r requirements.txt`

- [ ] **Step 3: Escrever `.gitignore`**

```
# credenciais
credentials/
*.json
!data/**/*.json
!reports/*.json
.env

# ambiente
.venv*/
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/

# dados pesados e gerados (Gold/Silver leves e external são versionados)
data/silver/alunos/
data/processed/
models/

# SO / IDE
.DS_Store
.vscode/
```

- [ ] **Step 4: Escrever `.env.example` e `pytest.ini`**

`.env.example`:
```ini
# Só para scripts/extrair_externas.py (BigQuery da Base dos Dados)
GOOGLE_APPLICATION_CREDENTIALS=./credentials/sua-chave.json
GCP_PROJECT_ID=seu-project-id

# Só para scripts/baixar_dados.py: raiz do lake da Fase 2 (pasta data/ do pipeline)
FASE2_LAKE_PATH=../pipeline-dados-alfabetiza-brasil/data
```

`pytest.ini`:
```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 5: Escrever o teste de configuração**

`tests/test_config.py`:
```python
from src import config


def test_caminhos_derivam_da_raiz():
    assert config.GOLD == config.DATA / "gold"
    assert config.PROCESSED == config.DATA / "processed"
    assert config.MODELS.parent == config.RAIZ


def test_constantes_do_dominio():
    assert config.CORTE_ALFABETIZACAO == 743
    assert config.ANO_ALVO == 2024
    assert config.SEED == 42


def test_colunas_proibidas_cobrem_o_alvo_e_a_presenca():
    for col in ["proficiencia", "alfabetizado", "presente", "sem_nota", "peso_aluno", "id_escola"]:
        assert col in config.COLUNAS_PROIBIDAS
```

- [ ] **Step 6: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 7: Escrever `src/config.py` e os `__init__.py`**

`src/config.py`:
```python
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
```

Os cinco `__init__.py` ficam vazios.

- [ ] **Step 8: Escrever `tests/conftest.py`**

```python
"""Fixtures compartilhadas. Nenhum teste lê data/ de verdade."""
import pandas as pd
import pytest

from src import config


@pytest.fixture
def lake_tmp(tmp_path, monkeypatch):
    """Redireciona todos os caminhos de dados para uma pasta temporária."""
    for nome in ["DATA", "GOLD", "SILVER", "EXTERNAL", "PROCESSED", "MODELS", "REPORTS", "IMAGES"]:
        destino = tmp_path / nome.lower()
        destino.mkdir()
        monkeypatch.setattr(config, nome, destino)
    return tmp_path


def montar_alunos(linhas: list[dict]) -> pd.DataFrame:
    """Linhas com as colunas da Silver de alunos; ausentes vêm sem proficiencia."""
    base = {
        "ano": 2024, "id_municipio": 3550308, "id_escola": 60000001, "sigla_uf": "SP",
        "rede_nome": "municipal", "caderno": "1", "serie": 2, "rede": 3,
        "presenca": 1, "preenchimento_caderno": 1, "peso_aluno": 1.0,
    }
    df = pd.DataFrame([{**base, **l} for l in linhas])
    df["id_aluno"] = range(1, len(df) + 1)
    df["presente"] = df["presenca"] == 1
    df["sem_nota"] = df["proficiencia"].isna()
    df["alfabetizado"] = (df["proficiencia"] >= config.CORTE_ALFABETIZACAO).fillna(False).astype(int)
    return df
```

- [ ] **Step 9: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: 3 passed

- [ ] **Step 10: README placeholder e commit**

`README.md`:
```markdown
# predicao-alfabetiza-brasil

Modelo supervisionado de alfabetização infantil a partir da camada Gold de
[pipeline-dados-alfabetiza-brasil](https://github.com/tuanyfortunato/pipeline-dados-alfabetiza-brasil).
Tech Challenge — Fase 3. README completo em construção.
```

```bash
git add .gitignore requirements.txt .env.example pytest.ini src tests README.md
git commit -m "esqueleto do projeto: config, fixtures e requirements"
```

---

### Task 2: Trazer Gold e Silver da Fase 2 (`scripts/baixar_dados.py`)

**Files:**
- Create: `scripts/baixar_dados.py`, `tests/test_baixar_dados.py`
- Data: `data/gold/*.parquet`, `data/silver/metas.parquet`, `data/silver/resultados_municipio.parquet` (commitados), `data/silver/alunos/` (ignorado)

**Interfaces:**
- Produces: `copiar_lake(origem: Path, destino: Path) -> dict[str, int]` (nome → linhas copiadas). Layout de saída fixo: `gold/<tabela>.parquet`, `silver/metas.parquet`, `silver/resultados_municipio.parquet`, `silver/alunos/ano=YYYY/<arquivo>.parquet`.

- [ ] **Step 1: Teste com um lake falso**

`tests/test_baixar_dados.py`:
```python
import pandas as pd

from scripts.baixar_dados import TABELAS_GOLD, copiar_lake


def _lake_fase2(raiz):
    for t in TABELAS_GOLD:
        (raiz / "gold" / t).mkdir(parents=True)
        pd.DataFrame({"ano": [2024], "x": [1.0]}).to_parquet(raiz / "gold" / t / "data.parquet")
    (raiz / "silver" / "metas").mkdir(parents=True)
    pd.DataFrame({"ano": [2024]}).to_parquet(raiz / "silver" / "metas" / "data.parquet")
    (raiz / "silver" / "resultados" / "municipio").mkdir(parents=True)
    pd.DataFrame({"ano": [2024]}).to_parquet(raiz / "silver" / "resultados" / "municipio" / "data.parquet")
    for ano in (2023, 2024):
        p = raiz / "silver" / "alunos" / f"ano={ano}"
        p.mkdir(parents=True)
        pd.DataFrame({"id_aluno": [1, 2]}).to_parquet(p / "parte-0.parquet")
    return raiz


def test_copia_gold_silver_e_alunos_particionado(tmp_path):
    origem = _lake_fase2(tmp_path / "fase2")
    destino = tmp_path / "novo"
    resumo = copiar_lake(origem, destino)

    assert sorted(p.name for p in (destino / "gold").glob("*.parquet")) == sorted(f"{t}.parquet" for t in TABELAS_GOLD)
    assert (destino / "silver" / "metas.parquet").exists()
    assert (destino / "silver" / "resultados_municipio.parquet").exists()
    assert (destino / "silver" / "alunos" / "ano=2023" / "parte-0.parquet").exists()
    assert resumo["alunos"] == 4
    assert resumo["indicador_municipio"] == 1


def test_falha_claro_se_origem_nao_tem_gold(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="gold"):
        copiar_lake(tmp_path / "vazio", tmp_path / "novo")
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_baixar_dados.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Implementar `scripts/baixar_dados.py`**

```python
"""Copia a Gold e a Silver do pipeline da Fase 2 para data/.

Uso:
    python scripts/baixar_dados.py                    # usa FASE2_LAKE_PATH do .env
    python scripts/baixar_dados.py --origem D:/lake   # pasta data/ do pipeline
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

TABELAS_GOLD = [
    "indicador_municipio", "meta_vs_resultado", "evolucao_temporal",
    "perfil_escola", "distribuicao_proficiencia",
]
TABELAS_SILVER = {
    "metas": Path("silver/metas/data.parquet"),
    "resultados_municipio": Path("silver/resultados/municipio/data.parquet"),
}


def _linhas(caminho: Path) -> int:
    return pq.read_metadata(caminho).num_rows


def copiar_lake(origem: Path, destino: Path) -> dict[str, int]:
    origem, destino = Path(origem), Path(destino)
    if not (origem / "gold").is_dir():
        raise FileNotFoundError(f"não achei a pasta gold em {origem}")

    resumo = {}
    (destino / "gold").mkdir(parents=True, exist_ok=True)
    for t in TABELAS_GOLD:
        alvo = destino / "gold" / f"{t}.parquet"
        shutil.copy(origem / "gold" / t / "data.parquet", alvo)
        resumo[t] = _linhas(alvo)

    (destino / "silver").mkdir(exist_ok=True)
    for nome, rel in TABELAS_SILVER.items():
        alvo = destino / "silver" / f"{nome}.parquet"
        shutil.copy(origem / rel, alvo)
        resumo[nome] = _linhas(alvo)

    alunos_dst = destino / "silver" / "alunos"
    if alunos_dst.exists():
        shutil.rmtree(alunos_dst)
    shutil.copytree(origem / "silver" / "alunos", alunos_dst,
                    ignore=shutil.ignore_patterns("*.crc", "_*"))
    resumo["alunos"] = sum(_linhas(p) for p in alunos_dst.rglob("*.parquet"))
    return resumo


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default=os.environ.get("FASE2_LAKE_PATH", "../pipeline-dados-alfabetiza-brasil/data"))
    ap.add_argument("--destino", default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()
    for nome, n in copiar_lake(args.origem, args.destino).items():
        print(f"{nome:<28}{n:>12,} linhas")


if __name__ == "__main__":
    main()
```

Criar `scripts/__init__.py` vazio para o import nos testes.

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_baixar_dados.py -v`
Expected: 2 passed

- [ ] **Step 5: Executar de verdade e conferir**

A pasta `data/` que existe hoje no repositório é uma cópia integral do lake da Fase 2 (com `bronze/`, streaming, checkpoints e quarentena) no layout `gold/<tabela>/data.parquet`, que não é o layout que este script produz e que o `.gitignore` acima não cobre. Apagar antes de rodar; o script recria só o que o projeto usa:

```bash
rm -rf data
cp .env.example .env
.venv/Scripts/python.exe scripts/baixar_dados.py
ls -la data/gold data/silver
du -sh data/gold data/silver/alunos
```
Expected: 5 arquivos em `data/gold`, 2 em `data/silver`, `alunos/` com `ano=2023` e `ano=2024`; ~3,6 MB na Gold e ~124 MB em alunos. `git status` **não** deve listar `data/silver/alunos`.

- [ ] **Step 6: Commit (dados leves incluídos)**

```bash
git add scripts/__init__.py scripts/baixar_dados.py tests/test_baixar_dados.py data/gold data/silver/metas.parquet data/silver/resultados_municipio.parquet
git commit -m "traz a gold e a silver leve da fase 2; script de copia do lake"
```

---

### Task 3: Leitura dos dados (`src/preprocessing/carregar.py`)

**Files:**
- Create: `src/preprocessing/carregar.py`, `tests/test_carregar.py`

**Interfaces:**
- Consumes: layout de `data/` da Task 2; `config.GOLD/SILVER/EXTERNAL`.
- Produces:
  - `carregar_gold(nome: str) -> pd.DataFrame`
  - `carregar_alunos(ano: int | None = None, apenas_com_nota: bool = True) -> pd.DataFrame` (colunas da Silver + `ano` como int)
  - `carregar_metas() -> pd.DataFrame`
  - `carregar_externa(nome: str) -> pd.DataFrame`

- [ ] **Step 1: Teste**

`tests/test_carregar.py`:
```python
import pandas as pd

from src.preprocessing import carregar
from tests.conftest import montar_alunos


def _grava_alunos(silver, ano, linhas):
    p = silver / "alunos" / f"ano={ano}"
    p.mkdir(parents=True)
    df = montar_alunos(linhas).drop(columns="ano")
    df.to_parquet(p / "parte.parquet")


def test_carregar_alunos_filtra_ano_e_presentes_com_nota(lake_tmp):
    from src import config
    _grava_alunos(config.SILVER, 2023, [{"proficiencia": 700.0}])
    _grava_alunos(config.SILVER, 2024, [
        {"proficiencia": 760.0},
        {"proficiencia": None, "presenca": 0, "preenchimento_caderno": 0, "peso_aluno": None},
    ])
    df = carregar.carregar_alunos(ano=2024)
    assert len(df) == 1
    assert df["ano"].dtype.kind == "i"
    assert df["ano"].iloc[0] == 2024

    todos = carregar.carregar_alunos(ano=2024, apenas_com_nota=False)
    assert len(todos) == 2


def test_carregar_gold_le_tabela_por_nome(lake_tmp):
    from src import config
    pd.DataFrame({"ano": [2023], "id_municipio": [1]}).to_parquet(config.GOLD / "indicador_municipio.parquet")
    df = carregar.carregar_gold("indicador_municipio")
    assert list(df.columns) == ["ano", "id_municipio"]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_carregar.py -v`
Expected: FAIL com `ImportError: cannot import name 'carregar'`

- [ ] **Step 3: Implementar**

`src/preprocessing/carregar.py`:
```python
"""Leitura das bases de entrada (Gold/Silver da Fase 2 e fontes externas)."""
import pandas as pd

from src import config


def carregar_gold(nome: str) -> pd.DataFrame:
    return pd.read_parquet(config.GOLD / f"{nome}.parquet")


def carregar_metas() -> pd.DataFrame:
    return pd.read_parquet(config.SILVER / "metas.parquet")


def carregar_externa(nome: str) -> pd.DataFrame:
    return pd.read_parquet(config.EXTERNAL / f"{nome}.parquet")


def carregar_alunos(ano: int | None = None, apenas_com_nota: bool = True) -> pd.DataFrame:
    filtros = [("ano", "==", ano)] if ano is not None else None
    df = pd.read_parquet(config.SILVER / "alunos", filters=filtros)
    # a partição vem como category; int evita surpresa em groupby e merge
    df["ano"] = df["ano"].astype(int)
    if apenas_com_nota:
        df = df[df["presente"] & ~df["sem_nota"]].copy()
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_carregar.py -v`
Expected: 2 passed

- [ ] **Step 5: Smoke test com dado real**

```bash
.venv/Scripts/python.exe -c "from src.preprocessing.carregar import carregar_alunos as c; d=c(2024); print(len(d), d.alfabetizado.mean().round(3))"
```
Expected: em torno de 1,84 mi linhas e média ≈ 0,598.

- [ ] **Step 6: Commit e PR**

```bash
git add src/preprocessing/carregar.py tests/test_carregar.py
git commit -m "leitura da gold, silver e externas"
git push -u origin feature/setup-e-dados
```
Abrir PR `feature/setup-e-dados` → `develop` com a descrição: por que a Gold é commitada (4,7 MB, reprodutibilidade sem credencial) e a Silver de alunos não (124 MB).

---

### Task 4: Extração das fontes externas (`scripts/extrair_externas.py`)

**Files:**
- Create: `scripts/extrair_externas.py`, `tests/test_extrair_externas.py`
- Data: `data/external/{censo2022_municipio,pib_municipio,populacao_municipio,ideb_municipio,indicadores_municipio,censo_escolar_municipio,diretorios_municipio}.parquet` + `data/external/_metadados.json` (todos commitados)

**Interfaces:**
- Produces: `CONSULTAS: dict[str, dict]` (nome → `{"sql": str, "chave": list[str]}`), `NOMES: list[str]`, `validar(df, chave) -> None` (levanta `ValueError`), `extrair(client, nome, destino: Path) -> dict` (`{"linhas", "bytes_processados"}`), `main(argv=None)`.
- Colunas confirmadas em setembro/2026 por `client.get_table` (metadados, sem custo). `id_municipio` é STRING na Base dos Dados: **sempre** `CAST(... AS INT64)` para casar com a Gold. Valores categóricos confirmados por consulta: Censo Escolar usa códigos string (`rede`: `'2'` estadual, `'3'` municipal; `tipo_localizacao`: `'2'` rural; `tipo_situacao_funcionamento`: `'1'` em atividade); indicadores usam `localizacao = 'Total'` e `rede = 'Pública'`; IDEB usa `rede = 'publica'`, `ensino = 'fundamental'`, `anos_escolares = 'iniciais (1-5)'`. PIB vai até 2023; população até 2025; IDEB tem 2021, 2023 e 2025; indicadores 2021–2024.

- [ ] **Step 1: Branch**

```bash
git checkout -b feature/enriquecimento-externo develop
```

- [ ] **Step 2: Teste com cliente falso (nenhum teste toca o BigQuery)**

`tests/test_extrair_externas.py`:
```python
import pandas as pd
import pytest

from scripts.extrair_externas import CONSULTAS, NOMES, extrair, validar


class _ClienteFalso:
    """Imita client.query(sql).result().to_dataframe() e job.total_bytes_processed."""
    total_bytes_processed = 1234

    def __init__(self, df):
        self._df = df

    def query(self, sql):
        return self

    def result(self, timeout=None):
        return self

    def to_dataframe(self):
        return self._df


def test_sete_fontes_com_cast_do_id_municipio():
    assert sorted(CONSULTAS) == sorted(NOMES) and len(NOMES) == 7
    for nome, c in CONSULTAS.items():
        assert "CAST(id_municipio AS INT64)" in c["sql"], nome
        assert "id_municipio" in c["chave"], nome


def test_extrair_grava_parquet_e_resume(tmp_path):
    df = pd.DataFrame({"id_municipio": [3550308, 3304557], "ano": [2023, 2023], "x": [1.0, 2.0]})
    resumo = extrair(_ClienteFalso(df), "pib_municipio", tmp_path)
    assert (tmp_path / "pib_municipio.parquet").exists()
    assert resumo == {"linhas": 2, "bytes_processados": 1234}


def test_validar_rejeita_id_curto_chave_duplicada_e_vazio():
    with pytest.raises(ValueError, match="7 dígitos"):
        validar(pd.DataFrame({"id_municipio": [355030]}), ["id_municipio"])
    with pytest.raises(ValueError, match="duplicad"):
        validar(pd.DataFrame({"id_municipio": [3550308, 3550308], "ano": [2023, 2023]}), ["id_municipio", "ano"])
    with pytest.raises(ValueError, match="vazia"):
        validar(pd.DataFrame({"id_municipio": []}), ["id_municipio"])
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extrair_externas.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'scripts.extrair_externas'`

- [ ] **Step 4: Implementar**

`scripts/extrair_externas.py`:
```python
"""Extrai as fontes externas da Base dos Dados (BigQuery) já agregadas por município.

As saídas são pequenas e vão para data/external (versionado): quem clona o repo
não precisa de credencial. Só quem for reextrair precisa do .env.

Uso:
    python scripts/extrair_externas.py                 # todas
    python scripts/extrair_externas.py ideb_municipio  # só as citadas
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BD = "basedosdados"
ANO_MIN = 2019          # séries temporais: só o que a regra t-1/t-2 pode usar
ANOS_CENSO_ESCOLAR = (2022, 2023)
TIMEOUT_S = 600

CONSULTAS = {
    "censo2022_municipio": {
        "chave": ["id_municipio"],
        "sql": f"""
            SELECT CAST(id_municipio AS INT64) AS id_municipio, populacao, domicilios, area,
                   taxa_alfabetizacao AS taxa_alfabetizacao_adultos, idade_mediana,
                   indice_envelhecimento, razao_sexo, populacao_indigena, populacao_quilombola
            FROM `{BD}.br_ibge_censo_2022.municipio`
        """,
    },
    "pib_municipio": {
        "chave": ["id_municipio", "ano"],
        "sql": f"""
            SELECT CAST(id_municipio AS INT64) AS id_municipio, ano, pib, impostos_liquidos, va,
                   va_agropecuaria, va_industria, va_servicos, va_adespss
            FROM `{BD}.br_ibge_pib.municipio`
            WHERE ano >= {ANO_MIN}
        """,
    },
    "populacao_municipio": {
        "chave": ["id_municipio", "ano"],
        "sql": f"""
            SELECT CAST(id_municipio AS INT64) AS id_municipio, ano, populacao
            FROM `{BD}.br_ibge_populacao.municipio`
            WHERE ano >= {ANO_MIN}
        """,
    },
    "ideb_municipio": {
        "chave": ["id_municipio", "ano"],
        "sql": f"""
            SELECT CAST(id_municipio AS INT64) AS id_municipio, ano,
                   ideb AS ideb_ai, taxa_aprovacao AS taxa_aprovacao_ideb_ai,
                   indicador_rendimento AS rendimento_ideb_ai,
                   nota_saeb_lingua_portuguesa AS nota_saeb_lp_ai,
                   nota_saeb_matematica AS nota_saeb_mat_ai, projecao AS projecao_ideb_ai
            FROM `{BD}.br_inep_ideb.municipio`
            WHERE ano >= {ANO_MIN} AND rede = 'publica' AND ensino = 'fundamental'
              AND anos_escolares = 'iniciais (1-5)'
        """,
    },
    "indicadores_municipio": {
        "chave": ["id_municipio", "ano"],
        "sql": f"""
            SELECT CAST(id_municipio AS INT64) AS id_municipio, ano,
                   atu_ef_anos_iniciais AS atu_ai, had_ef_anos_iniciais AS had_ai,
                   tdi_ef_anos_iniciais AS tdi_ai,
                   taxa_aprovacao_ef_anos_iniciais AS taxa_aprovacao_ai,
                   taxa_reprovacao_ef_anos_iniciais AS taxa_reprovacao_ai,
                   taxa_abandono_ef_anos_iniciais AS taxa_abandono_ai,
                   dsu_ef_anos_iniciais AS dsu_ai,
                   afd_ef_anos_iniciais_grupo_1 AS afd_ai_grupo1,
                   ied_ef_anos_iniciais_nivel_1 AS ied_ai_nivel1,
                   ird_baixa_regularidade AS ird_baixa
            FROM `{BD}.br_inep_indicadores_educacionais.municipio`
            WHERE ano >= {ANO_MIN} AND localizacao = 'Total' AND rede = 'Pública'
        """,
    },
    "censo_escolar_municipio": {
        "chave": ["id_municipio", "ano"],
        "sql": f"""
            SELECT ano, CAST(id_municipio AS INT64) AS id_municipio,
                   COUNT(*) AS n_escolas_ai,
                   AVG(IF(tipo_localizacao = '2', 1, 0)) AS pct_escolas_rurais,
                   AVG(internet) AS pct_escolas_internet,
                   AVG(GREATEST(IFNULL(biblioteca, 0), IFNULL(biblioteca_sala_leitura, 0))) AS pct_escolas_biblioteca,
                   AVG(esgoto_rede_publica) AS pct_escolas_esgoto_rede,
                   AVG(agua_potavel) AS pct_escolas_agua_potavel,
                   AVG(energia_rede_publica) AS pct_escolas_energia_rede,
                   AVG(laboratorio_informatica) AS pct_escolas_lab_informatica,
                   AVG(quadra_esportes) AS pct_escolas_quadra,
                   AVG(alimentacao) AS pct_escolas_alimentacao,
                   SUM(quantidade_matricula_fundamental_anos_iniciais) AS matriculas_ai,
                   SUM(quantidade_docente_fundamental_anos_iniciais) AS docentes_ai,
                   SAFE_DIVIDE(SUM(quantidade_matricula_fundamental_anos_iniciais_integral),
                               SUM(quantidade_matricula_fundamental_anos_iniciais)) AS pct_matriculas_integral_ai,
                   SAFE_DIVIDE(SUM(quantidade_matricula_fundamental_anos_iniciais),
                               SUM(quantidade_turma_fundamental_anos_iniciais)) AS alunos_por_turma_ai
            FROM `{BD}.br_inep_censo_escolar.escola`
            WHERE ano IN {ANOS_CENSO_ESCOLAR}
              AND tipo_situacao_funcionamento = '1'
              AND rede IN ('2', '3')
              AND etapa_ensino_fundamental_anos_iniciais = 1
            GROUP BY 1, 2
        """,
    },
    "diretorios_municipio": {
        "chave": ["id_municipio"],
        "sql": f"""
            SELECT CAST(id_municipio AS INT64) AS id_municipio, nome AS nome_municipio, sigla_uf,
                   nome_regiao AS regiao, capital_uf, amazonia_legal,
                   ST_Y(centroide) AS latitude, ST_X(centroide) AS longitude
            FROM `{BD}.br_bd_diretorios_brasil.municipio`
        """,
    },
}
NOMES = list(CONSULTAS)


def validar(df: pd.DataFrame, chave: list[str]) -> None:
    if df.empty:
        raise ValueError("consulta voltou vazia")
    if (df["id_municipio"].astype(str).str.len() != 7).any():
        raise ValueError("id_municipio precisa ter 7 dígitos (código IBGE)")
    if df.duplicated(chave).any():
        raise ValueError(f"chave {chave} duplicada")


def extrair(client, nome: str, destino: Path) -> dict:
    cfg = CONSULTAS[nome]
    job = client.query(cfg["sql"])
    df = job.result(timeout=TIMEOUT_S).to_dataframe()
    validar(df, cfg["chave"])
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino / f"{nome}.parquet", index=False)
    return {"linhas": int(len(df)), "bytes_processados": int(job.total_bytes_processed or 0)}


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("nomes", nargs="*", default=NOMES)
    ap.add_argument("--destino", default=Path(__file__).resolve().parents[1] / "data" / "external")
    args = ap.parse_args(argv)

    from google.cloud import bigquery
    client = bigquery.Client(project=os.environ["GCP_PROJECT_ID"])

    meta_path = Path(args.destino) / "_metadados.json"
    metadados = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for nome in args.nomes:
        resumo = extrair(client, nome, args.destino)
        resumo["extraido_em"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        metadados[nome] = resumo
        print(f"{nome:<26}{resumo['linhas']:>9,} linhas  {resumo['bytes_processados']/1e6:>8.1f} MB lidos")
    meta_path.write_text(json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extrair_externas.py -v`
Expected: 3 passed

- [ ] **Step 6: Executar de verdade**

Preencher `.env` com `GOOGLE_APPLICATION_CREDENTIALS` e `GCP_PROJECT_ID` (os mesmos da Fase 2) e rodar:

```bash
.venv/Scripts/python.exe scripts/extrair_externas.py
ls -la data/external
```
Expected (ordem de grandeza): censo2022 5.570 · pib ≈ 27,8 mil (5 anos) · populacao ≈ 39 mil · ideb ≈ 22 mil · indicadores ≈ 33 mil · censo_escolar ≈ 11 mil (2 anos) · diretorios 5.571. Total lido abaixo de 1 GB (o Censo Escolar é o maior: duas partições, só as colunas selecionadas). Nenhum parquet acima de 2 MB.

- [ ] **Step 7: Conferir cobertura contra a Gold**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
g = set(pd.read_parquet('data/gold/indicador_municipio.parquet').id_municipio)
for n in ['censo2022_municipio','pib_municipio','ideb_municipio','indicadores_municipio','censo_escolar_municipio','diretorios_municipio']:
    e = set(pd.read_parquet(f'data/external/{n}.parquet').id_municipio)
    print(f'{n:<26} cobre {len(g & e)/len(g):.1%} dos municípios da Gold')
"
```
Expected: ≥ 99% em todas (IDEB pode ficar em ~97%: municípios sem escola pública com 5º ano avaliado; vira NaN + indicador de faltante).

- [ ] **Step 8: Commit**

```bash
git add scripts/extrair_externas.py tests/test_extrair_externas.py data/external
git commit -m "extrai fontes externas do bigquery agregadas por municipio"
```

---

### Task 5: Contexto externo por município (`src/preprocessing/externas.py`)

**Files:**
- Create: `src/preprocessing/externas.py`, `tests/test_externas.py`

**Interfaces:**
- Consumes: `carregar_externa(nome)` (Task 3) e os 7 parquets da Task 4.
- Produces:
  - `DEFASAGEM: dict[str, int]` — `{"pib_municipio": 2, "populacao_municipio": 1, "ideb_municipio": 1, "indicadores_municipio": 1, "censo_escolar_municipio": 1}`. Regra: feature do ano `t` usa o **último ano disponível ≤ t − defasagem**. PIB municipal de `t−1` só é publicado em dezembro de `t+1`, por isso 2. IDEB é bienal: para 2024 usa 2023 (publicado em agosto de 2024, antes da prova de novembro); para 2023 usa 2021.
  - `ano_referencia(fonte: str, ano: int, anos_disponiveis) -> int`
  - `montar_contexto_externo(ano: int, fontes: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame` — uma linha por `id_municipio` (base: `diretorios_municipio`, todos os 5.571). `fontes=None` lê de `data/external`.
  - `anos_referencia(ano: int, fontes=None) -> dict[str, int]` — para o dicionário.

- [ ] **Step 1: Teste**

`tests/test_externas.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import externas


def _fontes():
    ids = [1100015, 1100023]
    return {
        "diretorios_municipio": pd.DataFrame({"id_municipio": ids, "nome_municipio": ["A", "B"], "sigla_uf": ["RO", "RO"],
                                              "regiao": ["Norte", "Norte"], "capital_uf": [0, 1], "amazonia_legal": [1, 1],
                                              "latitude": [-11.9, -12.7], "longitude": [-61.9, -60.1]}),
        "censo2022_municipio": pd.DataFrame({"id_municipio": ids, "populacao": [20000, 80000], "domicilios": [7000, 30000],
                                             "area": [4000, 8000], "taxa_alfabetizacao_adultos": [90.0, 95.0],
                                             "idade_mediana": [30.0, 32.0], "indice_envelhecimento": [40.0, 50.0],
                                             "razao_sexo": [100.0, 98.0], "populacao_indigena": [200, 0],
                                             "populacao_quilombola": [0, 800]}),
        "pib_municipio": pd.DataFrame({"id_municipio": ids * 3, "ano": [2021, 2021, 2022, 2022, 2023, 2023],
                                       "pib": [100, 400, 110, 440, 120, 480], "impostos_liquidos": [10] * 6, "va": [100, 400, 100, 400, 100, 400],
                                       "va_agropecuaria": [50, 40, 50, 40, 50, 40], "va_industria": [10, 100, 10, 100, 10, 100],
                                       "va_servicos": [20, 200, 20, 200, 20, 200], "va_adespss": [20, 60, 20, 60, 20, 60]}),
        "populacao_municipio": pd.DataFrame({"id_municipio": ids * 3, "ano": [2021, 2021, 2022, 2022, 2023, 2023],
                                             "populacao": [20000, 80000, 20500, 80500, 21000, 81000]}),
        "ideb_municipio": pd.DataFrame({"id_municipio": [1100015, 1100015], "ano": [2021, 2023], "ideb_ai": [5.0, 5.5],
                                        "taxa_aprovacao_ideb_ai": [0.9, 0.95], "rendimento_ideb_ai": [0.9, 0.95],
                                        "nota_saeb_lp_ai": [200.0, 210.0], "nota_saeb_mat_ai": [205.0, 215.0], "projecao_ideb_ai": [5.2, 5.6]}),
        "indicadores_municipio": pd.DataFrame({"id_municipio": ids * 2, "ano": [2022, 2022, 2023, 2023], "atu_ai": [20.0, 25.0, 21.0, 26.0],
                                               "had_ai": [4.5] * 4, "tdi_ai": [10.0, 5.0, 9.0, 4.0], "taxa_aprovacao_ai": [90.0] * 4,
                                               "taxa_reprovacao_ai": [5.0] * 4, "taxa_abandono_ai": [1.0] * 4, "dsu_ai": [80.0] * 4,
                                               "afd_ai_grupo1": [60.0] * 4, "ied_ai_nivel1": [10.0] * 4, "ird_baixa": [5.0] * 4}),
        "censo_escolar_municipio": pd.DataFrame({"id_municipio": ids * 2, "ano": [2022, 2022, 2023, 2023], "n_escolas_ai": [10, 30, 11, 31],
                                                 "pct_escolas_rurais": [0.5, 0.1] * 2, "pct_escolas_internet": [0.6, 0.9] * 2,
                                                 "pct_escolas_biblioteca": [0.3, 0.7] * 2, "pct_escolas_esgoto_rede": [0.2, 0.8] * 2,
                                                 "pct_escolas_agua_potavel": [0.9, 1.0] * 2, "pct_escolas_energia_rede": [1.0, 1.0] * 2,
                                                 "pct_escolas_lab_informatica": [0.2, 0.5] * 2, "pct_escolas_quadra": [0.3, 0.6] * 2,
                                                 "pct_escolas_alimentacao": [1.0, 1.0] * 2, "matriculas_ai": [1000, 4000, 1100, 4100],
                                                 "docentes_ai": [50, 200, 55, 205], "pct_matriculas_integral_ai": [0.1, 0.3] * 2,
                                                 "alunos_por_turma_ai": [22.0, 26.0] * 2}),
    }


def test_regra_temporal_por_fonte():
    f = _fontes()
    assert externas.ano_referencia("pib_municipio", 2024, f["pib_municipio"]["ano"]) == 2022
    assert externas.ano_referencia("ideb_municipio", 2024, f["ideb_municipio"]["ano"]) == 2023
    assert externas.ano_referencia("ideb_municipio", 2023, f["ideb_municipio"]["ano"]) == 2021
    assert externas.ano_referencia("indicadores_municipio", 2024, f["indicadores_municipio"]["ano"]) == 2023
    with pytest.raises(ValueError):
        externas.ano_referencia("pib_municipio", 2020, f["pib_municipio"]["ano"])


def test_uma_linha_por_municipio_com_derivadas_e_faltantes():
    ctx = externas.montar_contexto_externo(2024, _fontes())
    assert len(ctx) == 2 and ctx["id_municipio"].is_unique
    a = ctx.set_index("id_municipio").loc[1100015]
    assert a["pib_per_capita"] == pytest.approx(110 / 20500)        # pib 2022 / população 2022
    assert a["log_pib_per_capita"] == pytest.approx(np.log1p(110 / 20500))
    assert a["densidade_demografica"] == pytest.approx(20000 / 4000)
    assert a["pct_va_agropecuaria"] == pytest.approx(0.5)
    assert a["pct_pop_indigena"] == pytest.approx(0.01)
    assert a["ideb_ai"] == 5.5 and a["tdi_ai"] == 9.0 and a["n_escolas_ai"] == 11
    assert a["alunos_por_docente_ai"] == pytest.approx(1100 / 55)
    assert a["log_populacao"] == pytest.approx(np.log1p(21000))     # população 2023
    b = ctx.set_index("id_municipio").loc[1100023]
    assert np.isnan(b["ideb_ai"])                                   # sem IDEB: NaN, linha permanece
    assert "ano" not in ctx.columns and "nome_municipio" in ctx.columns


def test_anos_referencia_para_o_dicionario():
    assert externas.anos_referencia(2024, _fontes()) == {
        "pib_municipio": 2022, "populacao_municipio": 2023, "ideb_municipio": 2023,
        "indicadores_municipio": 2023, "censo_escolar_municipio": 2023,
    }
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_externas.py -v`
Expected: FAIL com `ImportError: cannot import name 'externas'`

- [ ] **Step 3: Implementar**

`src/preprocessing/externas.py`:
```python
"""Contexto externo por município (IBGE, INEP, diretórios), com regra temporal por fonte.

Feature do ano t usa o último ano disponível <= t - defasagem. Censo 2022 e
diretórios são estruturais (sem ano).
"""
import numpy as np
import pandas as pd

from src.preprocessing.carregar import carregar_externa

ESTRUTURAIS = ["diretorios_municipio", "censo2022_municipio"]
DEFASAGEM = {
    "pib_municipio": 2,
    "populacao_municipio": 1,
    "ideb_municipio": 1,
    "indicadores_municipio": 1,
    "censo_escolar_municipio": 1,
}
NOMES = ESTRUTURAIS + list(DEFASAGEM)
SETORES = ["agropecuaria", "industria", "servicos", "adespss"]


def ano_referencia(fonte: str, ano: int, anos_disponiveis) -> int:
    limite = ano - DEFASAGEM[fonte]
    candidatos = [int(a) for a in pd.unique(pd.Series(anos_disponiveis)) if a <= limite]
    if not candidatos:
        raise ValueError(f"{fonte}: nenhum ano <= {limite} disponível")
    return max(candidatos)


def _carregar_todas() -> dict[str, pd.DataFrame]:
    return {n: carregar_externa(n) for n in NOMES}


def anos_referencia(ano: int, fontes: dict | None = None) -> dict[str, int]:
    fontes = fontes or _carregar_todas()
    return {f: ano_referencia(f, ano, fontes[f]["ano"]) for f in DEFASAGEM}


def _no_ano(fontes, nome, ano):
    df = fontes[nome]
    ref = ano_referencia(nome, ano, df["ano"])
    return df[df["ano"] == ref].drop(columns="ano").copy()


def montar_contexto_externo(ano: int, fontes: dict | None = None) -> pd.DataFrame:
    fontes = fontes or _carregar_todas()

    ctx = fontes["diretorios_municipio"].copy()

    c22 = fontes["censo2022_municipio"].rename(columns={"populacao": "pop_2022", "domicilios": "domicilios_2022", "area": "area_km2"})
    c22["densidade_demografica"] = c22["pop_2022"] / c22["area_km2"]
    c22["pct_pop_indigena"] = c22["populacao_indigena"] / c22["pop_2022"]
    c22["pct_pop_quilombola"] = c22["populacao_quilombola"] / c22["pop_2022"]
    ctx = ctx.merge(c22.drop(columns=["populacao_indigena", "populacao_quilombola"]), on="id_municipio", how="left")

    # PIB per capita usa a população do mesmo ano do PIB (t-2), não a de t-1
    pib = _no_ano(fontes, "pib_municipio", ano)
    ano_pib = ano_referencia("pib_municipio", ano, fontes["pib_municipio"]["ano"])
    pop_pib = fontes["populacao_municipio"].query("ano == @ano_pib")[["id_municipio", "populacao"]]
    pib = pib.merge(pop_pib, on="id_municipio", how="left")
    pib["pib_per_capita"] = pib["pib"] / pib["populacao"]
    pib["log_pib_per_capita"] = np.log1p(pib["pib_per_capita"])
    for s in SETORES:
        pib[f"pct_va_{s}"] = pib[f"va_{s}"] / pib["va"]
    ctx = ctx.merge(pib[["id_municipio", "pib_per_capita", "log_pib_per_capita"] + [f"pct_va_{s}" for s in SETORES]],
                    on="id_municipio", how="left")

    pop = _no_ano(fontes, "populacao_municipio", ano)
    pop["log_populacao"] = np.log1p(pop["populacao"])
    ctx = ctx.merge(pop, on="id_municipio", how="left")

    ctx = ctx.merge(_no_ano(fontes, "ideb_municipio", ano), on="id_municipio", how="left")
    ctx = ctx.merge(_no_ano(fontes, "indicadores_municipio", ano), on="id_municipio", how="left")

    ce = _no_ano(fontes, "censo_escolar_municipio", ano)
    ce["alunos_por_docente_ai"] = ce["matriculas_ai"] / ce["docentes_ai"]
    ctx = ctx.merge(ce, on="id_municipio", how="left")

    assert ctx["id_municipio"].is_unique
    return ctx.reset_index(drop=True)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_externas.py -v`
Expected: 3 passed

- [ ] **Step 5: Smoke test com dado real e mapa de faltantes**

```bash
.venv/Scripts/python.exe -c "
from src.preprocessing.externas import montar_contexto_externo, anos_referencia
c = montar_contexto_externo(2024); print(c.shape); print(anos_referencia(2024))
print(c.isna().mean().sort_values(ascending=False).head(8).round(3))
"
```
Expected: 5.571 linhas × ~50 colunas; `anos_referencia` = pib 2022, populacao 2023, ideb 2023, indicadores 2023, censo_escolar 2023. Faltantes concentrados no IDEB (poucos %). Anotar as taxas de faltante: entram no dicionário (Task 7) e na EDA (Task 18).

- [ ] **Step 6: Commit e PR**

```bash
git add src/preprocessing/externas.py tests/test_externas.py
git commit -m "contexto externo por municipio com regra temporal por fonte"
git push -u origin feature/enriquecimento-externo
```
PR `feature/enriquecimento-externo` → `develop`. Descrição: tabela fonte × colunas × defasagem × justificativa (data de publicação), decisão de agregar o Censo Escolar no BigQuery (por que não juntar por escola: `id_escola` pseudônimo), FUNDEB e INSE fora.

---

### Task 6: Contexto municipal defasado e leave-one-out da escola (`src/preprocessing/contexto.py`)

**Files:**
- Create: `src/preprocessing/contexto.py`, `tests/test_contexto.py`

**Interfaces:**
- Consumes: Gold `indicador_municipio`, `distribuicao_proficiencia` (grão `nivel == "municipio"`; `rede == "total"` para o aluno, `rede == "municipal"` para o modelo B), Silver `metas` (`nivel == "municipio"`, `rede_padronizada == "municipal"`), Silver de alunos presentes com nota.
- Produces:
  - `contexto_municipal_defasado(ano_alvo, indicador, distribuicao, rede="total") -> pd.DataFrame` — uma linha por `id_municipio` que **existe em `ano_alvo − 1`**, todas as colunas com sufixo `_mun_t1`. Municípios sem linha em t−1 não aparecem (o `feature_store` marca `sem_historico`).
  - `meta_pactuada(ano_alvo, metas) -> pd.DataFrame[id_municipio, meta_alvo]` — `meta_alfabetizacao_{ano_alvo}`; NaN quando não há.
  - `contexto_escola_loo(alunos) -> pd.DataFrame` — mesmo índice de `alunos`; `taxa_escola_loo`, `prof_media_escola_loo`, `n_alunos_escola` (outros alunos com nota na escola). Escola com um aluno → NaN.
  - `participacao_escola_loo(alunos, perfil_escola) -> pd.Series` — `(alunos_presentes − 1) / (alunos_avaliados − 1)` da escola no mesmo ano, alinhada ao índice de `alunos`.

- [ ] **Step 1: Branch**

```bash
git checkout -b feature/feature-store develop
```

- [ ] **Step 2: Teste**

`tests/test_contexto.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import contexto
from tests.conftest import montar_alunos


def test_loo_exclui_o_proprio_aluno_e_da_nan_para_escola_solitaria():
    alunos = montar_alunos([
        {"id_escola": 1, "proficiencia": 700.0},
        {"id_escola": 1, "proficiencia": 750.0},
        {"id_escola": 1, "proficiencia": 800.0},
        {"id_escola": 2, "proficiencia": 760.0},
    ])
    loo = contexto.contexto_escola_loo(alunos)
    assert list(loo.columns) == ["taxa_escola_loo", "prof_media_escola_loo", "n_alunos_escola"]
    assert loo.loc[0, "prof_media_escola_loo"] == pytest.approx(775.0)
    assert loo.loc[0, "taxa_escola_loo"] == pytest.approx(1.0)      # 750 e 800 >= 743
    assert loo.loc[1, "taxa_escola_loo"] == pytest.approx(0.5)      # 700 e 800
    assert loo.loc[0, "n_alunos_escola"] == 2
    assert np.isnan(loo.loc[3, "prof_media_escola_loo"]) and loo.loc[3, "n_alunos_escola"] == 0


def test_participacao_loo_desconta_o_aluno():
    alunos = montar_alunos([{"id_escola": 1, "proficiencia": 750.0}, {"id_escola": 1, "proficiencia": 700.0}])
    perfil = pd.DataFrame({"ano": [2024], "id_escola": [1], "alunos_avaliados": [11], "alunos_presentes": [10]})
    p = contexto.participacao_escola_loo(alunos, perfil)
    assert p.tolist() == pytest.approx([0.9, 0.9])
    assert p.name == "taxa_participacao_escola_loo"


def test_contexto_municipal_usa_apenas_o_ano_anterior():
    indicador = pd.DataFrame({
        "ano": [2023, 2024, 2024], "id_municipio": [1, 1, 2], "sigla_uf": ["SP"] * 3,
        "alunos_avaliados": [100, 120, 50], "alunos_presentes": [90, 110, 45], "alunos_com_nota": [90, 110, 45],
        "taxa_participacao": [0.9, 0.92, 0.9], "taxa_alfabetizacao": [0.6, 0.7, 0.5], "ic95": [0.05, 0.04, 0.1],
        "taxa_limite_inferior": [0.55, 0.66, 0.4], "taxa_limite_superior": [0.65, 0.74, 0.6],
        "proficiencia_media": [750.0, 760.0, 740.0], "criancas_nao_alfabetizadas": [40.0, 36.0, 25.0],
        "alerta_participacao": [False, False, True],
    })
    dist = pd.DataFrame({
        "ano": [2023, 2023, 2024], "nivel": ["municipio"] * 3, "rede": ["total", "municipal", "total"],
        "sigla_uf": ["SP"] * 3, "id_municipio": [1, 1, 1], "alunos_com_nota": [90, 80, 110],
        **{f"pct_nivel_{i}": [0.1, 0.2, 0.3] for i in range(9)},
        "pct_critico": [0.2, 0.3, 0.1], "pct_atencao": [0.2, 0.2, 0.2], "pct_alfabetizado": [0.6, 0.5, 0.7], "pct_quase_la": [0.1, 0.1, 0.1],
    })
    ctx = contexto.contexto_municipal_defasado(2024, indicador, dist)
    assert ctx["id_municipio"].tolist() == [1]                      # município 2 não tem 2023
    assert ctx.loc[0, "taxa_alfabetizacao_mun_t1"] == 0.6
    assert ctx.loc[0, "pct_critico_mun_t1"] == 0.2                  # rede total, não municipal
    assert "ano" not in ctx.columns and "sigla_uf" not in ctx.columns
    assert all(c.endswith("_mun_t1") or c == "id_municipio" for c in ctx.columns)


def test_meta_pactuada_do_ano_alvo():
    metas = pd.DataFrame({
        "ano": [2024, 2024, 2024], "nivel": ["municipio", "municipio", "uf"], "rede_padronizada": ["municipal"] * 3,
        "id_municipio": [1, 2, None], "meta_alfabetizacao_2024": [0.55, None, 0.6], "meta_alfabetizacao_2025": [0.6, 0.62, 0.65],
    })
    m = contexto.meta_pactuada(2024, metas).set_index("id_municipio")["meta_alvo"]
    assert len(m) == 2 and m[1] == 0.55 and np.isnan(m[2])
    assert contexto.meta_pactuada(2025, metas).set_index("id_municipio").loc[2, "meta_alvo"] == 0.62
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_contexto.py -v`
Expected: FAIL com `ImportError: cannot import name 'contexto'`

- [ ] **Step 4: Implementar**

`src/preprocessing/contexto.py`:
```python
"""Contexto que o aluno recebe sem vazar o alvo: município em t-1 e escola com leave-one-out."""
import numpy as np
import pandas as pd

COLS_INDICADOR = [
    "taxa_alfabetizacao", "ic95", "taxa_participacao", "proficiencia_media", "alunos_avaliados",
    "criancas_nao_alfabetizadas", "taxa_limite_inferior", "taxa_limite_superior", "alerta_participacao",
]
COLS_DISTRIBUICAO = [f"pct_nivel_{i}" for i in range(9)] + ["pct_critico", "pct_atencao", "pct_quase_la"]


def contexto_municipal_defasado(ano_alvo: int, indicador: pd.DataFrame, distribuicao: pd.DataFrame,
                                rede: str = "total") -> pd.DataFrame:
    t1 = ano_alvo - 1
    ind = indicador.loc[indicador["ano"] == t1, ["id_municipio"] + COLS_INDICADOR]
    dist = distribuicao.loc[
        (distribuicao["ano"] == t1) & (distribuicao["nivel"] == "municipio") & (distribuicao["rede"] == rede),
        ["id_municipio"] + COLS_DISTRIBUICAO,
    ]
    ctx = ind.merge(dist, on="id_municipio", how="left")
    ctx["alerta_participacao"] = ctx["alerta_participacao"].astype(float)
    ctx = ctx.rename(columns={c: f"{c}_mun_t1" for c in ctx.columns if c != "id_municipio"})
    assert ctx["id_municipio"].is_unique
    return ctx.reset_index(drop=True)


def meta_pactuada(ano_alvo: int, metas: pd.DataFrame) -> pd.DataFrame:
    col = f"meta_alfabetizacao_{ano_alvo}"
    m = metas[(metas["nivel"] == "municipio") & (metas["rede_padronizada"] == "municipal")]
    # a meta é pactuada uma vez; qualquer linha do município serve, a mais recente por garantia
    m = m.sort_values("ano").groupby("id_municipio", as_index=False)[col].last()
    m["id_municipio"] = m["id_municipio"].astype(int)
    return m.rename(columns={col: "meta_alvo"})


def contexto_escola_loo(alunos: pd.DataFrame) -> pd.DataFrame:
    g = alunos.groupby("id_escola")
    n = g["proficiencia"].transform("count")
    soma_prof = g["proficiencia"].transform("sum")
    soma_alf = g["alfabetizado"].transform("sum")
    n_outros = n - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "taxa_escola_loo": (soma_alf - alunos["alfabetizado"]) / n_outros,
            "prof_media_escola_loo": (soma_prof - alunos["proficiencia"]) / n_outros,
            "n_alunos_escola": n_outros,
        }, index=alunos.index)
    out.loc[n_outros == 0, ["taxa_escola_loo", "prof_media_escola_loo"]] = np.nan
    return out


def participacao_escola_loo(alunos: pd.DataFrame, perfil_escola: pd.DataFrame) -> pd.Series:
    ano = int(alunos["ano"].iloc[0])
    p = perfil_escola.loc[perfil_escola["ano"] == ano].set_index("id_escola")
    aval = alunos["id_escola"].map(p["alunos_avaliados"]) - 1
    pres = alunos["id_escola"].map(p["alunos_presentes"]) - 1
    return (pres / aval.replace(0, np.nan)).rename("taxa_participacao_escola_loo")
```

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_contexto.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing/contexto.py tests/test_contexto.py
git commit -m "contexto municipal t-1 e leave-one-out da escola"
```

---

### Task 7: Feature store (`src/preprocessing/feature_store.py`) e dicionário

**Files:**
- Create: `src/preprocessing/feature_store.py`, `tests/test_feature_store.py`, `docs/dicionario_base_modelagem.md`
- Modify: `tests/conftest.py` (fixture `lake_minimo`)
- Data: `data/processed/base_modelagem_aluno.parquet`, `data/processed/base_modelagem_municipio.parquet` (não commitados)

**Interfaces:**
- Produces:
  - `montar_base_aluno(ano_alvo=config.ANO_ALVO) -> pd.DataFrame` — uma linha por aluno presente com nota em `ano_alvo`. Colunas: as da Silver (mantidas para split, pesos e métricas; `features.py` decide o que entra), `sem_historico` (bool), contexto `_mun_t1`, `meta_alvo`, externas (Task 5), `taxa_escola_loo`, `prof_media_escola_loo`, `n_alunos_escola`, `taxa_participacao_escola_loo`.
  - `montar_base_municipio(anos=(2023, 2024)) -> pd.DataFrame` — grão `(ano, id_municipio)`, rede **municipal**: `taxa_alfabetizacao`, `ic95`, `alunos_com_nota`, `meta_ano`, `gap`, `situacao_meta` (de `meta_vs_resultado`), `taxa_participacao`, `proficiencia_media`, `criancas_nao_alfabetizadas` (de `evolucao_temporal`), `pct_nivel_*`, `pct_critico`, `pct_atencao`, `pct_quase_la` (de `distribuicao_proficiencia`, rede municipal), externas do ano, `meta_prox` (`meta_alfabetizacao_{ano+1}`), e os **alvos do ano seguinte**: `taxa_prox`, `situacao_meta_prox`, `nao_atingiu_prox` (NaN em 2024).
  - CLI: `python -m src.preprocessing.feature_store [--ano 2024]` grava as duas bases em `config.PROCESSED`.

- [ ] **Step 1: Fixture de lake mínimo em `tests/conftest.py`**

Acrescentar `import numpy as np` no topo e, ao final de `tests/conftest.py`:
```python
def _gravar(df: pd.DataFrame, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(caminho, index=False)


@pytest.fixture
def lake_minimo(lake_tmp):
    """Gold, Silver e externas com 3 municípios (1 e 2 com 2023; 3 só em 2024), 2 escolas por município."""
    from tests.test_externas import _fontes
    anos = [2023, 2024]
    muns = [1100015, 1100023, 1100031]

    def linha_ind(ano, m, taxa):
        return {"ano": ano, "id_municipio": m, "sigla_uf": "RO", "alunos_avaliados": 40, "alunos_presentes": 36,
                "alunos_com_nota": 36, "taxa_participacao": 0.9, "taxa_alfabetizacao": taxa, "ic95": 0.08,
                "taxa_limite_inferior": taxa - 0.08, "taxa_limite_superior": taxa + 0.08, "proficiencia_media": 700 + 100 * taxa,
                "criancas_nao_alfabetizadas": 36 * (1 - taxa), "alerta_participacao": False}
    ind = [linha_ind(2023, m, t) for m, t in zip(muns[:2], [0.5, 0.7])] + [linha_ind(2024, m, t) for m, t in zip(muns, [0.55, 0.72, 0.4])]
    _gravar(pd.DataFrame(ind), config.GOLD / "indicador_municipio.parquet")

    dist, mvr, evo = [], [], []
    for ano in anos:
        for m in (muns if ano == 2024 else muns[:2]):
            for rede in ["total", "municipal"]:
                dist.append({"ano": ano, "nivel": "municipio", "rede": rede, "sigla_uf": "RO", "id_municipio": m, "alunos_com_nota": 36,
                             **{f"pct_nivel_{i}": 1 / 9 for i in range(9)}, "pct_critico": 0.2, "pct_atencao": 0.2,
                             "pct_alfabetizado": 0.6, "pct_quase_la": 0.1})
            taxa = 0.5 + 0.1 * (ano - 2023) + 0.05 * muns.index(m)
            mvr.append({"ano": ano, "nivel": "municipio", "rede": "municipal", "sigla_uf": "RO", "id_municipio": m, "alunos_com_nota": 36,
                        "taxa_alfabetizacao": taxa, "ic95": 0.08, "meta_ano": 0.6 if ano == 2024 else None, "gap": taxa - 0.6,
                        "atingiu_meta": taxa >= 0.6, "situacao_meta": "sem_meta" if ano == 2023 else ("atingiu" if taxa >= 0.6 else "nao_atingiu")})
            evo.append({"ano": ano, "nivel": "municipio", "rede": "municipal", "sigla_uf": "RO", "id_municipio": m, "alunos_avaliados": 40,
                        "alunos_presentes": 36, "alunos_com_nota": 36, "taxa_participacao": 0.9, "taxa_alfabetizacao": taxa, "ic95": 0.08,
                        "proficiencia_media": 750.0, "criancas_nao_alfabetizadas": 36 * (1 - taxa)})
    _gravar(pd.DataFrame(dist), config.GOLD / "distribuicao_proficiencia.parquet")
    _gravar(pd.DataFrame(mvr), config.GOLD / "meta_vs_resultado.parquet")
    _gravar(pd.DataFrame(evo), config.GOLD / "evolucao_temporal.parquet")

    escolas = {m: [m * 10 + 1, m * 10 + 2] for m in muns}
    perfil = [{"ano": ano, "id_escola": e, "id_municipio": m, "sigla_uf": "RO", "rede": "municipal", "alunos_avaliados": 6,
               "alunos_presentes": 5, "alunos_com_nota": 5, "taxa_participacao": 5 / 6, "taxa_alfabetizacao": 0.6, "ic95": 0.2,
               "proficiencia_media": 750.0, "taxa_municipio": 0.6, "residuo": 0.0}
              for ano in anos for m in (muns if ano == 2024 else muns[:2]) for e in escolas[m]]
    _gravar(pd.DataFrame(perfil), config.GOLD / "perfil_escola.parquet")

    metas = [{"ano": 2024, "nivel": "municipio", "rede_padronizada": "municipal", "id_municipio": m, "sigla_uf": None,
              "meta_alfabetizacao_2024": 0.6, "meta_alfabetizacao_2025": 0.65, "meta_alfabetizacao_2030": 0.8} for m in muns]
    _gravar(pd.DataFrame(metas), config.SILVER / "metas.parquet")

    rng = np.random.default_rng(0)
    for ano in anos:
        linhas = []
        for m in (muns if ano == 2024 else muns[:2]):
            for e in escolas[m]:
                for _ in range(5):
                    linhas.append({"id_municipio": m, "id_escola": e, "sigla_uf": "RO", "proficiencia": float(rng.normal(750, 40))})
                linhas.append({"id_municipio": m, "id_escola": e, "sigla_uf": "RO", "proficiencia": None,
                               "presenca": 0, "preenchimento_caderno": 0, "peso_aluno": None})
        _gravar(montar_alunos(linhas).drop(columns="ano"), config.SILVER / "alunos" / f"ano={ano}" / "parte.parquet")

    for nome, df in _fontes().items():
        extra = df[df["id_municipio"] == 1100015].assign(id_municipio=1100031)   # terceiro município copia o primeiro
        _gravar(pd.concat([df, extra], ignore_index=True), config.EXTERNAL / f"{nome}.parquet")
    return lake_tmp
```

- [ ] **Step 2: Teste**

`tests/test_feature_store.py`:
```python
import numpy as np
import pandas as pd

from src import config
from src.preprocessing import feature_store


def test_base_aluno_tem_uma_linha_por_aluno_com_nota_e_flags(lake_minimo):
    base = feature_store.montar_base_aluno(2024)
    assert len(base) == 30                                   # 3 municípios × 2 escolas × 5 com nota
    assert base["id_aluno"].is_unique
    assert base["sem_historico"].sum() == 10                 # município 3 não tem 2023
    assert base.loc[base["sem_historico"], "taxa_alfabetizacao_mun_t1"].isna().all()
    assert base.loc[~base["sem_historico"], "taxa_alfabetizacao_mun_t1"].isin([0.5, 0.7]).all()
    assert base["meta_alvo"].eq(0.6).all()
    assert base["regiao"].eq("Norte").all()
    assert np.allclose(base["taxa_participacao_escola_loo"], 4 / 5)
    assert base["n_alunos_escola"].eq(4).all()
    for c in ["taxa_escola_loo", "prof_media_escola_loo", "log_pib_per_capita", "tdi_ai", "pct_escolas_rurais", "sigla_uf"]:
        assert c in base.columns
    assert not [c for c in base.columns if c.endswith("_x") or c.endswith("_y")]


def test_base_aluno_nao_carrega_gold_do_proprio_ano(lake_minimo):
    base = feature_store.montar_base_aluno(2024)
    assert "taxa_alfabetizacao" not in base.columns
    assert "pct_critico" not in base.columns                 # só a versão _mun_t1


def test_base_municipio_alvo_do_ano_seguinte(lake_minimo):
    b = feature_store.montar_base_municipio()
    assert sorted(b["ano"].unique()) == [2023, 2024]
    assert len(b) == 5 and not b.duplicated(["ano", "id_municipio"]).any()
    l23 = b[(b.ano == 2023) & (b.id_municipio == 1100015)].iloc[0]
    l24 = b[(b.ano == 2024) & (b.id_municipio == 1100015)].iloc[0]
    assert l23["taxa_prox"] == l24["taxa_alfabetizacao"]
    assert l23["meta_prox"] == 0.6 and l24["meta_prox"] == 0.65
    assert np.isnan(l24["taxa_prox"]) and pd.isna(l24["situacao_meta_prox"]) and np.isnan(l24["nao_atingiu_prox"])
    assert l23["nao_atingiu_prox"] in (0.0, 1.0)
    assert "pct_nivel_0" in b.columns and "log_pib_per_capita" in b.columns


def test_cli_grava_as_duas_bases(lake_minimo):
    feature_store.main(["--ano", "2024"])
    assert (config.PROCESSED / "base_modelagem_aluno.parquet").exists()
    assert (config.PROCESSED / "base_modelagem_municipio.parquet").exists()
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_feature_store.py -v`
Expected: FAIL com `ImportError: cannot import name 'feature_store'`

- [ ] **Step 4: Implementar**

`src/preprocessing/feature_store.py`:
```python
"""Monta as bases de modelagem (feature store) a partir da Gold/Silver da Fase 2 e das externas.

    python -m src.preprocessing.feature_store            # grava as duas bases em data/processed
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.preprocessing.carregar import carregar_alunos, carregar_gold, carregar_metas
from src.preprocessing.contexto import (COLS_DISTRIBUICAO, contexto_escola_loo,
                                        contexto_municipal_defasado, meta_pactuada,
                                        participacao_escola_loo)
from src.preprocessing.externas import montar_contexto_externo

# a Silver e a Gold já trazem sigla_uf; tirar da externa evita sufixo _x/_y no merge
COLS_EXTERNAS_REPETIDAS = ["sigla_uf"]


def montar_base_aluno(ano_alvo: int = config.ANO_ALVO) -> pd.DataFrame:
    alunos = carregar_alunos(ano_alvo)
    perfil = carregar_gold("perfil_escola")

    base = pd.concat([alunos, contexto_escola_loo(alunos), participacao_escola_loo(alunos, perfil)], axis=1)

    ctx = contexto_municipal_defasado(ano_alvo, carregar_gold("indicador_municipio"), carregar_gold("distribuicao_proficiencia"))
    base = base.merge(ctx, on="id_municipio", how="left")
    base["sem_historico"] = base["taxa_alfabetizacao_mun_t1"].isna()

    base = base.merge(meta_pactuada(ano_alvo, carregar_metas()), on="id_municipio", how="left")

    ext = montar_contexto_externo(ano_alvo).drop(columns=COLS_EXTERNAS_REPETIDAS)
    base = base.merge(ext, on="id_municipio", how="left")
    return base.reset_index(drop=True)


def _municipio_no_ano(ano: int) -> pd.DataFrame:
    filtro = "ano == @ano and nivel == 'municipio' and rede == 'municipal'"
    mvr = carregar_gold("meta_vs_resultado").query(filtro)
    evo = carregar_gold("evolucao_temporal").query(filtro)
    dist = carregar_gold("distribuicao_proficiencia").query(filtro)

    df = mvr[["id_municipio", "sigla_uf", "alunos_com_nota", "taxa_alfabetizacao", "ic95", "meta_ano", "gap", "situacao_meta"]]
    df = df.merge(evo[["id_municipio", "taxa_participacao", "proficiencia_media", "criancas_nao_alfabetizadas"]], on="id_municipio", how="left")
    df = df.merge(dist[["id_municipio"] + COLS_DISTRIBUICAO], on="id_municipio", how="left")
    df = df.merge(montar_contexto_externo(ano).drop(columns=COLS_EXTERNAS_REPETIDAS), on="id_municipio", how="left")
    df = df.merge(meta_pactuada(ano + 1, carregar_metas()).rename(columns={"meta_alvo": "meta_prox"}), on="id_municipio", how="left")
    df.insert(0, "ano", ano)
    return df


def montar_base_municipio(anos=(2023, 2024)) -> pd.DataFrame:
    base = pd.concat([_municipio_no_ano(a) for a in anos], ignore_index=True)
    prox = base[["ano", "id_municipio", "taxa_alfabetizacao", "situacao_meta"]].copy()
    prox["ano"] = prox["ano"] - 1
    prox = prox.rename(columns={"taxa_alfabetizacao": "taxa_prox", "situacao_meta": "situacao_meta_prox"})
    base = base.merge(prox, on=["ano", "id_municipio"], how="left")
    base["nao_atingiu_prox"] = np.where(base["situacao_meta_prox"].isna(), np.nan,
                                        (base["situacao_meta_prox"] == "nao_atingiu").astype(float))
    return base


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano", type=int, default=config.ANO_ALVO)
    args = ap.parse_args(argv)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    aluno = montar_base_aluno(args.ano)
    aluno.to_parquet(config.PROCESSED / "base_modelagem_aluno.parquet", index=False)
    print(f"base_modelagem_aluno     {aluno.shape[0]:>10,} x {aluno.shape[1]}  sem_historico={aluno['sem_historico'].mean():.1%}")

    mun = montar_base_municipio()
    mun.to_parquet(config.PROCESSED / "base_modelagem_municipio.parquet", index=False)
    print(f"base_modelagem_municipio {mun.shape[0]:>10,} x {mun.shape[1]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_feature_store.py tests/test_contexto.py -v`
Expected: 8 passed

- [ ] **Step 6: Gerar de verdade**

```bash
.venv/Scripts/python.exe -m src.preprocessing.feature_store
```
Expected: `base_modelagem_aluno` ≈ 1.851.852 linhas × ~85 colunas, `sem_historico` cobrindo 675 municípios (anotar a fração de alunos); `base_modelagem_municipio` ≈ 10.276 linhas (5.452 em 2024 + 4.824 em 2023). Tempo: alguns minutos.

- [ ] **Step 7: Escrever `docs/dicionario_base_modelagem.md`**

Uma tabela por base. Colunas da tabela: `coluna | origem | ano de referência (para alvo 2024) | regime (produção / diagnóstico / fora) | observação`. Listar **todas** as colunas da base (gerar a lista com `base.dtypes`), agrupadas: identificadores e alvo (regime "fora", com o motivo), Silver do aluno, contexto municipal t−1, meta pactuada, externas por fonte (usar `anos_referencia(2024)` para preencher o ano), LOO da escola (regime diagnóstico). Registrar no cabeçalho: universo (presentes com nota, 2024), corte 743, fração `sem_historico`, taxa de faltante por fonte externa (medida no Step 5 da Task 5) e a data de publicação de cada fonte (IDEB 2023: ago/2024; Censo Escolar 2023: mar/2024; PIB 2022: dez/2024; estimativa de população 2023: ago/2023; indicadores educacionais 2023: 2024).

- [ ] **Step 8: Commit e PR**

```bash
git add src/preprocessing/feature_store.py tests/test_feature_store.py tests/conftest.py docs/dicionario_base_modelagem.md
git commit -m "feature store: base de aluno e de municipio, dicionario"
git push -u origin feature/feature-store
```
PR `feature/feature-store` → `develop`. Descrição: os dois grãos, o inventário de vazamentos da spec (seção 3.3) mapeado para o código (t−1 em `contexto.py`, LOO, `sem_historico`) e por que `base_modelagem_*` não vai para o Git.

---

### Task 8: Listas de features, guarda de leakage e pipeline sklearn (`features.py`, `pipeline.py`)

**Files:**
- Create: `src/preprocessing/features.py`, `src/preprocessing/pipeline.py`, `tests/test_features.py`, `tests/test_pipeline.py`
- Modify: `tests/conftest.py` (helper `montar_base_sintetica`)

**Interfaces:**
- `features.py`:
  - `CATEGORICAS = ["rede_nome", "sigla_uf", "regiao"]`, `DIAGNOSTICO = ["taxa_escola_loo", "prof_media_escola_loo", "n_alunos_escola", "taxa_participacao_escola_loo"]`, `NAO_FEATURES = COLUNAS_PROIBIDAS | {"ano", "nome_municipio"}`, `REGIMES = ("producao", "diagnostico")`.
  - `verificar_leakage(colunas) -> None` — `ValueError` se qualquer coluna proibida aparecer.
  - `colunas_por_regime(df, regime) -> tuple[list[str], list[str]]` — `(numericas, categoricas)` derivadas do DataFrame: numéricas = tudo que não é proibido, não é categórica e (no regime produção) não é de diagnóstico; bool conta como numérica.
  - `separar_xy(df, regime) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]` — `(X, y, grupos=id_escola, pesos=peso_aluno)`; numéricas viram `float64` (bool e `Int64` inclusos); chama `verificar_leakage` nas colunas de `X`.
- `pipeline.py`:
  - `build_preprocessor(num_cols, cat_cols) -> ColumnTransformer` — `SimpleImputer(median, add_indicator=True)` + `StandardScaler` nas numéricas; `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` nas categóricas (denso: o SHAP da Task 13 precisa).
  - `build_pipeline(modelo, num_cols, cat_cols, seed=SEED, params=None) -> Pipeline` com passos `("prep", ...)`, `("clf", ...)`. `modelo ∈ {"dummy", "logistica", "hgb"}`; `params` são chaves com prefixo `clf__` (saída da Task 12).

- [ ] **Step 1: Branch**

```bash
git checkout -b feature/pipeline-modelo develop
```

- [ ] **Step 2: Helper de base sintética em `tests/conftest.py`**

Acrescentar ao final:
```python
def montar_base_sintetica(n_escolas: int = 30, alunos_por_escola: int = 20, seed: int = 0) -> pd.DataFrame:
    """Imita base_modelagem_aluno: contexto com sinal de verdade, colunas proibidas presentes, NaN nas externas."""
    rng = np.random.default_rng(seed)
    muns = [1100015, 1100023, 1100031, 2900108, 3550308]
    ufs = {1100015: "RO", 1100023: "RO", 1100031: "RO", 2900108: "BA", 3550308: "SP"}
    regioes = {"RO": "Norte", "BA": "Nordeste", "SP": "Sudeste"}
    taxa_mun = {m: t for m, t in zip(muns, [0.35, 0.45, np.nan, 0.55, 0.75])}
    linhas = []
    for e in range(n_escolas):
        m = muns[e % len(muns)]
        efeito_escola = rng.normal(0, 0.8)
        for _ in range(alunos_por_escola):
            base_logit = 3 * ((taxa_mun[m] if not np.isnan(taxa_mun[m]) else 0.5) - 0.5) + efeito_escola
            p = 1 / (1 + np.exp(-(base_logit + rng.normal(0, 1))))
            alf = int(rng.random() < p)
            linhas.append({
                "id_municipio": m, "id_escola": 60000 + e, "sigla_uf": ufs[m], "regiao": regioes[ufs[m]],
                "rede_nome": "municipal" if e % 4 else "estadual", "rede": 3 if e % 4 else 2,
                "taxa_alfabetizacao_mun_t1": taxa_mun[m], "ic95_mun_t1": 0.05, "sem_historico": np.isnan(taxa_mun[m]),
                "meta_alvo": 0.6, "log_pib_per_capita": rng.normal(3, 0.5), "tdi_ai": rng.normal(8, 3) if rng.random() > 0.1 else np.nan,
                "pct_escolas_rurais": rng.random(), "matriculas_ai": int(rng.integers(100, 5000)),
                "taxa_escola_loo": np.nan, "prof_media_escola_loo": np.nan, "n_alunos_escola": alunos_por_escola - 1,
                "taxa_participacao_escola_loo": 0.9,
                "alfabetizado": alf, "proficiencia": 743 + (1 if alf else -1) * abs(rng.normal(30, 20)),
                "peso_aluno": float(rng.uniform(0.5, 2.0)), "presente": True, "sem_nota": False, "presenca": 1,
                "preenchimento_caderno": 1, "caderno": "1", "serie": 2, "ano": 2024, "presenca_nome": "presente",
                "_row_hash": 0, "_ingestion_ts": "", "_source": "", "nome_municipio": "x",
            })
    df = pd.DataFrame(linhas)
    df["id_aluno"] = range(1, len(df) + 1)
    df["matriculas_ai"] = df["matriculas_ai"].astype("Int64")   # dtype nullable como vem do parquet
    g = df.groupby("id_escola")
    n = g["alfabetizado"].transform("count")
    df["taxa_escola_loo"] = (g["alfabetizado"].transform("sum") - df["alfabetizado"]) / (n - 1)
    df["prof_media_escola_loo"] = (g["proficiencia"].transform("sum") - df["proficiencia"]) / (n - 1)
    return df
```

- [ ] **Step 3: Testes**

`tests/test_features.py`:
```python
import pytest

from src import config
from src.preprocessing import features
from tests.conftest import montar_base_sintetica


def test_verificar_leakage_barra_colunas_proibidas():
    with pytest.raises(ValueError, match="proficiencia"):
        features.verificar_leakage(["log_pib_per_capita", "proficiencia"])
    features.verificar_leakage(["log_pib_per_capita"])          # não levanta


def test_colunas_por_regime():
    base = montar_base_sintetica(n_escolas=5, alunos_por_escola=4)
    num, cat = features.colunas_por_regime(base, "producao")
    assert cat == ["rede_nome", "sigla_uf", "regiao"]
    assert "sem_historico" in num and "taxa_alfabetizacao_mun_t1" in num and "matriculas_ai" in num
    assert not set(num) & config.COLUNAS_PROIBIDAS and not set(num) & set(features.DIAGNOSTICO)
    assert "nome_municipio" not in num and "ano" not in num
    num_d, _ = features.colunas_por_regime(base, "diagnostico")
    assert set(features.DIAGNOSTICO) <= set(num_d)
    with pytest.raises(ValueError):
        features.colunas_por_regime(base, "outro")


def test_separar_xy_devolve_x_limpo_grupos_e_pesos():
    base = montar_base_sintetica(n_escolas=5, alunos_por_escola=4)
    X, y, grupos, pesos = features.separar_xy(base, "producao")
    assert not set(X.columns) & config.COLUNAS_PROIBIDAS
    assert y.name == "alfabetizado" and set(y.unique()) <= {0, 1}
    assert grupos.name == "id_escola" and pesos.name == "peso_aluno"
    assert X["sem_historico"].dtype == "float64" and X["matriculas_ai"].dtype == "float64"
    assert len(X) == len(y) == len(grupos) == len(pesos) == 20
```

`tests/test_pipeline.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import features, pipeline
from tests.conftest import montar_base_sintetica


@pytest.mark.parametrize("modelo", ["dummy", "logistica", "hgb"])
def test_pipeline_treina_com_nan_e_categoria_nova(modelo):
    base = montar_base_sintetica(n_escolas=20, alunos_por_escola=15)
    X, y, _, _ = features.separar_xy(base, "producao")
    num, cat = features.colunas_por_regime(base, "producao")
    pipe = pipeline.build_pipeline(modelo, num, cat)
    pipe.fit(X, y)
    novo = X.head(3).copy()
    novo.loc[novo.index[0], "sigla_uf"] = "ZZ"                    # categoria nunca vista
    novo.loc[novo.index[1], "log_pib_per_capita"] = np.nan
    proba = pipe.predict_proba(novo)[:, 1]
    assert proba.shape == (3,) and np.all((proba >= 0) & (proba <= 1))
    nomes = pipe.named_steps["prep"].get_feature_names_out()
    assert any("missingindicator" in n for n in nomes)            # add_indicator ligado
    assert not any(n.startswith("remainder") for n in nomes)      # nada passa sem transformar


def test_params_com_prefixo_clf_chegam_no_modelo():
    pipe = pipeline.build_pipeline("hgb", ["a"], [], params={"clf__max_depth": 3, "clf__learning_rate": 0.2})
    assert pipe.named_steps["clf"].max_depth == 3
    assert pipe.named_steps["clf"].random_state == 42


def test_modelo_desconhecido():
    with pytest.raises(ValueError, match="modelo"):
        pipeline.build_pipeline("xgboost", ["a"], [])
```

- [ ] **Step 4: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features.py tests/test_pipeline.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 5: Implementar**

`src/preprocessing/features.py`:
```python
"""Quais colunas entram no modelo, por regime, e a guarda contra vazamento."""
import pandas as pd

from src import config

CATEGORICAS = ["rede_nome", "sigla_uf", "regiao"]
DIAGNOSTICO = ["taxa_escola_loo", "prof_media_escola_loo", "n_alunos_escola", "taxa_participacao_escola_loo"]
NAO_FEATURES = frozenset(config.COLUNAS_PROIBIDAS | {"ano", "nome_municipio"})
REGIMES = ("producao", "diagnostico")
ALVO = "alfabetizado"


def verificar_leakage(colunas) -> None:
    vazadas = sorted(set(colunas) & config.COLUNAS_PROIBIDAS)
    if vazadas:
        raise ValueError(f"colunas proibidas na matriz de features: {vazadas}")


def colunas_por_regime(df: pd.DataFrame, regime: str) -> tuple[list[str], list[str]]:
    if regime not in REGIMES:
        raise ValueError(f"regime deve ser um de {REGIMES}, veio {regime!r}")
    cat = [c for c in CATEGORICAS if c in df.columns]
    num = []
    for c in df.columns:
        if c in NAO_FEATURES or c in cat:
            continue
        if regime == "producao" and c in DIAGNOSTICO:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            num.append(c)
    return num, cat


def separar_xy(df: pd.DataFrame, regime: str):
    num, cat = colunas_por_regime(df, regime)
    verificar_leakage(num + cat)
    X = df[num + cat].copy()
    X[num] = X[num].astype("float64")
    return X, df[ALVO].astype(int), df["id_escola"], df["peso_aluno"]
```

`src/preprocessing/pipeline.py`:
```python
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
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    categorico = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("num", numerico, num_cols), ("cat", categorico, cat_cols)],
        remainder="drop", verbose_feature_names_out=True,
    )


def _estimador(modelo: str, seed: int):
    if modelo == "dummy":
        return DummyClassifier(strategy="prior")
    if modelo == "logistica":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if modelo == "hgb":
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
```

- [ ] **Step 6: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features.py tests/test_pipeline.py -v`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add src/preprocessing/features.py src/preprocessing/pipeline.py tests/test_features.py tests/test_pipeline.py tests/conftest.py
git commit -m "listas de features por regime, guarda de leakage e pipeline sklearn"
```

---

### Task 9: Split e validação cruzada por escola (`src/modeling/split.py`)

**Files:**
- Create: `src/modeling/split.py`, `tests/test_split.py`

**Interfaces:**
- `dividir_por_escola(grupos, frac_val=0.15, frac_teste=0.15, seed=SEED) -> dict[str, np.ndarray]` — chaves `treino`, `validacao`, `teste` com **posições** (0..n−1). Nenhum grupo em duas partes.
- `cv_por_grupo(n_splits=5, seed=SEED) -> StratifiedGroupKFold` (`shuffle=True`).
- `conferir_sem_vazamento(grupos, partes) -> None` — `AssertionError` se um grupo cair em duas partes.

- [ ] **Step 1: Teste**

`tests/test_split.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.modeling import split


def _grupos(n_escolas=100, por_escola=10):
    return pd.Series(np.repeat(np.arange(n_escolas), por_escola))


def test_nenhuma_escola_em_duas_partes_e_proporcoes():
    g = _grupos()
    partes = split.dividir_por_escola(g)
    assert set(partes) == {"treino", "validacao", "teste"}
    total = sum(len(v) for v in partes.values())
    assert total == len(g) and len(np.intersect1d(partes["treino"], partes["teste"])) == 0
    split.conferir_sem_vazamento(g, partes)                      # não levanta
    assert 0.6 <= len(partes["treino"]) / total <= 0.8
    assert 0.1 <= len(partes["teste"]) / total <= 0.2


def test_split_e_deterministico():
    g = _grupos()
    a, b = split.dividir_por_escola(g, seed=42), split.dividir_por_escola(g, seed=42)
    assert np.array_equal(a["teste"], b["teste"])
    assert not np.array_equal(a["teste"], split.dividir_por_escola(g, seed=7)["teste"])


def test_conferir_detecta_vazamento():
    g = _grupos(n_escolas=2, por_escola=2)
    with pytest.raises(AssertionError, match="escola"):
        split.conferir_sem_vazamento(g, {"treino": np.array([0, 1, 2]), "teste": np.array([3])})


def test_cv_por_grupo_nao_mistura_escolas():
    g = _grupos(n_escolas=20, por_escola=5)
    y = (np.arange(len(g)) % 3 == 0).astype(int)
    X = np.zeros((len(g), 1))
    for tr, te in split.cv_por_grupo(n_splits=4).split(X, y, g):
        assert not set(g.iloc[tr]) & set(g.iloc[te])
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_split.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar**

`src/modeling/split.py`:
```python
"""Divisão treino/validação/teste e CV sempre por escola: nenhum id_escola em dois lados."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src import config


def dividir_por_escola(grupos, frac_val: float = 0.15, frac_teste: float = 0.15,
                       seed: int = config.SEED) -> dict[str, np.ndarray]:
    grupos = np.asarray(grupos)
    idx = np.arange(len(grupos))
    resto, teste = next(GroupShuffleSplit(n_splits=1, test_size=frac_teste, random_state=seed).split(idx, groups=grupos))
    frac_val_rel = frac_val / (1 - frac_teste)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=frac_val_rel, random_state=seed).split(resto, groups=grupos[resto]))
    partes = {"treino": resto[tr], "validacao": resto[va], "teste": teste}
    conferir_sem_vazamento(grupos, partes)
    return partes


def cv_por_grupo(n_splits: int = 5, seed: int = config.SEED) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def conferir_sem_vazamento(grupos, partes: dict[str, np.ndarray]) -> None:
    grupos = pd.Series(np.asarray(grupos))
    vistos: dict = {}
    for nome, pos in partes.items():
        for g in set(grupos.iloc[pos]):
            assert vistos.setdefault(g, nome) == nome, f"escola {g} está em {vistos[g]} e em {nome}"
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_split.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/modeling/split.py tests/test_split.py
git commit -m "split e validacao cruzada por escola"
```

---

### Task 10: Métricas (`src/evaluation/metrics.py`)

**Files:**
- Create: `src/evaluation/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Convenção: `y` é `alfabetizado` (1/0), `proba` é P(alfabetizado). A classe de interesse para política é **não alfabetizado** (`y == 0`), prevista quando `proba < limiar`.
- `calcular_metricas(y, proba, limiar=0.5, pesos=None) -> dict` — chaves: `n`, `limiar`, `roc_auc`, `pr_auc_nao_alf`, `f1_nao_alf`, `recall_nao_alf`, `precisao_nao_alf`, `balanced_accuracy`, `acuracia`, `brier`, `matriz` (lista 2×2 `[[VN, FP], [FN, VP]]` com positivo = alfabetizado), `prevalencia_nao_alf`. Com `pesos`, todas ponderadas (`sample_weight`).
- `escolher_limiar(y, proba, recall_minimo=0.8) -> float` — menor limiar que garante `recall_nao_alf ≥ recall_minimo` (maximiza precisão dado o recall).
- `metricas_por_recorte(recorte, y, proba, limiar, pesos=None, minimo=500) -> pd.DataFrame` — uma linha por valor de `recorte` com ≥ `minimo` linhas.
- `tabela_calibracao(y, proba, n_bins=10) -> pd.DataFrame[faixa, proba_media, taxa_observada, n]`.

- [ ] **Step 1: Teste**

`tests/test_metrics.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics


def test_predicao_perfeita_e_aleatoria():
    y = np.array([1, 1, 1, 0, 0, 0])
    m = metrics.calcular_metricas(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]))
    assert m["roc_auc"] == 1.0 and m["f1_nao_alf"] == 1.0 and m["recall_nao_alf"] == 1.0
    assert m["matriz"] == [[3, 0], [0, 3]] and m["n"] == 6 and m["prevalencia_nao_alf"] == 0.5
    m2 = metrics.calcular_metricas(y, np.full(6, 0.5))
    assert m2["roc_auc"] == 0.5 and m2["brier"] == pytest.approx(0.25)


def test_pesos_mudam_a_metrica():
    y = np.array([1, 0, 0, 1])
    proba = np.array([0.9, 0.6, 0.2, 0.4])
    sem = metrics.calcular_metricas(y, proba)
    com = metrics.calcular_metricas(y, proba, pesos=np.array([1, 10, 1, 1]))
    assert sem["acuracia"] == 0.5 and com["acuracia"] < 0.5


def test_escolher_limiar_garante_recall_minimo():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.6 * y + rng.normal(0, 0.3, 2000), 0, 1)
    limiar = metrics.escolher_limiar(y, proba, recall_minimo=0.8)
    m = metrics.calcular_metricas(y, proba, limiar)
    assert m["recall_nao_alf"] >= 0.8
    assert metrics.calcular_metricas(y, proba, limiar - 0.02)["recall_nao_alf"] < m["recall_nao_alf"]


def test_recorte_e_calibracao():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1200)
    proba = np.clip(y * 0.5 + rng.random(1200) * 0.5, 0, 1)
    uf = pd.Series(np.where(np.arange(1200) < 900, "SP", "AC"))
    r = metrics.metricas_por_recorte(uf, y, proba, 0.5, minimo=500)
    assert r["recorte"].tolist() == ["SP"] and r.loc[0, "n"] == 900
    cal = metrics.tabela_calibracao(y, proba, n_bins=5)
    assert len(cal) <= 5 and cal["n"].sum() == 1200 and cal["taxa_observada"].between(0, 1).all()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar**

`src/evaluation/metrics.py`:
```python
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
    captura recall_minimo dos não alfabetizados é o que erra menos alfabetizados."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    q = np.quantile(proba[y == 0], recall_minimo, method="higher")
    return float(np.nextafter(q, np.inf))


def metricas_por_recorte(recorte, y, proba, limiar: float, pesos=None, minimo: int = 500) -> pd.DataFrame:
    recorte = pd.Series(np.asarray(recorte))
    y, proba = np.asarray(y), np.asarray(proba)
    pesos = None if pesos is None else np.asarray(pesos)
    linhas = []
    for valor, pos in recorte.groupby(recorte).groups.items():
        pos = np.asarray(pos)
        if len(pos) < minimo or len(np.unique(y[pos])) < 2:
            continue
        m = calcular_metricas(y[pos], proba[pos], limiar, None if pesos is None else pesos[pos])
        m.pop("matriz")
        linhas.append({"recorte": valor, **m})
    return pd.DataFrame(linhas).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def tabela_calibracao(y, proba, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "proba": np.asarray(proba)})
    df["faixa"] = pd.cut(df["proba"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = df.groupby("faixa", observed=True).agg(proba_media=("proba", "mean"), taxa_observada=("y", "mean"), n=("y", "size"))
    return out.reset_index()
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "metricas ponderadas, por recorte, limiar e calibracao"
```

---

### Task 11: Treino ponta a ponta do modelo A (`src/modeling/train.py`)

**Files:**
- Create: `src/modeling/train.py`, `tests/test_train.py`
- Outputs: `models/modelo_aluno_<regime>_<modelo>.joblib`, `models/particao_<regime>.parquet` (id_aluno, parte), `reports/metricas_<regime>_<modelo>.json`

**Interfaces:**
- `treinar(base, regime, modelo="hgb", seed=SEED, params=None, recall_minimo=0.8, cv_municipio=False) -> dict` com chaves `pipeline`, `partes`, `colunas` (`{"numericas", "categoricas"}`), `metricas`. `metricas` é serializável em JSON: `regime`, `modelo`, `seed`, `params`, `n_treino`, `n_validacao`, `n_teste`, `n_escolas_teste`, `limiar`, `validacao` (dict de `calcular_metricas`), `teste`, `teste_ponderado`, `por_uf` (records), `por_rede` (records), `calibracao` (records), `cv_municipio` (`{"roc_auc_media", "roc_auc_dp"}` ou `None`).
- Regras: fit só no treino; limiar escolhido na **validação**; teste tocado **uma vez**; `GroupKFold(5)` por `id_municipio` (só se `cv_municipio=True`, é caro) para generalização a municípios não vistos.
- `salvar(resultado, base) -> dict[str, Path]`.
- CLI: `python -m src.modeling.train --regime producao --modelo hgb [--params reports/melhores_params_producao_hgb.json] [--amostra-escolas 3000] [--cv-municipio]`.

- [ ] **Step 1: Teste**

`tests/test_train.py`:
```python
import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.modeling import train
from tests.conftest import montar_base_sintetica


def test_treinar_logistica_ponta_a_ponta():
    base = montar_base_sintetica(n_escolas=40, alunos_por_escola=15)
    r = train.treinar(base, "producao", modelo="logistica", recall_minimo=0.7, cv_municipio=True)
    m = r["metricas"]
    assert m["regime"] == "producao" and m["modelo"] == "logistica"
    assert m["n_treino"] + m["n_validacao"] + m["n_teste"] == 600
    assert 0.5 < m["teste"]["roc_auc"] <= 1.0                     # há sinal na base sintética
    assert m["validacao"]["recall_nao_alf"] >= 0.7
    assert set(m["teste_ponderado"]) == set(m["teste"])
    assert isinstance(m["por_uf"], list) and isinstance(m["calibracao"], list)
    assert m["cv_municipio"] is not None and "roc_auc_media" in m["cv_municipio"]
    json.dumps(m)                                                 # serializável
    escolas_teste = set(base.iloc[r["partes"]["teste"]]["id_escola"])
    assert not escolas_teste & set(base.iloc[r["partes"]["treino"]]["id_escola"])


def test_diagnostico_usa_loo_e_producao_nao():
    base = montar_base_sintetica(n_escolas=20, alunos_por_escola=10)
    prod = train.treinar(base, "producao", modelo="dummy")
    diag = train.treinar(base, "diagnostico", modelo="dummy")
    assert "taxa_escola_loo" not in prod["colunas"]["numericas"]
    assert "taxa_escola_loo" in diag["colunas"]["numericas"]


def test_salvar_grava_modelo_particao_e_metricas(lake_tmp):
    base = montar_base_sintetica(n_escolas=20, alunos_por_escola=10)
    r = train.treinar(base, "producao", modelo="dummy")
    caminhos = train.salvar(r, base)
    assert caminhos["modelo"] == config.MODELS / "modelo_aluno_producao_dummy.joblib"
    assert joblib.load(caminhos["modelo"]).predict_proba(base.head(2)[r["colunas"]["numericas"] + r["colunas"]["categoricas"]]).shape == (2, 2)
    part = pd.read_parquet(caminhos["particao"])
    assert set(part["parte"]) == {"treino", "validacao", "teste"} and len(part) == 200
    assert json.loads(caminhos["metricas"].read_text())["modelo"] == "dummy"


def test_cli_com_amostra(lake_tmp, monkeypatch):
    base = montar_base_sintetica(n_escolas=30, alunos_por_escola=10)
    config.PROCESSED.mkdir(exist_ok=True)
    base.to_parquet(config.PROCESSED / "base_modelagem_aluno.parquet")
    train.main(["--regime", "producao", "--modelo", "dummy", "--amostra-escolas", "10"])
    m = json.loads((config.REPORTS / "metricas_producao_dummy.json").read_text())
    assert m["n_treino"] + m["n_validacao"] + m["n_teste"] == 100
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar**

`src/modeling/train.py`:
```python
"""Treina, avalia e grava o modelo de aluno de ponta a ponta.

    python -m src.modeling.train --regime producao --modelo hgb
    python -m src.modeling.train --regime diagnostico --modelo hgb --params reports/melhores_params_diagnostico_hgb.json
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score

from src import config
from src.evaluation.metrics import calcular_metricas, escolher_limiar, metricas_por_recorte, tabela_calibracao
from src.modeling.split import dividir_por_escola
from src.preprocessing.features import colunas_por_regime, separar_xy
from src.preprocessing.pipeline import build_pipeline


def treinar(base: pd.DataFrame, regime: str, modelo: str = "hgb", seed: int = config.SEED,
            params: dict | None = None, recall_minimo: float = 0.8, cv_municipio: bool = False) -> dict:
    X, y, grupos, pesos = separar_xy(base, regime)
    num, cat = colunas_por_regime(base, regime)
    partes = dividir_por_escola(grupos, seed=seed)
    tr, va, te = partes["treino"], partes["validacao"], partes["teste"]

    pipe = build_pipeline(modelo, num, cat, seed=seed, params=params)
    pipe.fit(X.iloc[tr], y.iloc[tr])

    proba_va = pipe.predict_proba(X.iloc[va])[:, 1]
    limiar = escolher_limiar(y.iloc[va], proba_va, recall_minimo)

    proba_te = pipe.predict_proba(X.iloc[te])[:, 1]
    y_te, p_te = y.iloc[te].to_numpy(), pesos.iloc[te].to_numpy()

    cv = None
    if cv_municipio:
        scores = cross_val_score(build_pipeline(modelo, num, cat, seed=seed, params=params), X.iloc[tr], y.iloc[tr],
                                 groups=base["id_municipio"].iloc[tr], cv=GroupKFold(n_splits=5), scoring="roc_auc")
        cv = {"roc_auc_media": float(scores.mean()), "roc_auc_dp": float(scores.std())}

    metricas = {
        "regime": regime, "modelo": modelo, "seed": seed, "params": params or {},
        "n_treino": int(len(tr)), "n_validacao": int(len(va)), "n_teste": int(len(te)),
        "n_escolas_teste": int(grupos.iloc[te].nunique()), "limiar": limiar,
        "validacao": calcular_metricas(y.iloc[va], proba_va, limiar),
        "teste": calcular_metricas(y_te, proba_te, limiar),
        "teste_ponderado": calcular_metricas(y_te, proba_te, limiar, pesos=p_te),
        "por_uf": metricas_por_recorte(base["sigla_uf"].iloc[te], y_te, proba_te, limiar).to_dict("records"),
        "por_rede": metricas_por_recorte(base["rede_nome"].iloc[te], y_te, proba_te, limiar).to_dict("records"),
        "calibracao": tabela_calibracao(y_te, proba_te).assign(faixa=lambda d: d["faixa"].astype(str)).to_dict("records"),
        "cv_municipio": cv,
    }
    return {"pipeline": pipe, "partes": partes, "colunas": {"numericas": num, "categoricas": cat}, "metricas": metricas}


def salvar(resultado: dict, base: pd.DataFrame) -> dict[str, Path]:
    m = resultado["metricas"]
    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    nome = f"{m['regime']}_{m['modelo']}"

    caminhos = {
        "modelo": config.MODELS / f"modelo_aluno_{nome}.joblib",
        "particao": config.MODELS / f"particao_{m['regime']}.parquet",
        "metricas": config.REPORTS / f"metricas_{nome}.json",
    }
    joblib.dump(resultado["pipeline"], caminhos["modelo"])
    partes = pd.concat([pd.DataFrame({"id_aluno": base["id_aluno"].iloc[pos].to_numpy(), "parte": nome_parte})
                        for nome_parte, pos in resultado["partes"].items()], ignore_index=True)
    partes.to_parquet(caminhos["particao"], index=False)
    caminhos["metricas"].write_text(json.dumps({**m, "colunas": resultado["colunas"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    return caminhos


def _amostrar_escolas(base: pd.DataFrame, n_escolas: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    escolhidas = rng.choice(base["id_escola"].unique(), size=n_escolas, replace=False)
    return base[base["id_escola"].isin(escolhidas)].reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["producao", "diagnostico"], default="producao")
    ap.add_argument("--modelo", choices=["dummy", "logistica", "hgb"], default="hgb")
    ap.add_argument("--params", type=Path, help="json com hiperparâmetros (saída de tune.py)")
    ap.add_argument("--amostra-escolas", type=int, help="treina numa amostra de escolas (execuções rápidas)")
    ap.add_argument("--recall-minimo", type=float, default=0.8)
    ap.add_argument("--cv-municipio", action="store_true")
    args = ap.parse_args(argv)

    base = pd.read_parquet(config.PROCESSED / "base_modelagem_aluno.parquet")
    if args.amostra_escolas:
        base = _amostrar_escolas(base, args.amostra_escolas, config.SEED)
    params = json.loads(args.params.read_text()) if args.params else None

    r = treinar(base, args.regime, args.modelo, params=params, recall_minimo=args.recall_minimo, cv_municipio=args.cv_municipio)
    caminhos = salvar(r, base)
    t = r["metricas"]["teste"]
    print(f"{args.regime}/{args.modelo}: roc_auc={t['roc_auc']:.4f} pr_auc_nao_alf={t['pr_auc_nao_alf']:.4f} "
          f"recall_nao_alf={t['recall_nao_alf']:.3f} brier={t['brier']:.4f} limiar={r['metricas']['limiar']:.3f}")
    for k, p in caminhos.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train.py -v`
Expected: 4 passed

- [ ] **Step 5: Rodar com dado real (piso, baseline e principal)**

```bash
.venv/Scripts/python.exe -m src.modeling.train --regime producao --modelo dummy
.venv/Scripts/python.exe -m src.modeling.train --regime producao --modelo logistica
.venv/Scripts/python.exe -m src.modeling.train --regime producao --modelo hgb
.venv/Scripts/python.exe -m src.modeling.train --regime diagnostico --modelo hgb
```
Expected: dummy `roc_auc = 0.5`; logística e HGB produção com `roc_auc` entre 0,65 e 0,72 (teto discutido na spec, seção 3.1); diagnóstico acima de produção (a escola explica além do município). Se HGB produção vier abaixo de 0,60 ou acima de 0,80, investigar antes de seguir: abaixo sugere merge quebrado (externas NaN em massa), acima sugere vazamento (conferir `colunas` no json). Tempo: HGB alguns minutos; logística pode levar mais (1,3 mi linhas, one-hot de 27 UFs).

- [ ] **Step 6: Commit**

```bash
git add src/modeling/train.py tests/test_train.py reports/metricas_producao_dummy.json reports/metricas_producao_logistica.json reports/metricas_producao_hgb.json reports/metricas_diagnostico_hgb.json
git commit -m "treino ponta a ponta do modelo de aluno com metricas em reports"
```

---

### Task 12: Busca de hiperparâmetros (`src/modeling/tune.py`)

**Files:**
- Create: `src/modeling/tune.py`, `tests/test_tune.py`
- Outputs: `reports/melhores_params_<regime>_<modelo>.json`, `reports/busca_<regime>_<modelo>.csv`

**Interfaces:**
- `ESPACOS: dict[str, dict]` — distribuições por modelo, chaves com prefixo `clf__`.
- `amostrar_por_escola(base, n_alunos, seed=SEED) -> pd.DataFrame` — sorteia escolas inteiras até somar ~`n_alunos` (nunca alunos soltos: manteria escolas em dois folds).
- `buscar(base, regime, modelo="hgb", n_amostra=300_000, n_candidatos=40, seed=SEED) -> tuple[dict, pd.DataFrame]` — `HalvingRandomSearchCV` com `cv_por_grupo(5)`, `scoring="roc_auc"`, sobre a amostra; devolve `(melhores_params, cv_results)` com valores em tipos nativos (JSON).
- CLI: `python -m src.modeling.tune --regime producao --modelo hgb [--n-amostra 300000] [--n-candidatos 40]`.
- Observação para o relatório: o `early_stopping` interno do HGB usa um `validation_fraction` aleatório (não por escola) dentro de cada fold. É aceitável para escolher hiperparâmetros; a métrica reportada vem do teste por escola da Task 11.

- [ ] **Step 1: Teste**

`tests/test_tune.py`:
```python
import json

import pandas as pd

from src.modeling import tune
from tests.conftest import montar_base_sintetica


def test_amostra_por_escola_nao_parte_escolas():
    base = montar_base_sintetica(n_escolas=40, alunos_por_escola=10)
    am = tune.amostrar_por_escola(base, n_alunos=100)
    assert 100 <= len(am) < 130
    contagem = base.groupby("id_escola").size()
    assert (am.groupby("id_escola").size() == contagem.loc[am["id_escola"].unique()]).all()


def test_buscar_devolve_params_serializaveis_e_resultados():
    base = montar_base_sintetica(n_escolas=40, alunos_por_escola=15)
    params, res = tune.buscar(base, "producao", modelo="logistica", n_amostra=600, n_candidatos=3)
    assert params and all(k.startswith("clf__") for k in params)
    json.dumps(params)
    assert isinstance(res, pd.DataFrame) and "mean_test_score" in res.columns and len(res) >= 3
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tune.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar**

`src/modeling/tune.py`:
```python
"""Busca de hiperparâmetros em amostra por escola; o refit na base completa é o train.py com --params.

    python -m src.modeling.tune --regime producao --modelo hgb
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from src import config
from src.modeling.split import cv_por_grupo
from src.preprocessing.features import colunas_por_regime, separar_xy
from src.preprocessing.pipeline import build_pipeline

ESPACOS = {
    "hgb": {
        "clf__learning_rate": loguniform(0.02, 0.3),
        "clf__max_leaf_nodes": randint(15, 128),
        "clf__min_samples_leaf": randint(20, 500),
        "clf__l2_regularization": loguniform(1e-3, 10),
        "clf__max_depth": [None, 4, 6, 8, 12],
    },
    "logistica": {"clf__C": loguniform(1e-3, 10)},
}


def amostrar_por_escola(base: pd.DataFrame, n_alunos: int, seed: int = config.SEED) -> pd.DataFrame:
    if n_alunos >= len(base):
        return base
    rng = np.random.default_rng(seed)
    tamanhos = base.groupby("id_escola").size()
    ordem = rng.permutation(tamanhos.index.to_numpy())
    acumulado = tamanhos.loc[ordem].cumsum()
    escolhidas = acumulado.index[acumulado <= n_alunos].tolist()
    if len(escolhidas) < len(ordem):
        escolhidas.append(ordem[len(escolhidas)])          # cruza o alvo com a próxima escola inteira
    return base[base["id_escola"].isin(escolhidas)].reset_index(drop=True)


def _nativo(v):
    return v.item() if isinstance(v, np.generic) else v


def buscar(base: pd.DataFrame, regime: str, modelo: str = "hgb", n_amostra: int = 300_000,
           n_candidatos: int = 40, seed: int = config.SEED) -> tuple[dict, pd.DataFrame]:
    amostra = amostrar_por_escola(base, n_amostra, seed)
    X, y, grupos, _ = separar_xy(amostra, regime)
    num, cat = colunas_por_regime(amostra, regime)
    busca = HalvingRandomSearchCV(
        build_pipeline(modelo, num, cat, seed=seed), ESPACOS[modelo],
        n_candidates=n_candidatos, factor=3, cv=cv_por_grupo(5, seed), scoring="roc_auc",
        random_state=seed, n_jobs=-1, refit=False, verbose=1,
    )
    busca.fit(X, y, groups=grupos)
    melhores = {k: _nativo(v) for k, v in busca.best_params_.items()}
    return melhores, pd.DataFrame(busca.cv_results_)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["producao", "diagnostico"], default="producao")
    ap.add_argument("--modelo", choices=list(ESPACOS), default="hgb")
    ap.add_argument("--n-amostra", type=int, default=300_000)
    ap.add_argument("--n-candidatos", type=int, default=40)
    args = ap.parse_args(argv)

    base = pd.read_parquet(config.PROCESSED / "base_modelagem_aluno.parquet")
    melhores, res = buscar(base, args.regime, args.modelo, args.n_amostra, args.n_candidatos)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    nome = f"{args.regime}_{args.modelo}"
    (config.REPORTS / f"melhores_params_{nome}.json").write_text(json.dumps(melhores, indent=2), encoding="utf-8")
    cols = ["iter", "n_resources", "mean_test_score", "std_test_score", "rank_test_score"] + [c for c in res.columns if c.startswith("param_")]
    res[cols].sort_values(["iter", "rank_test_score"]).to_csv(config.REPORTS / f"busca_{nome}.csv", index=False)
    print(json.dumps(melhores, indent=2))
    print(f"melhor roc_auc (cv por escola, amostra): {res['mean_test_score'].max():.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tune.py -v`
Expected: 2 passed

- [ ] **Step 5: Buscar e refazer o treino com os melhores parâmetros**

```bash
.venv/Scripts/python.exe -m src.modeling.tune --regime producao --modelo hgb
.venv/Scripts/python.exe -m src.modeling.train --regime producao --modelo hgb --params reports/melhores_params_producao_hgb.json --cv-municipio
.venv/Scripts/python.exe -m src.modeling.train --regime diagnostico --modelo hgb --params reports/melhores_params_producao_hgb.json
```
Expected: busca em 300 mil linhas leva dezenas de minutos (40 candidatos, fator 3 → 4 rodadas). O ganho sobre o padrão costuma ser pequeno (+0,005 a +0,015 de AUC); registrar os dois números no relatório. `cv_municipio` (municípios nunca vistos) deve ficar próximo ou um pouco abaixo do teste por escola.

- [ ] **Step 6: Commit e PR**

```bash
git add src/modeling/tune.py tests/test_tune.py reports/melhores_params_producao_hgb.json reports/busca_producao_hgb.csv reports/metricas_producao_hgb.json reports/metricas_diagnostico_hgb.json
git commit -m "busca de hiperparametros por halving em amostra por escola"
git push -u origin feature/pipeline-modelo
```
PR `feature/pipeline-modelo` → `develop`. Descrição: tabela dummy × logística × HGB (produção) × HGB (diagnóstico) com ROC-AUC, PR-AUC e recall no limiar; split por escola e por município; limiar por recall mínimo; o que o tuning mudou.

---

### Task 13: Interpretabilidade (`src/evaluation/interpret.py`)

**Files:**
- Create: `src/evaluation/interpret.py`, `tests/test_interpret.py`

**Interfaces:**
- `nomes_features(pipeline) -> list[str]` — nomes pós-transformação sem prefixos `num__`/`cat__`; indicadores de faltante viram `faltante_<col>`.
- `importancia_permutacao(pipeline, X, y, n_repeats=5, seed=SEED, pesos=None, n_amostra=100_000) -> pd.DataFrame[feature, importancia_media, importancia_dp]` — sobre as **colunas cruas** (uma linha por coluna de `X`, categórica inteira), `scoring="roc_auc"`, em amostra.
- `explicar_shap(pipeline, X, n_amostra=10_000, seed=SEED) -> shap.Explanation` — `TreeExplainer` para HGB, `LinearExplainer` para logística, sobre a matriz transformada; `feature_names` de `nomes_features`.
- `resumo_shap(exp) -> pd.DataFrame[feature, shap_medio_abs]`, `shap_por_grupo(exp, grupo) -> pd.DataFrame` (índice = valor do grupo, colunas = features, valor = |SHAP| médio), `coeficientes_logistica(pipeline) -> pd.DataFrame[feature, coeficiente]`.

- [ ] **Step 1: Branch**

```bash
git checkout -b feature/interpretacao develop
```

- [ ] **Step 2: Teste**

`tests/test_interpret.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.evaluation import interpret
from src.preprocessing import features, pipeline
from tests.conftest import montar_base_sintetica


@pytest.fixture(scope="module")
def ajustado():
    base = montar_base_sintetica(n_escolas=30, alunos_por_escola=20)
    X, y, _, pesos = features.separar_xy(base, "producao")
    num, cat = features.colunas_por_regime(base, "producao")
    hgb = pipeline.build_pipeline("hgb", num, cat).fit(X, y)
    log = pipeline.build_pipeline("logistica", num, cat).fit(X, y)
    return base, X, y, pesos, hgb, log


def test_nomes_features_sem_prefixo(ajustado):
    _, X, _, _, hgb, _ = ajustado
    nomes = interpret.nomes_features(hgb)
    assert "log_pib_per_capita" in nomes and "sigla_uf_SP" in nomes and "faltante_tdi_ai" in nomes
    assert not any(n.startswith(("num__", "cat__")) for n in nomes)
    assert len(nomes) == hgb.named_steps["prep"].transform(X.head(1)).shape[1]


def test_permutacao_uma_linha_por_coluna_crua(ajustado):
    _, X, y, pesos, hgb, _ = ajustado
    imp = interpret.importancia_permutacao(hgb, X, y, n_repeats=2, pesos=pesos, n_amostra=300)
    assert set(imp["feature"]) == set(X.columns)
    assert imp["importancia_media"].iloc[0] >= imp["importancia_media"].iloc[-1]


def test_shap_para_arvore_e_linear(ajustado):
    _, X, _, _, hgb, log = ajustado
    for pipe in (hgb, log):
        exp = interpret.explicar_shap(pipe, X, n_amostra=100)
        assert exp.values.shape == (100, len(interpret.nomes_features(pipe)))
        resumo = interpret.resumo_shap(exp)
        assert resumo["shap_medio_abs"].ge(0).all() and len(resumo) == exp.values.shape[1]
    grupo = interpret.shap_por_grupo(exp, X["regiao"].head(100))
    assert set(grupo.index) <= {"Norte", "Nordeste", "Sudeste"}


def test_coeficientes_logistica(ajustado):
    *_, log = ajustado
    coef = interpret.coeficientes_logistica(log)
    assert "coeficiente" in coef.columns and "taxa_alfabetizacao_mun_t1" in coef["feature"].values
    with pytest.raises(TypeError):
        interpret.coeficientes_logistica(ajustado[4])
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_interpret.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 4: Implementar**

`src/evaluation/interpret.py`:
```python
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
        n = n.split("__", 1)[1]
        nomes.append(n.replace("missingindicator_", "faltante_") if n.startswith("missingindicator_") else n)
    return nomes


def _amostra(X, n, seed, *outros):
    if n is None or n >= len(X):
        return (X, *outros)
    pos = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
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
```

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_interpret.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/interpret.py tests/test_interpret.py
git commit -m "permutation importance e shap para o modelo de aluno"
```

---

### Task 14: Figuras (`src/visualization/plots.py`)

**Files:**
- Create: `src/visualization/plots.py`, `tests/test_plots.py`

**Interfaces:**
- `aplicar_estilo()` — paleta e fontes únicas (chamado no import); backend `Agg`.
- `salvar(fig, nome) -> Path` — grava `config.IMAGES / f"{nome}.png"` (dpi 150, `bbox_inches="tight"`), fecha a figura.
- Todas devolvem `matplotlib.figure.Figure`: `plot_roc_pr(curvas: dict[str, tuple[y, proba]])`, `plot_calibracao(tabela)`, `plot_matriz_confusao(matriz, titulo="")`, `plot_barras_horizontais(df, coluna_valor, coluna_nome="feature", top=20, titulo="")`, `plot_shap_beeswarm(exp, top=15)`, `plot_shap_dependence(exp, feature)`, `plot_por_recorte(df, metrica="roc_auc")`, `plot_comparacao_modelos(metricas: list[dict], chaves=("roc_auc", "pr_auc_nao_alf", "recall_nao_alf"))`, `plot_distribuicao_proficiencia(prof, corte=743)`, `plot_dispersao(df, x, y, hue=None)`, `plot_clusters_pca(coords, labels)`, `plot_ranking_risco(df, top=20)`.

- [ ] **Step 1: Teste**

`tests/test_plots.py`:
```python
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
    assert isinstance(plots.plot_clusters_pca(np.random.default_rng(2).normal(size=(50, 2)), np.arange(50) % 3), Figure)
    rank = pd.DataFrame({"nome_municipio": [f"m{i}" for i in range(30)], "sigla_uf": ["BA"] * 30,
                         "prob_nao_atingir_2025": np.linspace(0.9, 0.3, 30), "criancas_nao_alfabetizadas_2024": np.linspace(5000, 100, 30)})
    assert isinstance(plots.plot_ranking_risco(rank, top=10), Figure)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_plots.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar**

`src/visualization/plots.py` — esqueleto obrigatório (as demais funções seguem o mesmo padrão: recebem dados prontos, devolvem `Figure`, sem ler arquivo):
```python
"""Figuras do README, dos notebooks e do vídeo: um estilo só, sem I/O além de salvar()."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


def aplicar_estilo() -> None:
    sns.set_theme(style="whitegrid", palette=PALETA, font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})


aplicar_estilo()


def salvar(fig, nome: str) -> Path:
    config.IMAGES.mkdir(parents=True, exist_ok=True)
    caminho = config.IMAGES / f"{nome}.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def plot_roc_pr(curvas: dict) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for nome, (y, proba) in curvas.items():
        RocCurveDisplay.from_predictions(y, proba, name=nome, ax=a1)
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
    ax.barh(d[coluna_nome], d[coluna_valor], color=COR_ALF)
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
    ax.set(title=f"{metrica} por recorte (n ≥ mínimo)", xlim=(0.4, 1.0))
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


def plot_clusters_pca(coords, labels) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=np.asarray(labels).astype(str), s=14, alpha=0.7, ax=ax)
    ax.set(xlabel="PC1", ylabel="PC2", title="Clusters de vulnerabilidade (PCA)")
    return fig


def plot_ranking_risco(df: pd.DataFrame, top: int = 20) -> plt.Figure:
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(d) + 1))
    ax.barh(d["nome_municipio"] + " (" + d["sigla_uf"] + ")", d["criancas_nao_alfabetizadas_2024"], color=COR_NAO_ALF)
    for i, (n, p) in enumerate(zip(d["criancas_nao_alfabetizadas_2024"], d["prob_nao_atingir_2025"])):
        ax.text(n, i, f" {p:.0%}", va="center", fontsize=8)
    ax.set(title="Maior risco de não atingir a meta 2025 × crianças não alfabetizadas", xlabel="crianças não alfabetizadas (2024)")
    return fig
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_plots.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/visualization/plots.py tests/test_plots.py
git commit -m "figuras com estilo unico para readme e notebooks"
```

---

### Task 15: Pontuação de base nova (`src/modeling/predict.py`)

**Files:**
- Create: `src/modeling/predict.py`, `tests/test_predict.py`
- Output: `reports/predicoes_<regime>_<modelo>.csv` (por padrão só a partição de teste, para caber no Git)

**Interfaces:**
- `carregar_modelo(caminho=None, regime="producao", modelo="hgb") -> Pipeline`.
- `pontuar(pipeline, base, limiar=0.5) -> pd.DataFrame[id_aluno, id_escola, id_municipio, sigla_uf, prob_alfabetizado, risco_nao_alf, classe_prevista]` — seleciona as colunas por `pipeline.feature_names_in_`; `risco_nao_alf = 1 − prob`; `classe_prevista ∈ {"alfabetizado", "nao_alfabetizado"}`.
- CLI: `python -m src.modeling.predict --regime producao --modelo hgb [--entrada data/processed/base_modelagem_aluno.parquet] [--saida reports/predicoes_producao_hgb.csv] [--so-teste]`. Sem `--limiar`, lê o do `reports/metricas_<regime>_<modelo>.json`.

- [ ] **Step 1: Teste**

`tests/test_predict.py`:
```python
import json

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import config
from src.modeling import predict, train
from tests.conftest import montar_base_sintetica


@pytest.fixture
def treinado(lake_tmp):
    base = montar_base_sintetica(n_escolas=30, alunos_por_escola=15)
    r = train.treinar(base, "producao", modelo="logistica")
    train.salvar(r, base)
    config.PROCESSED.mkdir(exist_ok=True)
    base.to_parquet(config.PROCESSED / "base_modelagem_aluno.parquet")
    return base, r


def test_pontuar_reproduz_a_metrica_gravada(treinado):
    base, r = treinado
    pipe = predict.carregar_modelo(regime="producao", modelo="logistica")
    teste = base.iloc[r["partes"]["teste"]]
    pred = predict.pontuar(pipe, teste, limiar=r["metricas"]["limiar"])
    assert list(pred.columns) == ["id_aluno", "id_escola", "id_municipio", "sigla_uf", "prob_alfabetizado", "risco_nao_alf", "classe_prevista"]
    assert roc_auc_score(teste["alfabetizado"], pred["prob_alfabetizado"]) == pytest.approx(r["metricas"]["teste"]["roc_auc"])
    assert set(pred["classe_prevista"]) <= {"alfabetizado", "nao_alfabetizado"}


def test_cli_so_teste_usa_limiar_do_json(treinado):
    base, r = treinado
    predict.main(["--regime", "producao", "--modelo", "logistica", "--so-teste"])
    saida = pd.read_csv(config.REPORTS / "predicoes_producao_logistica.csv")
    assert len(saida) == len(r["partes"]["teste"])
    limiar = json.loads((config.REPORTS / "metricas_producao_logistica.json").read_text())["limiar"]
    assert ((saida["prob_alfabetizado"] < limiar) == (saida["classe_prevista"] == "nao_alfabetizado")).all()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_predict.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar**

`src/modeling/predict.py`:
```python
"""Pontua uma base nova com o pipeline salvo.

    python -m src.modeling.predict --regime producao --modelo hgb --so-teste
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config

ID_COLS = ["id_aluno", "id_escola", "id_municipio", "sigla_uf"]


def carregar_modelo(caminho: Path | None = None, regime: str = "producao", modelo: str = "hgb"):
    return joblib.load(caminho or config.MODELS / f"modelo_aluno_{regime}_{modelo}.joblib")


def pontuar(pipeline, base: pd.DataFrame, limiar: float = 0.5) -> pd.DataFrame:
    X = base[list(pipeline.feature_names_in_)]
    proba = pipeline.predict_proba(X)[:, 1]
    out = base[ID_COLS].reset_index(drop=True).copy()
    out["prob_alfabetizado"] = proba
    out["risco_nao_alf"] = 1 - proba
    out["classe_prevista"] = np.where(proba >= limiar, "alfabetizado", "nao_alfabetizado")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="producao")
    ap.add_argument("--modelo", default="hgb")
    ap.add_argument("--caminho-modelo", type=Path)
    ap.add_argument("--entrada", type=Path, default=None)
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--limiar", type=float, default=None)
    ap.add_argument("--so-teste", action="store_true", help="pontua só a partição de teste gravada pelo train.py")
    args = ap.parse_args(argv)

    base = pd.read_parquet(args.entrada or config.PROCESSED / "base_modelagem_aluno.parquet")
    if args.so_teste:
        part = pd.read_parquet(config.MODELS / f"particao_{args.regime}.parquet")
        base = base[base["id_aluno"].isin(part.loc[part["parte"] == "teste", "id_aluno"])]
    limiar = args.limiar
    if limiar is None:
        limiar = json.loads((config.REPORTS / f"metricas_{args.regime}_{args.modelo}.json").read_text())["limiar"]

    pred = pontuar(carregar_modelo(args.caminho_modelo, args.regime, args.modelo), base, limiar)
    saida = args.saida or config.REPORTS / f"predicoes_{args.regime}_{args.modelo}.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(saida, index=False)
    print(f"{len(pred):,} alunos pontuados (limiar {limiar:.3f}); {(pred['classe_prevista'] == 'nao_alfabetizado').mean():.1%} previstos não alfabetizados -> {saida}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_predict.py -v`
Expected: 2 passed

- [ ] **Step 5: Gerar figuras de interpretação com o modelo real**

Script curto (não versionado, ou célula do notebook 02): carregar `modelo_aluno_producao_hgb`, partição de teste, `importancia_permutacao` (n_amostra 100 mil), `explicar_shap` (10 mil), e salvar `images/importancia_permutacao_producao.png`, `images/shap_beeswarm_producao.png`, `images/shap_dependence_<top5>.png`, `images/shap_por_regiao.png`, o mesmo para diagnóstico (`images/shap_beeswarm_diagnostico.png`), `images/roc_pr_modelos.png`, `images/calibracao_producao.png`, `images/metricas_por_uf.png`. Isso fica formalizado no notebook 02 (Task 18); aqui só se confirma que roda.

```bash
.venv/Scripts/python.exe -m src.modeling.predict --regime producao --modelo hgb --so-teste
```
Expected: ~280 mil linhas em `reports/predicoes_producao_hgb.csv` (partição de teste; ~10 MB, aceitável).

- [ ] **Step 6: Commit e PR**

```bash
git add src/modeling/predict.py tests/test_predict.py reports/predicoes_producao_hgb.csv
git commit -m "pontuacao de base nova com o pipeline salvo"
git push -u origin feature/interpretacao
```
PR `feature/interpretacao` → `develop`. Descrição: as 5 features mais importantes por permutação e por SHAP, sinal dos coeficientes da logística como conferência, o que o regime diagnóstico acrescenta (ganho de AUC e o peso de `taxa_escola_loo`).

---

### Task 16: Modelo B — risco de não atingir a meta (`src/modeling/risco_municipio.py`)

**Files:**
- Create: `src/modeling/risco_municipio.py`, `tests/test_risco_municipio.py`
- Outputs: `models/modelo_b_regressao.joblib`, `models/modelo_b_classificacao.joblib`, `reports/metricas_modelo_b.json`, `reports/ranking_risco_municipios.csv`

**Interfaces:**
- Grão: município × rede municipal (grão da meta). Treino: features do ano 2023 → alvos de 2024 (`taxa_prox` regressão; `nao_atingiu_prox` classificação, linhas `sem_meta` excluídas da classificação, `indistinguivel` conta como 0 e é reportado à parte). Aplicação: features de 2024 → estimativa 2025 contra `meta_prox` (= `meta_alfabetizacao_2025`).
- `NAO_FEATURES_B = {"taxa_prox", "situacao_meta_prox", "nao_atingiu_prox", "ano", "id_municipio", "nome_municipio", "meta_ano", "gap", "atingiu_meta", "situacao_meta"}` — as quatro últimas são NaN/`sem_meta` em 2023 inteiro (mudariam de distribuição entre treino e aplicação). `meta_prox` **entra** (pactuada antes do resultado). `CATEGORICAS_B = ["sigla_uf", "regiao"]`.
- `colunas_b(base) -> (num, cat)`, `preparar_treino(base, ano=2023) -> tuple[X, y_reg, y_clf, ids]` (`y_clf` com NaN onde `sem_meta`).
- `avaliar(X, y_reg, y_clf, seed=SEED) -> pd.DataFrame[modelo, tarefa, metrica, media, dp]` — `RepeatedKFold(5, 3)`: `Ridge` × `HistGradientBoostingRegressor` (MAE, RMSE, R²) e `LogisticRegression` × `HistGradientBoostingClassifier` (ROC-AUC, PR-AUC, Brier). Baseline ingênua para a regressão: prever `taxa_alfabetizacao` do próprio ano (persistência), reportada na mesma tabela.
- `treinar_final(X, y_reg, y_clf, seed=SEED) -> tuple[Pipeline, Pipeline]` (HGB nos dois).
- `gerar_ranking(base, pipe_reg, pipe_clf, ano_aplicacao=2024) -> pd.DataFrame` — colunas: `id_municipio, nome_municipio, sigla_uf, regiao, taxa_2024, ic95_2024, criancas_nao_alfabetizadas_2024, meta_2025, taxa_prevista_2025, gap_previsto_2025, prob_nao_atingir_2025, acima_da_margem` (gap previsto maior que `ic95`), `prioridade` (= `prob × criancas`), ordenado por `prioridade` decrescente.
- CLI: `python -m src.modeling.risco_municipio`.

- [ ] **Step 1: Branch**

```bash
git checkout -b feature/risco-e-clusters develop
```

- [ ] **Step 2: Helper de base municipal sintética em `tests/conftest.py`**

```python
def montar_base_municipio_sintetica(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Imita base_modelagem_municipio: 2023 com alvo de 2024, 2024 sem alvo; taxa_prox depende da taxa e do PIB."""
    rng = np.random.default_rng(seed)
    ids = 1100000 + np.arange(n)
    ufs = rng.choice(["RO", "BA", "SP"], n)
    regioes = {"RO": "Norte", "BA": "Nordeste", "SP": "Sudeste"}
    linhas = []
    for ano in (2023, 2024):
        taxa = np.clip(rng.normal(0.55, 0.15, n), 0.05, 0.98)
        pib = rng.normal(3, 0.5, n)
        for i in range(n):
            linhas.append({
                "ano": ano, "id_municipio": ids[i], "nome_municipio": f"m{i}", "sigla_uf": ufs[i], "regiao": regioes[ufs[i]],
                "alunos_com_nota": 200, "taxa_alfabetizacao": taxa[i], "ic95": 0.06, "meta_ano": np.nan if ano == 2023 else 0.6,
                "gap": np.nan, "atingiu_meta": None, "situacao_meta": "sem_meta" if ano == 2023 else "atingiu",
                "taxa_participacao": 0.9, "proficiencia_media": 700 + 100 * taxa[i], "criancas_nao_alfabetizadas": 200 * (1 - taxa[i]),
                **{f"pct_nivel_{k}": 1 / 9 for k in range(9)}, "pct_critico": 0.2, "pct_atencao": 0.2, "pct_quase_la": 0.1,
                "log_pib_per_capita": pib[i], "tdi_ai": rng.normal(8, 3) if rng.random() > 0.1 else np.nan,
                "taxa_alfabetizacao_adultos": 90.0, "pct_escolas_rurais": rng.random(), "log_populacao": rng.normal(9, 1),
                "densidade_demografica": rng.lognormal(3, 1), "meta_prox": 0.6 if ano == 2023 else 0.65,
            })
    df = pd.DataFrame(linhas)
    e23 = df["ano"] == 2023
    prox = np.clip(df.loc[e23, "taxa_alfabetizacao"] * 0.8 + 0.05 * df.loc[e23, "log_pib_per_capita"] + rng.normal(0, 0.05, e23.sum()), 0, 1)
    df.loc[e23, "taxa_prox"] = prox.to_numpy()
    df.loc[e23, "situacao_meta_prox"] = np.where(prox < 0.6 - 0.06, "nao_atingiu", np.where(prox < 0.6 + 0.06, "indistinguivel", "atingiu"))
    df.loc[e23 & (df["id_municipio"] < 1100000 + 10), "situacao_meta_prox"] = "sem_meta"
    df["nao_atingiu_prox"] = np.where(df["situacao_meta_prox"].isna(), np.nan, (df["situacao_meta_prox"] == "nao_atingiu").astype(float))
    return df
```

- [ ] **Step 3: Teste**

`tests/test_risco_municipio.py`:
```python
import json

import numpy as np

from src import config
from src.modeling import risco_municipio as rm
from tests.conftest import montar_base_municipio_sintetica


def test_preparar_treino_sem_alvo_nas_features():
    base = montar_base_municipio_sintetica()
    X, y_reg, y_clf, ids = rm.preparar_treino(base)
    assert len(X) == 200 and not set(X.columns) & rm.NAO_FEATURES_B
    assert "meta_prox" in X.columns and "taxa_alfabetizacao" in X.columns
    assert y_clf.isna().sum() == 10                            # sem_meta fica fora da classificação


def test_avaliar_bate_a_persistencia_e_treina_final():
    base = montar_base_municipio_sintetica()
    X, y_reg, y_clf, _ = rm.preparar_treino(base)
    res = rm.avaliar(X, y_reg, y_clf)
    assert {"persistencia", "ridge", "hgb_reg", "logistica", "hgb_clf"} <= set(res["modelo"])
    mae = res[(res.metrica == "mae")].set_index("modelo")["media"]
    assert mae["hgb_reg"] < mae["persistencia"] or mae["ridge"] < mae["persistencia"]
    reg, clf = rm.treinar_final(X, y_reg, y_clf)
    assert reg.predict(X.head(2)).shape == (2,) and clf.predict_proba(X.head(2)).shape == (2, 2)


def test_ranking_2025(lake_tmp):
    base = montar_base_municipio_sintetica()
    X, y_reg, y_clf, _ = rm.preparar_treino(base)
    reg, clf = rm.treinar_final(X, y_reg, y_clf)
    rank = rm.gerar_ranking(base, reg, clf)
    assert len(rank) == 200 and rank["id_municipio"].is_unique
    assert rank["prob_nao_atingir_2025"].between(0, 1).all()
    assert rank["prioridade"].is_monotonic_decreasing
    assert np.allclose(rank["gap_previsto_2025"], rank["taxa_prevista_2025"] - rank["meta_2025"])
    assert rank["meta_2025"].eq(0.65).all()


def test_cli(lake_tmp):
    config.PROCESSED.mkdir(exist_ok=True)
    montar_base_municipio_sintetica().to_parquet(config.PROCESSED / "base_modelagem_municipio.parquet")
    rm.main([])
    assert (config.REPORTS / "ranking_risco_municipios.csv").exists()
    m = json.loads((config.REPORTS / "metricas_modelo_b.json").read_text())
    assert "avaliacao" in m and m["n_treino"] == 200
```

- [ ] **Step 4: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_risco_municipio.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 5: Implementar**

`src/modeling/risco_municipio.py`:
```python
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

NAO_FEATURES_B = frozenset({
    "taxa_prox", "situacao_meta_prox", "nao_atingiu_prox", "ano", "id_municipio", "nome_municipio",
    "meta_ano", "gap", "atingiu_meta", "situacao_meta",
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
    t = base[(base["ano"] == ano) & base["taxa_prox"].notna()].reset_index(drop=True)
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
    linhas = []
    persist = mean_absolute_error(y_reg, X["taxa_alfabetizacao"])
    linhas.append({"modelo": "persistencia", "tarefa": "regressao", "metrica": "mae", "media": float(persist), "dp": 0.0})
    for nome, est in [("ridge", Ridge(alpha=1.0)), ("hgb_reg", HistGradientBoostingRegressor(random_state=seed, max_iter=300, learning_rate=0.05))]:
        r = cross_validate(_pipe(est, X), X, y_reg, cv=cv, scoring=METRICAS_REG, n_jobs=-1)
        linhas += _resumir(nome, "regressao", r, METRICAS_REG)
    com_meta = y_clf.notna()
    for nome, est in [("logistica", LogisticRegression(max_iter=1000)), ("hgb_clf", HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05))]:
        r = cross_validate(_pipe(est, X), X[com_meta], y_clf[com_meta].astype(int), cv=cv, scoring=METRICAS_CLF, n_jobs=-1)
        linhas += _resumir(nome, "classificacao", r, METRICAS_CLF)
    return pd.DataFrame(linhas)


def treinar_final(X, y_reg, y_clf, seed: int = config.SEED):
    reg = _pipe(HistGradientBoostingRegressor(random_state=seed, max_iter=300, learning_rate=0.05), X).fit(X, y_reg)
    com_meta = y_clf.notna()
    clf = _pipe(HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05), X).fit(X[com_meta], y_clf[com_meta].astype(int))
    return reg, clf


def gerar_ranking(base: pd.DataFrame, pipe_reg, pipe_clf, ano_aplicacao: int = 2024) -> pd.DataFrame:
    t = base[base["ano"] == ano_aplicacao].reset_index(drop=True)
    X = t[list(pipe_reg.feature_names_in_)].copy()
    out = pd.DataFrame({
        "id_municipio": t["id_municipio"], "nome_municipio": t.get("nome_municipio"), "sigla_uf": t["sigla_uf"], "regiao": t.get("regiao"),
        f"taxa_{ano_aplicacao}": t["taxa_alfabetizacao"], f"ic95_{ano_aplicacao}": t["ic95"],
        f"criancas_nao_alfabetizadas_{ano_aplicacao}": t["criancas_nao_alfabetizadas"],
        f"meta_{ano_aplicacao + 1}": t["meta_prox"],
        f"taxa_prevista_{ano_aplicacao + 1}": np.clip(pipe_reg.predict(X), 0, 1),
        f"prob_nao_atingir_{ano_aplicacao + 1}": pipe_clf.predict_proba(X)[:, 1],
    })
    p = ano_aplicacao + 1
    out[f"gap_previsto_{p}"] = out[f"taxa_prevista_{p}"] - out[f"meta_{p}"]
    out["acima_da_margem"] = out[f"gap_previsto_{p}"] < -out[f"ic95_{ano_aplicacao}"]
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
    rank.to_csv(config.REPORTS / "ranking_risco_municipios.csv", index=False)
    resumo = {
        "n_treino": int(len(X)), "n_com_meta": int(y_clf.notna().sum()), "n_indistinguivel_2024": int((base["situacao_meta_prox"] == "indistinguivel").sum()),
        "features": list(X.columns), "avaliacao": aval.to_dict("records"),
        "n_aplicacao_2024": int(len(rank)), "n_acima_da_margem": int(rank["acima_da_margem"].sum()),
        "prob_media_nao_atingir_2025": float(rank["prob_nao_atingir_2025"].mean()),
    }
    (config.REPORTS / "metricas_modelo_b.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(aval.pivot_table(index="modelo", columns="metrica", values="media").round(4))
    print(rank.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_risco_municipio.py -v`
Expected: 4 passed

- [ ] **Step 7: Rodar com dado real**

```bash
.venv/Scripts/python.exe -m src.modeling.risco_municipio
```
Expected: treino em ≈ 4.800 municípios (os de 2023 com resultado em 2024); MAE do HGB abaixo da persistência (a persistência é forte por causa da regressão à média, então a diferença pode ser pequena: reportar honestamente); ROC-AUC da classificação em torno de 0,80–0,90 (a taxa atual é muito informativa da meta). Ranking com 5.452 municípios; conferir que os primeiros são municípios grandes do Norte/Nordeste com `prob` alta, não municípios minúsculos com `ic95` largo.

- [ ] **Step 8: Commit**

```bash
git add src/modeling/risco_municipio.py tests/test_risco_municipio.py tests/conftest.py reports/metricas_modelo_b.json reports/ranking_risco_municipios.csv
git commit -m "modelo de risco de meta por municipio e ranking 2025"
```

---

### Task 17: Clusters de vulnerabilidade (`src/modeling/clusters.py`)

**Files:**
- Create: `src/modeling/clusters.py`, `tests/test_clusters.py`
- Outputs: `reports/avaliacao_k.csv`, `reports/perfis_clusters.csv`, `reports/clusters_municipios.csv`, `images/clusters_k.png`, `images/clusters_pca.png`

**Interfaces:**
- `COLUNAS_FORMA = [pct_nivel_0..8]`, `COLUNAS_CONTEXTO = ["taxa_alfabetizacao_adultos", "log_pib_per_capita", "pct_escolas_rurais", "tdi_ai", "log_populacao", "densidade_demografica"]`.
- `preparar_matriz(base, ano=2024) -> tuple[pd.DataFrame, pd.DataFrame]` — `(X, ids)`; `ids` com `id_municipio, nome_municipio, sigla_uf, regiao, taxa_alfabetizacao, criancas_nao_alfabetizadas`.
- `build_cluster_pipeline(k, seed=SEED) -> Pipeline` (`SimpleImputer(median)` → `StandardScaler` → `KMeans(n_init=10)`).
- `avaliar_k(X, ks=range(2, 9), seed=SEED) -> pd.DataFrame[k, inercia, silhueta]` (silhueta em amostra ≤ 5.000).
- `ajustar(X, k, seed=SEED) -> Pipeline`, `rotulos(pipeline, X) -> np.ndarray`, `projetar_pca(pipeline, X) -> np.ndarray (n, 2)`.
- `perfis(X, ids, labels) -> pd.DataFrame` — por cluster: `n`, `taxa_media`, `criancas_nao_alfabetizadas` (soma), `regiao_dominante`, `pct_regiao_dominante` e a média de cada coluna de `X`.
- CLI: `python -m src.modeling.clusters [--k 4]` (sem `--k`, usa o melhor por silhueta).

- [ ] **Step 1: Teste**

`tests/test_clusters.py`:
```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clusters.py -v`
Expected: FAIL com `ImportError`

- [ ] **Step 3: Implementar**

`src/modeling/clusters.py`:
```python
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
    regiao = g["regiao"].agg(lambda s: s.value_counts().index[0])
    pct_regiao = g["regiao"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    out = pd.DataFrame({
        "n": g.size(), "taxa_media": g["taxa_alfabetizacao"].mean(),
        "criancas_nao_alfabetizadas": g["criancas_nao_alfabetizadas"].sum(),
        "regiao_dominante": regiao, "pct_regiao_dominante": pct_regiao,
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
    k = args.k or int(aval.loc[aval["silhueta"].idxmax(), "k"])

    fig, (a1, a2) = plots.plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(aval["k"], aval["inercia"], "o-"); a1.set(title="Cotovelo", xlabel="k", ylabel="inércia")
    a2.plot(aval["k"], aval["silhueta"], "o-"); a2.set(title="Silhueta", xlabel="k")
    plots.salvar(fig, "clusters_k")

    pipe = ajustar(X, k)
    lab = rotulos(pipe, X)
    perfis(X, ids, lab).round(4).to_csv(config.REPORTS / "perfis_clusters.csv", index=False)
    ids.assign(cluster=lab).to_csv(config.REPORTS / "clusters_municipios.csv", index=False)
    plots.salvar(plots.plot_clusters_pca(projetar_pca(pipe, X), lab), "clusters_pca")
    print(f"k={k}"); print(perfis(X, ids, lab)[["cluster", "n", "taxa_media", "criancas_nao_alfabetizadas", "regiao_dominante"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clusters.py -v`
Expected: 3 passed

- [ ] **Step 5: Rodar com dado real e nomear os perfis**

```bash
.venv/Scripts/python.exe -m src.modeling.clusters
```
Expected: silhueta sugerindo k entre 3 e 5. Ler `reports/perfis_clusters.csv` e dar nome a cada cluster (ex.: "quase lá", "cauda crítica rural", "alto desempenho urbano", "desigual") — os nomes vão no notebook 04 e no README, não no código.

- [ ] **Step 6: Commit e PR**

```bash
git add src/modeling/clusters.py tests/test_clusters.py reports/avaliacao_k.csv reports/perfis_clusters.csv reports/clusters_municipios.csv images/clusters_k.png images/clusters_pca.png
git commit -m "clusters de vulnerabilidade por forma da distribuicao e contexto"
git push -u origin feature/risco-e-clusters
```
PR `feature/risco-e-clusters` → `develop`. Descrição: tabela de avaliação do modelo B (persistência × ridge × HGB), top 10 do ranking, k escolhido e os perfis nomeados.

---

### Task 18: Notebooks (EDA e modelagem executados)

**Files:**
- Create: `notebooks/01_eda.ipynb`, `notebooks/02_modelagem_aluno.ipynb`, `notebooks/03_risco_municipio.ipynb`, `notebooks/04_clusters.ipynb`
- Outputs: figuras em `images/` (lista abaixo)

**Regras:** todo notebook começa com `import sys; sys.path.insert(0, "..")` e `from src import config`; nenhuma lógica nova, só chamadas a `src/` e texto. Cada célula de figura chama `plots.salvar(fig, nome)`. Executar de ponta a ponta com `nbconvert` antes de commitar (as saídas ficam no arquivo: é o que o avaliador vê no GitHub).

- [ ] **Step 1: Branch**

```bash
git checkout -b feature/notebooks develop
```

- [ ] **Step 2: `01_eda.ipynb` — cada hipótese termina com uma decisão**

Seções (título → função usada → figura salva → **decisão**):
1. Universo e alvo: contagens 2023/2024, presentes com nota, balanceamento 60/40, `plot_distribuicao_proficiencia` → `images/eda_distribuicao_proficiencia.png`. Decisão: sem reamostragem; métricas para a classe 0.
2. H1 rede: taxa por `rede_nome` pareada por município (`base.groupby`) → `eda_h1_rede.png`. Decisão: manter `rede_nome`.
3. H2 socioeconômico: Spearman de `taxa_alfabetizacao_mun_t1` com `taxa_alfabetizacao_adultos` e `log_pib_per_capita`, por região (`plot_dispersao`) → `eda_h2_socioeconomico.png`. Decisão: manter, log no PIB pc.
4. H3 infraestrutura: correlação e boxplots por quartil de `pct_escolas_internet`, `pct_escolas_biblioteca`, `pct_escolas_esgoto_rede` → `eda_h3_infraestrutura.png`. Decisão: quais agregados ficam.
5. H4 distorção idade-série: `tdi_ai` × taxa → `eda_h4_tdi.png`.
6. H5 porte/ruralidade: `log_populacao`, `pct_escolas_rurais` × taxa, com `ic95_mun_t1` → `eda_h5_porte.png`. Decisão: manter porte; alerta sobre `ic95`.
7. H6 ausência: `taxa_participacao_mun_t1` × taxa; perfil dos ausentes (usa `carregar_alunos(2024, apenas_com_nota=False)`) → `eda_h6_ausencia.png`. Decisão: universo presentes com nota.
8. H7 gradiente regional: boxplot por `regiao` e por UF → `eda_h7_regiao.png`. Decisão: `regiao` categórica; checar depois no SHAP se as socioeconômicas absorvem.
9. H8 forma da distribuição: municípios com taxa parecida e `pct_nivel_*` diferentes → `eda_h8_forma.png`. Decisão: `pct_nivel_*` entram; base dos clusters.
10. H9 escola além do município: decomposição de variância (intra/entre escola) e `taxa_escola_loo` × alvo → `eda_h9_escola.png`. Decisão: regime diagnóstico; teto do modelo.
11. H10 `caderno`: taxa por caderno → `eda_h10_caderno.png`. Decisão: fora do modelo (artefato do instrumento); registrar se houver diferença.
12. Faltantes e colinearidade: mapa de NaN das externas (`base.isna().mean()`), matriz de correlação das numéricas, VIF das 15 mais correlacionadas → `eda_faltantes.png`, `eda_correlacao.png`. Decisão: `add_indicator=True`; remover colunas com |ρ| > 0,95 (listar).
13. Tabela final "hipótese → decisão" (markdown), copiada para o README.

- [ ] **Step 3: `02_modelagem_aluno.ipynb`**

1. Carrega `base_modelagem_aluno`, mostra `colunas_por_regime` nos dois regimes e a lista `COLUNAS_PROIBIDAS` (o inventário de vazamento em código).
2. Lê os `reports/metricas_*.json` (não retreina: o treino é o `train.py`) e monta a tabela dummy × logística × HGB × HGB-diagnóstico, ponderada e não ponderada → `plot_comparacao_modelos` → `images/modelos_comparacao.png`.
3. Curvas ROC/PR com as predições de `reports/predicoes_producao_hgb.csv` e da logística → `roc_pr_modelos.png`; calibração → `calibracao_producao.png`; matriz de confusão no limiar → `matriz_confusao_producao.png`; justificativa do limiar (recall mínimo 0,8 dos não alfabetizados).
4. Recortes por UF e por rede → `metricas_por_uf.png`, `metricas_por_rede.png`; municípios sem histórico × com histórico.
5. Interpretação: `importancia_permutacao` → `importancia_permutacao_producao.png`; `explicar_shap` → `shap_beeswarm_producao.png`, dependence das 5 principais → `shap_dependence_<feature>.png`, `shap_por_grupo` por região → `shap_por_regiao.png`; `coeficientes_logistica` como conferência de sinal; diagnóstico → `shap_beeswarm_diagnostico.png` e o ganho de AUC.
6. Texto: teto intra-escola, o que o modelo responde e o que não responde.

- [ ] **Step 4: `03_risco_municipio.ipynb` e `04_clusters.ipynb`**

03: lê `metricas_modelo_b.json` e `ranking_risco_municipios.csv`; tabela de avaliação; `plot_ranking_risco` → `ranking_risco_top20.png`; dispersão `taxa_2024` × `taxa_prevista_2025` com `meta_2025` → `risco_dispersao.png`; leitura: risco × volume × margem, regressão à média, o que o ranking não é ("culpa").
04: lê `avaliacao_k.csv`, `perfis_clusters.csv`, `clusters_municipios.csv`; nomeia os perfis; cruza com região e porte (`crosstab`) → `clusters_por_regiao.png`; perfis médios de `pct_nivel_*` por cluster → `clusters_perfis_niveis.png`.

- [ ] **Step 5: Executar todos e conferir**

```bash
for nb in 01_eda 02_modelagem_aluno 03_risco_municipio 04_clusters; do
  .venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/$nb.ipynb
done
ls images | wc -l
git status --short images reports
```
Expected: 4 notebooks executam sem erro; ≥ 25 PNGs em `images/`; nenhum arquivo novo em `data/`.

- [ ] **Step 6: Commit e PR**

```bash
git add notebooks images
git commit -m "notebooks de eda, modelagem, risco e clusters executados"
git push -u origin feature/notebooks
```
PR `feature/notebooks` → `develop`. Descrição: a tabela hipótese → decisão e os 4–6 achados quantificados que vão para o README.

---

### Task 19: README (11 seções), relatório técnico e PR final

**Files:**
- Modify: `README.md`
- Create: `reports/relatorio_tecnico.md`
- Modify: `docs/dicionario_base_modelagem.md` (se colunas mudaram nas tasks 8–17)

- [ ] **Step 1: Branch**

```bash
git checkout -b docs/readme develop
```

- [ ] **Step 2: `README.md` — as 11 seções do enunciado, nesta ordem, com estes conteúdos**

| Seção | Fonte do conteúdo |
|---|---|
| 1. Contexto do problema | Compromisso Nacional, corte 743, o Indicador; por que prever e não só medir; link para a Fase 2 |
| 2. Objetivo analítico | classificar aluno alfabetizado (modelo A); risco de meta 2025 (B); clusters; as 5 perguntas do enunciado |
| 3. Descrição da base | tabela Gold/Silver usadas, tabela das 7 fontes externas (coluna, ano de referência, publicação), universo, alvo, 60/40, `sem_historico`, o que a base não permite (pseudônimos, sem atributo individual), link para `docs/dicionario_base_modelagem.md` |
| 4. Etapas de modelagem | diagrama Silver+Gold+externas → feature store → split por escola → pipeline → tuning → teste único → interpretação; regimes produção × diagnóstico; inventário de vazamentos (tabela 3.3 da spec) com onde cada um é tratado no código |
| 5. Escolha do algoritmo | HGB × logística × dummy; por que HGB (volume, NaN nativo, não linearidade); o que ficou de fora e por quê |
| 6. Métricas de avaliação | tabela de `reports/metricas_*.json` (ROC-AUC, PR-AUC, F1/recall/precisão da classe 0, Brier, balanced accuracy), ponderada e não; limiar e justificativa; recortes UF/rede; CV por município; `images/modelos_comparacao.png`, `roc_pr_modelos.png`, `calibracao_producao.png` |
| 7. Interpretação dos resultados | `shap_beeswarm_producao.png`, `importancia_permutacao_producao.png`, dependence das 5 principais, sinais da logística, o que o diagnóstico acrescenta; modelo B: tabela de avaliação e importâncias; comparação A × B (pergunta 5) |
| 8. Insights encontrados | 4–6 achados com número (formato da Fase 2), tirados dos notebooks |
| 9. Limitações | teto intra-escola, uma onda treinável, pseudônimos, rede privada, regressão à média, `ic95`, municípios sem histórico, tuning com early stopping aleatório, IDEB ~3% faltante |
| 10. Aplicação prática para políticas públicas | ranking risco × volume (`ranking_risco_top20.png`), clusters como carteiras de intervenção, uso do limiar, o que o número não sustenta |
| 11. Possíveis evoluções futuras | onda 2025 para validação temporal real, FUNDEB, dados de aluno se o INEP liberar a chave, modelo hierárquico, deploy do `predict.py`, publicar a feature store na esteira AWS da Fase 2 |

Depois das 11 seções: estrutura do repositório, **como executar** (bloco abaixo), testes (`pytest`, contagem), versionamento (branches e PRs com link), autoria, link do vídeo (placeholder até a gravação).

Bloco "como executar" (ordem exata):
```bash
git clone https://github.com/tuanyfortunato/predicao-alfabetiza-brasil.git && cd predicao-alfabetiza-brasil
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt
cp .env.example .env                                    # FASE2_LAKE_PATH ou URL da Silver de alunos
.venv/Scripts/python.exe scripts/baixar_dados.py        # Silver de alunos (124 MB; Gold e externas já vêm no repo)
.venv/Scripts/python.exe -m src.preprocessing.feature_store
.venv/Scripts/python.exe -m src.modeling.train --regime producao --modelo hgb --params reports/melhores_params_producao_hgb.json
.venv/Scripts/python.exe -m src.modeling.train --regime diagnostico --modelo hgb --params reports/melhores_params_producao_hgb.json
.venv/Scripts/python.exe -m src.modeling.predict --regime producao --modelo hgb --so-teste
.venv/Scripts/python.exe -m src.modeling.risco_municipio
.venv/Scripts/python.exe -m src.modeling.clusters
.venv/Scripts/python.exe -m pytest
```
Observação obrigatória no README: como obter a Silver de alunos sem o lake local (a Fase 2 não tem Release publicada; criar uma Release no repositório da Fase 2 com o `alunos/` zipado e apontar a URL, ou documentar que é preciso rodar a Fase 2 antes).

- [ ] **Step 3: `reports/relatorio_tecnico.md`**

Seções: 1 arquitetura da solução (o diagrama da seção 4 do README com mais detalhe), 2 feature store (grãos, regras temporais, LOO, flags), 3 inventário de vazamentos e testes que o garantem (`test_features.py`, `test_split.py`, `test_feature_store.py`), 4 protocolo de validação (split, CV, tuning, teste único, seeds), 5 resultados completos (todas as tabelas dos JSON, inclusive recortes e calibração), 6 modelo B e clusters (métricas, k, perfis), 7 reprodutibilidade (versões, comandos, tempos medidos), 8 decisões e alternativas descartadas (FUNDEB, INSE, `caderno`, `peso_aluno` no treino, split por município como principal).

- [ ] **Step 4: Checklist final antes do PR**

```bash
.venv/Scripts/python.exe -m pytest -q                    # tudo verde; anotar a contagem no README
git ls-files | grep -E "^data/" | xargs du -ch | tail -1  # dados versionados < 15 MB
git ls-files | grep -E "\.joblib$|data/processed|data/silver/alunos" ; echo "(vazio acima = ok)"
grep -c "^## " README.md                                  # >= 11
```
Conferir no GitHub que todos os links do README (notebooks, imagens, dicionário, relatório) abrem.

- [ ] **Step 5: Commit, PR e merge em `main`**

```bash
git add README.md reports/relatorio_tecnico.md docs/dicionario_base_modelagem.md
git commit -m "readme com as 11 secoes e relatorio tecnico"
git push -u origin docs/readme
```
PR `docs/readme` → `develop`; após o merge, PR `develop` → `main` com o resumo das entregas (R1–R18 da spec, seção 1) e o link do vídeo quando gravado (R19, fora deste plano).

---

## Ordem de execução resumida

| Task | Branch | O que fica pronto |
|---|---|---|
| 1–3 | `feature/setup-e-dados` | esqueleto, Gold/Silver no lugar, leitura |
| 4–5 | `feature/enriquecimento-externo` | 7 externas em `data/external`, contexto externo por município |
| 6–7 | `feature/feature-store` | contexto t−1, LOO, `base_modelagem_*`, dicionário |
| 8–12 | `feature/pipeline-modelo` | features/leakage, pipeline, split, métricas, treino, tuning, `reports/metricas_*` |
| 13–15 | `feature/interpretacao` | SHAP, figuras, predict |
| 16–17 | `feature/risco-e-clusters` | modelo B, ranking 2025, clusters |
| 18 | `feature/notebooks` | 4 notebooks executados, `images/` |
| 19 | `docs/readme` | README (11 seções), relatório técnico, PR para `main` |
