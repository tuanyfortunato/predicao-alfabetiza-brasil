# predicao-alfabetiza-brasil

Modelo supervisionado que tenta prever se uma criança será considerada alfabetizada ao final do 2º ano do Ensino Fundamental, a partir de contexto educacional, territorial e socioeconômico. É a Fase 3 de um Tech Challenge individual (pós-graduação em Data Analytics).

## Contexto

Este projeto não parte do zero. A Fase 2 ([pipeline-dados-alfabetiza-brasil](https://github.com/tuanyfortunato/pipeline-dados-alfabetiza-brasil)) construiu uma pipeline Bronze/Silver/Gold na AWS a partir dos microdados do Compromisso Nacional Criança Alfabetizada e entregou uma camada Gold com indicadores por município, distribuição de proficiência e metas — 3,87 milhões de alunos, 79 mil escolas, 10,4 mil municípios.

A decisão aqui foi **não** continuar no repositório da Fase 2. Este é um repositório novo, que consome a Gold/Silver de lá como dado de entrada (copiado, não gerado aqui) e soma enriquecimento externo (IBGE, INEP) via BigQuery da Base dos Dados. A Fase 2 não é tocada.

Um ponto que molda todo o desenho: o aluno, na base, não tem atributo próprio — sem sexo, idade, raça, nível socioeconômico individual. Tudo que o modelo enxerga é contexto (escola, município, rede). Isso limita o teto de acerto por construção (a Fase 2 já tinha medido que 75% da variância da proficiência é intra-escola), e o projeto assume essa limitação como parte da entrega em vez de escondê-la atrás de uma métrica bonita.

## Estado atual

Em andamento, o modelo de aluno (produção) já treina, avalia e explica de ponta a ponta. Até agora:

- Esqueleto do repositório, configuração (`src/config.py`) e fixtures de teste.
- Gold (5 tabelas) e Silver leve (metas, resultados por município) trazidas da Fase 2 e commitadas.
- Silver de alunos (3,87 mi de linhas, ~124 MB) presente localmente, fora do Git.
- Oito fontes externas por município em `data/external/` (IBGE, INEP, Bolsa Família), cada uma com sua própria regra de defasagem.
- Contexto municipal t−1 e contexto de escola leave-one-out (`src/preprocessing/contexto.py`), feature store e dicionário de dados (`docs/dicionario_base_modelagem.md`).
- Guarda de leakage e pipeline sklearn único (imputação + encoding + modelo) por regime — produção e diagnóstico têm listas de colunas separadas em `src/preprocessing/features.py`.
- Split e validação cruzada por escola (`src/modeling/split.py`), nunca por aluno.
- Treino ponta a ponta do modelo de aluno (`src/modeling/train.py`): piso (dummy), baseline (regressão logística) e principal (HistGradientBoosting), com busca de hiperparâmetros por halving em amostra por escola (`src/modeling/tune.py`).
- Métricas ponderadas por `peso_aluno`, por UF, por rede e por faixa de calibração, com limiar escolhido para um recall mínimo de 80% na classe "não alfabetizado" (`src/evaluation/metrics.py`) — é essa classe que interessa não deixar passar, então o limiar sacrifica precisão de propósito.
- Permutation importance e SHAP para o HGB, coeficientes para a logística (`src/evaluation/interpret.py`).

Números do regime produção, no teste (278 mil alunos, 6.350 escolas, nunca vistas no treino): o HGB fica em ROC-AUC 0,66 e recall de 0,80/precisão de 0,48 na classe não alfabetizado, contra 0,50 do piso dummy — a regressão logística chega quase no mesmo AUC (0,66), o que sugere que o sinal disponível é majoritariamente linear. No regime diagnóstico (mesmas features + contexto de escola do próprio ano, leave-one-out) o AUC sobe para 0,68, uma diferença pequena que reforça o limite estrutural: sem atributo do aluno, o teto de acerto é mesmo baixo.

Ainda não feito: figuras (`src/visualization`), pontuação de base nova (`predict.py`), modelo de risco por município, clusters de vulnerabilidade e os notebooks (planejamento e especificação ficam fora do repositório público).

## Como rodar

Requer Python 3.11.

```bash
python3.11 -m venv .venv
.venv/Scripts/activate        # ou source .venv/bin/activate no Linux/Mac
pip install -r requirements.txt
cp .env.example .env           # preencher se for reextrair dados externos
pytest
```

Os dados leves já vêm no repositório: Gold da Fase 2 (`data/gold/`), Silver de metas e resultados (`data/silver/*.parquet`) e as oito fontes externas agregadas por município (`data/external/`). Só a Silver de alunos (124 MB) precisa ser copiada do lake da Fase 2:

```bash
python -m scripts.baixar_dados            # precisa de FASE2_LAKE_PATH no .env
```

Para reextrair as externas (só se quiser atualizar; exige credencial GCP e os CSVs do MDS):

```bash
python -m scripts.extrair_externas        # 7 fontes da Base dos Dados (BigQuery)
python -m scripts.baixar_bolsa_familia    # Bolsa Família, dezembro de cada ano
```

## Estrutura

```
src/
├── config.py           # caminhos e constantes do projeto (seed, corte de alfabetização, colunas proibidas)
├── preprocessing/       # leitura, contexto externo (t-1), contexto municipal e de escola (LOO), feature store, features/leakage, pipeline sklearn
├── modeling/            # split por escola, treino ponta a ponta, tuning (halving)
├── evaluation/          # métricas ponderadas/por recorte, interpretabilidade (permutation importance, SHAP)
└── visualization/       # ainda vazio — figuras do README e do relatório
data/
├── gold/                # camada Gold da Fase 2 (commitada)
├── silver/              # metas e resultados por município (commitados); alunos fica fora do Git
├── external/             # fontes IBGE/INEP/Bolsa Família extraídas do BigQuery e do MDS (commitadas)
└── processed/            # feature store gerada localmente, fora do Git
models/                  # pipelines treinados e partições (fora do Git)
reports/                 # métricas, resultados da busca de hiperparâmetros (commitados)
docs/
└── dicionario_base_modelagem.md   # colunas da feature store, escala e regra de defasagem de cada uma
notebooks/               # EDA e modelagem (ainda vazio)
```

## Por que essas escolhas

- **scikit-learn puro**, sem framework de deep learning: o volume (≈1,8 mi de linhas treináveis) e a natureza tabular do problema não justificam a complexidade extra.
- **Pipeline sklearn end-to-end** (imputação, encoding e modelo num único `Pipeline`): evita vazamento de estatísticas do teste para o treino e serializa o pré-processamento junto do modelo.
- **Split por escola**, não por aluno: alunos da mesma escola compartilham contexto; deixá-los em lados diferentes do split infla a métrica de validação artificialmente.
- **Regime "produção" vs "diagnóstico"**: features de contexto do ano anterior (o que dá para saber antes da prova) separadas de features do mesmo ano com leave-one-out (o que a escola explica, usado só para interpretação).
- **Limiar calibrado para recall, não para acurácia**: o problema é achar quem corre risco de não se alfabetizar, então o limiar de decisão é escolhido para garantir 80% de recall na classe não alfabetizado, mesmo custando precisão. Acurácia bruta em cima de um corte 0,5 seria uma métrica bonita e inútil para esse uso.
