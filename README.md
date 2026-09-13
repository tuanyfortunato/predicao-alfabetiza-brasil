# predicao-alfabetiza-brasil

Modelo supervisionado que tenta prever se uma criança será considerada alfabetizada ao final do 2º ano do Ensino Fundamental, a partir de contexto educacional, territorial e socioeconômico. É a Fase 3 de um Tech Challenge individual (pós-graduação em Data Analytics).

## Contexto

Este projeto não parte do zero. A Fase 2 ([pipeline-dados-alfabetiza-brasil](https://github.com/tuanyfortunato/pipeline-dados-alfabetiza-brasil)) construiu uma pipeline Bronze/Silver/Gold na AWS a partir dos microdados do Compromisso Nacional Criança Alfabetizada e entregou uma camada Gold com indicadores por município, distribuição de proficiência e metas — 3,87 milhões de alunos, 79 mil escolas, 10,4 mil municípios.

A decisão aqui foi **não** continuar no repositório da Fase 2. Este é um repositório novo, que consome a Gold/Silver de lá como dado de entrada (copiado, não gerado aqui) e soma enriquecimento externo (IBGE, INEP) via BigQuery da Base dos Dados. A Fase 2 não é tocada.

Um ponto que molda todo o desenho: o aluno, na base, não tem atributo próprio — sem sexo, idade, raça, nível socioeconômico individual. Tudo que o modelo enxerga é contexto (escola, município, rede). Isso limita o teto de acerto por construção (a Fase 2 já tinha medido que 75% da variância da proficiência é intra-escola), e o projeto assume essa limitação como parte da entrega em vez de escondê-la atrás de uma métrica bonita.

## Estado atual

Em andamento. Até agora:

- Esqueleto do repositório, configuração (`src/config.py`) e fixtures de teste.
- Gold (5 tabelas) e Silver leve (metas, resultados por município) trazidas da Fase 2 e commitadas.
- Silver de alunos (3,87 mi de linhas, ~124 MB) presente localmente, fora do Git.
- Oito fontes externas por município em `data/external/` (IBGE, INEP, Bolsa Família), com scripts de extração testados.

Ainda não feito: feature store, pipeline de treino, avaliação, interpretabilidade e os notebooks. O plano completo, tarefa por tarefa, está em [`docs/plano-implementacao.md`](docs/plano-implementacao.md); a especificação original (o que o enunciado pede) está em [`docs/planejamento-fase3.md`](docs/planejamento-fase3.md).

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
├── preprocessing/       # feature store: leitura, contexto externo, contexto municipal, guarda de leakage
├── modeling/            # split, treino, tuning, predição
├── evaluation/          # métricas e interpretabilidade (SHAP)
└── visualization/       # figuras usadas no README e no relatório
data/
├── gold/                # camada Gold da Fase 2 (commitada)
├── silver/              # metas e resultados por município (commitados); alunos fica fora do Git
├── external/             # fontes IBGE/INEP extraídas do BigQuery (a implementar)
└── processed/            # feature store gerada localmente, fora do Git
docs/                    # especificação e plano de implementação
notebooks/               # EDA e modelagem
```

## Por que essas escolhas

- **scikit-learn puro**, sem framework de deep learning: o volume (≈1,8 mi de linhas treináveis) e a natureza tabular do problema não justificam a complexidade extra.
- **Pipeline sklearn end-to-end** (imputação, encoding e modelo num único `Pipeline`): evita vazamento de estatísticas do teste para o treino e serializa o pré-processamento junto do modelo.
- **Split por escola**, não por aluno: alunos da mesma escola compartilham contexto; deixá-los em lados diferentes do split infla a métrica de validação artificialmente.
- **Regime "produção" vs "diagnóstico"**: features de contexto do ano anterior (o que dá para saber antes da prova) separadas de features do mesmo ano com leave-one-out (o que a escola explica, usado só para interpretação).

Detalhes de cada decisão — inclusive o inventário de vazamentos de dados tratados e por quê — estão documentados em `docs/planejamento-fase3.md` e `docs/plano-implementacao.md`.
