# Planejamento — Tech Challenge Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil

> Especificação da entrega da Fase 3, elaborada a partir do enunciado oficial (PDF `[IAST] - Tech Challenge - Fase 3.pdf`, 9 páginas) e do estado atual do repositório `pipeline-dados-alfabetiza-brasil` (Fase 2, mergeada em `main`). Documento interno, versionado neste repositório (`docs/` inteiro é versionado — ver adendo do `plano-implementacao.md`, item 9).
>
> Projeto individual. **Decisão revista após esta especificação:** o projeto passou a viver em um **repositório novo**, `predicao-alfabetiza-brasil`, e não no repositório da Fase 2. A Fase 2 não é alterada; este repo **consome** a Gold e a Silver dela (commitadas/copiadas) como dado de entrada. Onde este documento abaixo ainda descreve "mesmo repositório" ou crescimento in-place sobre a Fase 2 (seções 2.2, 4.2, 4.7 e 6), vale o `docs/plano-implementacao.md` — ver seu "Adendo à especificação", item 1.

---

## 0. Resumo em uma tela

**O que o enunciado pede.** Desenvolver um **modelo supervisionado que preveja se um aluno será considerado alfabetizado ou não**, usando variáveis educacionais, territoriais e socioeconômicas, a partir da camada Gold da Fase 2 (com enriquecimento externo permitido). O trabalho começa com **análise exploratória** que sustente hipóteses e decisões de modelagem, passa por uma **pipeline completa de Machine Learning em scikit-learn** (imputação, transformação de variáveis, encoding, tratamento de data leakage, pré-processamento acoplado ao modelo, otimização, validação replicável) e termina em **interpretabilidade** (Feature Importance, SHAP) e **aplicação estratégica** (cinco perguntas de negócio). Entrega: repositório com estrutura mínima definida, README com 11 seções, documentação técnica, pipeline reproduzível, análises/visualizações e **vídeo executivo de até 5 minutos** simulando reunião com gestores públicos.

**O que já existe.** A Fase 2 entregou a Gold com 5 tabelas (10,4 mil municípios, 79 mil escolas), a Silver com 3,87 mi de alunos, gabarito oficial validado, margem de erro (`ic95`) em toda taxa, 47 testes, DQ nas três camadas e a esteira na AWS. A seção "Aplicação em IA" do README da Fase 2 já desenhou exatamente o que esta fase executa: feature store no nível do aluno com leave-one-out e contexto defasado, predição de risco de meta por município, clusters de vulnerabilidade.

**A decisão central desta fase.** O aluno, na base, **não tem atributo próprio** (sem sexo, idade, raça, nível socioeconômico individual). Tudo que prediz é **contexto**: escola, município, UF e fontes externas por município. Logo, o modelo de aluno tem um teto natural (75% da variância da proficiência está *dentro* da escola, achado da Fase 2) e a honestidade sobre esse teto é parte da entrega, não uma falha. A solução é organizada em **um modelo obrigatório** (aluno) e **duas aplicações estratégicas** (risco de meta por município e clusters), todas alimentadas por uma **nova tabela Gold de modelagem** construída com as mesmas regras de leakage que a Fase 2 já documentou.

---

## 1. O que o enunciado pede — checklist rastreável

| # | Requisito do PDF | Obrigatório | Onde será atendido |
|---|---|---|---|
| R1 | Modelo supervisionado: aluno alfabetizado × não alfabetizado | **sim** | `src/modeling/`, `notebooks/modelagem_alfabetizacao.ipynb` |
| R2 | Dados da Gold da Fase 2 (indicador, metas, território, socioeconômico, educacional, população) | **sim** | Gold existente + nova `gold/base_modelagem_aluno` |
| R3 | Enriquecimento externo (IBGE, Censo Escolar, FUNDEB, PNAD, Atlas, CadÚnico) | opcional, mas "socioeconômico" e "territorial" do R1 exigem na prática | novas entidades na Bronze/Silver via Base dos Dados (seção 4) |
| R4 | EDA: distribuições, padrões, correlações, variáveis relevantes, hipóteses; EDA orienta modelagem | **sim** | `notebooks/eda_modelagem.ipynb`, `images/` |
| R5 | Pipeline sklearn: imputação numérica | **sim** | `SimpleImputer(median, add_indicator=True)` no `ColumnTransformer` |
| R6 | Pipeline sklearn: transformação de numéricas e categóricas (encoding) | **sim** | `StandardScaler` / `OneHotEncoder(handle_unknown="ignore")` |
| R7 | Tratamento de data leakage | **sim** | seção 3.3: inventário de vazamentos + teste unitário de guarda |
| R8 | Pré-processamento integrado ao modelo | **sim** | `Pipeline([("prep", ColumnTransformer), ("clf", ...)])` serializado inteiro |
| R9 | Treino, validação e teste separados; otimização; redução de overfitting | **sim** | split por grupo (escola), `StratifiedGroupKFold`, busca de hiperparâmetros, early stopping |
| R10 | Validação com replicabilidade e generalização | **sim** | seeds fixas, `requirements.txt` pinado, `train.py` de ponta a ponta, métricas em `reports/metricas.json` |
| R11 | Interpretabilidade: Feature Importance e SHAP | **sim** | `src/evaluation/interpret.py`, gráficos em `images/` |
| R12 | Perguntas de negócio (5) | **sim** | seção 5 |
| R13 | Estrutura mínima do repositório (`data`, `notebooks`, `src/{preprocessing,modeling,evaluation,visualization}`, `reports`, `images`, `requirements.txt`, `README.md`, `.gitignore`) | **sim** | seção 6 |
| R14 | Git: commits, branches, PRs, decisões documentadas | **sim** | seção 7: uma branch por frente, PR com justificativa |
| R15 | README com 11 seções (contexto, objetivo analítico, base, etapas, algoritmo, métricas, interpretação, insights, limitações, políticas públicas, evoluções) | **sim** | seção 8 |
| R16 | Documentação técnica | **sim** | `reports/relatorio_tecnico.md` + dicionário Gold atualizado |
| R17 | Pipeline reproduzível | **sim** | `python src/modeling/train.py` recria modelo e métricas |
| R18 | Análises e visualizações | **sim** | notebooks executados + `images/` |
| R19 | Vídeo executivo ≤ 5 min: problema, insights, valor estratégico, apoio a políticas públicas; simular reunião executiva | **sim** | seção 9 |

Não há exigência de nuvem nesta fase. A AWS da Fase 2 continua como evidência de que a Gold é reproduzível na esteira; o treino roda local (seção 4.6).

---

## 2. Ponto de partida: o que a Fase 2 entrega e o que muda

### 2.1 O que serve diretamente

| Ativo da Fase 2 | Uso na Fase 3 |
|---|---|
| `silver/alunos/` (3,87 mi linhas, particionado por ano) | **grão do alvo**: uma linha por aluno, com `alfabetizado` já calculado pela fonte |
| `gold/indicador_municipio` | contexto municipal defasado (taxa, participação, `ic95`, limites, `criancas_nao_alfabetizadas`) |
| `gold/distribuicao_proficiencia` | formato da curva por município (`pct_nivel_0..8`, `pct_critico`, `pct_quase_la`): features e base dos clusters |
| `gold/meta_vs_resultado` + `silver/metas` (colunas `meta_alfabetizacao_2024..2030`) | alvo e horizonte do modelo de risco de meta (seção 4.5) |
| `gold/perfil_escola` | `residuo` para análise, não para feature (id_escola muda por ano, ver 3.3) |
| `src/01_bronze/ingestao_batch_bigquery.py` (`ENTITIES`) | **reuso direto** para ingerir as fontes externas: mesmas credenciais, mesmo lake, mesmo DQ |
| `src/utils/data_quality.py`, `lake.py`, `logger.py` | DQ das novas entidades e da tabela de modelagem |
| `tests/` (47 testes, `importlib` para pacotes numerados) | padrão para os novos testes de features e de guarda contra leakage |
| `docs/dicionario_dados_gold.md` | ganha a seção da `base_modelagem_aluno` e das tabelas de enriquecimento |

### 2.2 O que muda no repositório

> **Superada pela decisão do repositório novo** (ver nota no topo do documento). Esta seção descrevia crescimento in-place sobre o repositório da Fase 2; vale a "Estrutura de arquivos" do `plano-implementacao.md`, que já nasce como repo próprio.

- Novas pastas exigidas pelo enunciado: `src/preprocessing`, `src/modeling`, `src/evaluation`, `src/visualization`, `reports/`, `images/`. As pastas numeradas da Fase 2 (`01_bronze`, `02_silver`, `03_gold`) permanecem.
- `requirements.txt` ganha `scikit-learn`, `shap`, `seaborn`, `joblib` (versões pinadas).
- README reescrito com a Fase 3 como assunto principal (decisão em aberto na seção 10 sobre onde fica o texto da Fase 2).
- Modelos treinados ficam em `models/` (fora do Git, regenerados pelo script) — a reprodutibilidade é o script, não o binário.

---

## 3. Sondagem do dado: o que o grão aluno realmente oferece

Números medidos na Silver atual (`data/silver/alunos`), setembro/2026:

| Métrica | Valor |
|---|---|
| Linhas totais | 3.867.589 (2023: 1.747.029 · 2024: 2.120.560) |
| Presentes com nota (universo do modelo) | 3.354.661 |
| Ausentes | 511.743 (+ 1.185 presentes sem nota) |
| Alvo `alfabetizado` da fonte × regra `proficiencia >= 743` | **coincidência exata**: 1.984.546 sim / 1.370.115 não, zero discordâncias |
| Proporção de alfabetizados (presentes com nota) | 58,4% em 2023 · 59,8% em 2024 |
| Rede | municipal 3.432.166 · estadual 435.398 · privada 25 |
| Escolas distintas | 36.768 (2023) · 42.497 (2024) |
| Municípios distintos | 4.872 (2023) · 5.519 (2024) |
| Proficiência (presentes) | 578,5 a 904,4 · média 748,4 · mediana 753,2 · dp 47,6 |
| Colunas do aluno | `caderno`, `serie` (constante = 2), `rede`, `presenca`, `preenchimento_caderno`, `alfabetizado`, `proficiencia`, `peso_aluno`, `id_escola`, `id_municipio`, `sigla_uf` |

### 3.1 Consequências para o desenho

1. **Classes razoavelmente balanceadas (≈60/40).** Não precisa de reamostragem (SMOTE etc.); basta métrica adequada (ROC-AUC, PR-AUC, F1 da classe "não alfabetizado") e, se for o caso, `class_weight`.
2. **O aluno não tem atributo individual.** `caderno` é o caderno de prova (artefato do instrumento, não característica da criança), `serie` é constante. Sobram `rede` e a geografia. **Toda a capacidade preditiva vem de contexto** — o modelo responde "dado o contexto em que a criança estuda, qual a probabilidade de estar alfabetizada?". Isso é útil para política (identifica contextos de risco), mas o teto é baixo por construção: 75% da variância é intra-escola (Fase 2). Expectativa realista: **ROC-AUC entre 0,65 e 0,72**. O relatório deve dizer isso antes de mostrar a métrica, não depois.
3. **Ausente não é "não alfabetizado".** A fonte codifica `alfabetizado = 0` para os 512.928 sem nota. O universo do modelo é **presentes com nota**; a ausência vira análise (H6, seção 4.3), não classe.
4. **`peso_aluno` é peso amostral, não feature.** Entra como `sample_weight` opcional e nas métricas ponderadas (que estimam a população), reportadas ao lado das não ponderadas.
5. **Volume.** ~1,8 mi de linhas treináveis em 2024 (estimativa: 2.120.560 × ~87% de presença). Cabe em `HistGradientBoostingClassifier` sem cluster; busca de hiperparâmetros em subamostra (≈300 mil) e refit na base completa; SHAP em amostra de 10 a 20 mil.

### 3.2 Regimes de contexto: "produção" × "diagnóstico"

Há duas formas legítimas de dar contexto ao aluno, e elas respondem perguntas diferentes:

| Regime | Features de contexto | O que responde | Uso |
|---|---|---|---|
| **Produção** (principal) | tudo que existe **antes da prova**: Gold do município no ano anterior (t−1), externas ≤ t−1 ou estruturais (Censo 2022), `rede`, UF, metas pactuadas | "consigo antecipar risco antes de medir?" | modelo entregue, ranking de risco |
| **Diagnóstico** (complementar) | produção **+ contexto da escola no mesmo ano com leave-one-out** (taxa e proficiência média da escola excluindo o próprio aluno, nº de alunos, participação) | "quanto a escola explica além do município?" | interpretabilidade, argumento de política (o problema é a escola) |

Consequência: só **2024** tem contexto defasado (a base tem duas ondas), então o conjunto treinável do regime produção é o ano de 2024. 2023 serve à EDA e à validação do modelo municipal.

### 3.3 Inventário de vazamentos (tratamento obrigatório, R7)

| Vazamento | Por quê | Tratamento |
|---|---|---|
| `proficiencia` | define o alvo (≥ 743) | excluída; **teste unitário garante** que nunca entra na matriz de features |
| `presenca`, `presente`, `sem_nota`, `preenchimento_caderno` | determinam se o alvo existe; ausente = 0 na fonte | universo restrito a presentes com nota; colunas excluídas |
| Agregados de escola/município do mesmo ano | incluem o próprio aluno | leave-one-out: `(soma − valor_do_aluno) / (n − 1)`, só no regime diagnóstico |
| Gold do mesmo ano (`taxa_alfabetizacao` 2024 do município) | é o alvo agregado | contexto municipal sempre de t−1 |
| `gap`, `atingiu_meta`, `situacao_meta` de 2024 | derivados do resultado 2024 | fora; `meta_ano` (pactuada antes) pode entrar |
| `id_escola` como feature ou chave de lag | pseudônimo regerado a cada ano (Fase 2, aviso 1) | nunca como categoria; lag de escola é impossível |
| `id_municipio` como categoria | 5,5 mil níveis, target encoding vazaria | representado só pelos agregados defasados e externos |
| Split aleatório por aluno com features de escola | alunos da mesma escola dos dois lados do split | **split por grupo** (`GroupShuffleSplit` / `StratifiedGroupKFold` por `id_escola`) |
| Imputação/escala ajustadas na base inteira | estatísticas do teste contaminam o treino | tudo dentro do `Pipeline`, `fit` só no treino |

---

## 4. Desenho da solução

### 4.1 Três produtos, uma base

```
Silver alunos + Gold (t-1) + externas por município
                 │
                 ▼
   gold/base_modelagem_aluno  (nova, grão aluno, 2024)      ─┐
   gold/base_modelagem_municipio (nova, grão ano×município) ─┼─ feature store
                 │                                            │
      ┌──────────┼──────────────────┐                        │
      ▼          ▼                  ▼                        │
 Modelo A     Modelo B          Clusters                     │
 aluno        risco de meta     vulnerabilidade              │
 (R1, foco)   (perguntas 2 e 4) (pergunta 3)                 │
```

Prioridade: **A** é o obrigatório e recebe o grosso do esforço. **B** e **clusters** existem para responder as perguntas de negócio e reusam a mesma feature store e as mesmas utilidades; cada um cabe num notebook curto.

### 4.2 Enriquecimento externo: já está no BigQuery da Fase 2

> **Superada pela decisão do repositório novo:** a implementação não reaproveita `ingestao_batch_bigquery.py` da Fase 2 via `ENTITIES` — é um script próprio deste repo, `scripts/extrair_externas.py` (Task 4 do `plano-implementacao.md`), que grava direto em `data/external/` (commitado). O levantamento de fontes abaixo continua valendo como sondagem; o "como" mudou.

Sondagem feita em setembro/2026 no projeto `basedosdados` (metadados apenas). O que existe, o que serve e o custo de leitura:

| Dataset / tabela | Conteúdo útil | Anos | Tamanho | Uso |
|---|---|---|---|---|
| `br_ibge_censo_2022.municipio` | `taxa_alfabetizacao` (adultos), `idade_mediana`, `indice_envelhecimento`, `area`, `populacao`, `domicilios`, `populacao_indigena`, `populacao_quilombola` | 2022 (estrutural) | 1 MB | **socioeconômico principal**: alfabetização adulta, densidade, envelhecimento |
| `br_ibge_censo_2022.alfabetizacao_grupo_idade_sexo_raca` | alfabetização por cor/raça, sexo e idade | 2022 | 50 MB | opcional: desigualdade racial no município |
| `br_ibge_pib.municipio` | `pib`, `va_agropecuaria/industria/servicos/adespss`, `impostos_liquidos` | 2002–2023 | 8 MB | PIB per capita e perfil econômico (usar 2021/2022, defasado) |
| `br_ibge_populacao.municipio` | `populacao` | 1991–2025 | 6 MB | denominador do PIB pc, porte do município |
| `br_inep_censo_escolar.escola` | infraestrutura (água, esgoto, internet, biblioteca, laboratório), `tipo_localizacao` (rural/urbana), rede, etapas | 2007–2024, particionada por ano | 6,6 GB total (filtrar ano 2023 + colunas ≈ centenas de MB) | **agregado por município** (não junta por escola: `id_escola` da Alfabetiza é pseudônimo): % escolas com internet, % rurais etc. |
| `br_inep_indicadores_educacionais.municipio` | `tdi_*` (distorção idade-série), `atu_*` (alunos por turma), `had_*` (horas-aula), docentes com superior, adequação da formação | 2006–2024, particionada | 1,2 GB (filtrar ano) | indicadores educacionais complementares dos anos iniciais (filtrar `localizacao = total`, `rede = pública`) |
| `br_inep_ideb.municipio` | `ideb`, `taxa_aprovacao`, `nota_saeb_*`, `projecao` | 2005–2025 | 38 MB | IDEB anos iniciais 2021/2023 (rede pública) |
| `br_fnde_fundeb.indicador_municipal` | indicadores de financiamento por bimestre (`codigo_indicador`, `valor_percentual`, `valor_real`) | 2021+ | 321 MB, particionada | opcional: exige mapear `codigo_indicador` pelo `dicionario` |
| `br_bd_diretorios_brasil.municipio` | região, capital, microrregião, centroide | estrutural | pequeno | território: região, `capital_uf`, lat/long (confirmar colunas na ingestão) |
| `br_inep_indicadores_educacionais.escola_nivel_socioeconomico` (INSE) | nível socioeconômico por escola | só 2014–2015 | — | **descartado**: desatualizado e não junta por escola |
| CadÚnico, Bolsa Família, Atlas do Desenvolvimento Humano, PNAD Contínua | — | — | — | **não encontrados** no BigQuery público com esses nomes; ficam fora do escopo inicial (poderiam entrar por download manual, custo alto para o ganho) |

Todo o enriquecimento cabe no free tier de 1 TB/mês do BigQuery. Regra temporal: **feature externa do ano t usa dado publicado até t−1** ou estrutural (Censo 2022). IDEB 2023 é divulgado em 2024, antes da prova de novembro de 2024: aceitável no regime produção, com a data de publicação anotada no dicionário.

Implementação: cada tabela vira uma **entrada em `ENTITIES`** de `ingestao_batch_bigquery.py` (com `SELECT` de colunas e filtro de ano para as particionadas, em vez de `SELECT *`), passa pela Silver (tipos, chaves, agregação do Censo Escolar por município) e recebe checks de DQ (chave IBGE com 7 dígitos, unicidade por `(ano, id_municipio)`, cobertura de municípios). A arquitetura Medalhão da Fase 2 é preservada.

### 4.3 EDA (R4): hipóteses que viram decisões de modelagem

Cada hipótese termina com uma **decisão** (manter/remover/transformar feature), que é o que o enunciado cobra ("a EDA deve apoiar diretamente as decisões de modelagem").

| Hipótese | Como testar | Decisão que alimenta |
|---|---|---|
| H1. Rede estadual e municipal têm taxas diferentes dentro do mesmo município | taxa por rede, pareada por município | manter `rede` como categórica |
| H2. Alfabetização adulta (Censo 2022) e PIB per capita explicam a taxa municipal | dispersão + correlação de Spearman, por região | features socioeconômicas; log no PIB pc |
| H3. Infraestrutura escolar (internet, biblioteca, esgoto) do município se associa à taxa | correlação e boxplots por quartil | quais agregados do Censo Escolar entram |
| H4. Distorção idade-série nos anos iniciais antecipa não alfabetização | correlação `tdi_ef_anos_iniciais` × taxa | feature educacional complementar |
| H5. Porte/ruralidade: municípios pequenos e rurais têm taxa menor e mais incerta | taxa × `populacao`, × % escolas rurais, com `ic95` | feature de porte; alerta sobre `ic95` |
| H6. A ausência não é aleatória (correlação +0,29 da Fase 2) | participação × taxa; perfil dos ausentes | justifica universo "presentes com nota" e a análise de limites |
| H7. Existe gradiente regional forte (Norte/Nordeste × Sul/Sudeste) | mapa/boxplot por região | região como categórica; checar se socioeconômicas absorvem o efeito |
| H8. Formato da distribuição (`pct_nivel_*`) carrega informação além da média | comparar municípios de mesma taxa e curvas diferentes | features de forma; base dos clusters |
| H9. O que a escola explica além do município | decomposição de variância (reproduz Fase 2) e LOO | justifica o regime diagnóstico e o teto do modelo |

Também na EDA: distribuição da proficiência (bimodalidade?), mapa de valores faltantes das externas (quantos municípios sem match, onde), matriz de correlação e VIF para colinearidade, e a evolução 2023→2024 lida com o `ic95` (regressão à média).

### 4.4 Modelo A — aluno alfabetizado (R1, R5–R11)

**Alvo:** `alfabetizado` (1/0), presentes com nota, ano 2024.

**Features (regime produção):**

- Aluno: `rede` (cat), `sigla_uf` (cat), `regiao` (cat).
- Município t−1 (Gold 2023): `taxa_alfabetizacao`, `ic95`, `taxa_participacao`, `proficiencia_media`, `alunos_avaliados`, `criancas_nao_alfabetizadas`, `pct_critico`, `pct_atencao`, `pct_quase_la`, `pct_nivel_0..8`; `meta_2024` pactuada.
- Externas por município: alfabetização adulta, idade mediana, índice de envelhecimento, densidade (populacao/area), PIB per capita (log), composição do VA, % população indígena/quilombola, % escolas rurais, % escolas com internet/biblioteca/esgoto, alunos por turma, distorção idade-série, horas-aula, IDEB anos iniciais e taxa de aprovação, `capital_uf`.
- Flags de faltante: `sem_historico_2023` (≈650 municípios avaliados em 2024 sem linha em 2023) e indicadores gerados pelo `SimpleImputer(add_indicator=True)`.

**Regime diagnóstico** acrescenta: `taxa_escola_loo`, `prof_media_escola_loo`, `n_alunos_escola`, `taxa_participacao_escola`.

**Pipeline:**

```python
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
modelo = Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(random_state=42))])
```

(`StandardScaler` importa para a regressão logística; para árvores é inócuo e mantém um único pré-processamento para todos os modelos comparados.)

**Modelos comparados:** `DummyClassifier` (piso) → `LogisticRegression` (baseline interpretável, coeficientes) → `HistGradientBoostingClassifier` (principal: rápido em 1,8 mi de linhas, lida com não linearidade). `RandomForest` só se couber no tempo. Fica em scikit-learn puro, como o enunciado recomenda.

**Split e validação:** `GroupShuffleSplit` por `id_escola` em treino (70%) / validação (15%) / teste (15%), estratificado pelo alvo na medida do possível; `StratifiedGroupKFold(5)` dentro do treino para a busca de hiperparâmetros (`HalvingRandomSearchCV` em subamostra, refit na base completa, `early_stopping` no HGB). Teste tocado **uma vez**. Seeds fixas em todos os pontos. Também reportar `GroupKFold` por `id_municipio` como teste de generalização para municípios nunca vistos (mais duro, mais honesto para política).

**Métricas:** ROC-AUC, PR-AUC, F1 e recall da classe "não alfabetizado", balanced accuracy, Brier score e curva de calibração (a probabilidade é o produto, não a classe), matriz de confusão. Ponderadas por `peso_aluno` e não ponderadas. Recortes por UF e por rede (o modelo não pode ser bom só no Sudeste). Limiar de decisão escolhido por critério de negócio (ex.: recall mínimo de 80% dos não alfabetizados) e justificado.

**Interpretabilidade (R11):** permutation importance no teste; SHAP (`TreeExplainer`) em amostra: beeswarm global, dependence plots das 5 principais, comparação por região; coeficientes da logística como conferência de sinal. Comparação produção × diagnóstico mostra quanto a escola adiciona.

**Reprodutibilidade (R10, R17):** `python src/modeling/train.py --regime producao` lê a Gold, treina, avalia e grava `models/modelo_aluno_<regime>.joblib` + `reports/metricas_<regime>.json` + figuras em `images/`. `src/modeling/predict.py` pontua uma base nova. Testes: construção do LOO com massa pequena conferida à mão, guarda contra leakage (colunas proibidas nunca chegam ao `fit`), split sem escola em dois lados, `predict` reproduz métrica gravada.

### 4.5 Modelo B — risco de não atingir a meta (perguntas 2 e 4)

**Grão:** município, rede municipal (grão pactuado da meta).
**Treino:** features de 2023 (Gold + externas) → alvo 2024 (`taxa_alfabetizacao` como regressão, e `nao_atingiu` como classificação derivada de `situacao_meta`, tratando `indistinguivel` à parte). ~5 mil municípios: validação por `KFold` repetido, modelos `HistGradientBoostingRegressor`/`Classifier` e `Ridge` como baseline.
**Aplicação:** o modelo treinado em 2023→2024 é aplicado às features de **2024** para estimar a taxa de **2025**, confrontada com `meta_alfabetizacao_2025` da Silver de metas: é a resposta literal à pergunta "como prever municípios que podem não atingir metas futuras". Saída: `reports/ranking_risco_municipios.csv` com probabilidade, gap previsto, `ic95` e `criancas_nao_alfabetizadas` (para priorizar por volume, achado 2 da Fase 2).
**Limitação declarada:** uma única transição (2023→2024) treinável, regressão à média (correlação −0,45 da Fase 2). O ranking é de *risco*, não de *culpa*.

### 4.6 Clusters de vulnerabilidade (pergunta 3)

`KMeans` (k escolhido por silhueta e cotovelo) sobre `pct_nivel_0..8` + socioeconômicas padronizadas, PCA para visualização; perfis nomeados ("quase lá", "cauda crítica", "alto desempenho desigual"…), cruzados com região e porte. Sem exigência de rigor de modelo preditivo: é leitura estratégica.

### 4.7 Onde roda

Treino e notebooks rodam **local** (`.venv`, Python 3.11). Nada de treino na AWS: credencial de 4h e sem ganho para o volume.

> **Ajustada pela decisão do repositório novo:** não existe `src/03_gold/` neste repo; a feature store (`base_modelagem_aluno`/`base_modelagem_municipio`) nasce em `src/preprocessing/feature_store.py` e vive em `data/processed/` (fora do Git). Publicar essas tabelas na esteira AWS da Fase 2 fica só como opcional oportunista (seção 10, item 6), não como fluxo padrão.

---

## 5. Perguntas de negócio → artefato que responde

| Pergunta do enunciado | Resposta vem de | Artefato |
|---|---|---|
| Quais fatores mais impactam a alfabetização? | SHAP global + permutation importance do Modelo A; coeficientes da logística; H2–H5 da EDA | `images/shap_beeswarm.png`, seção "Interpretação" do README |
| Quais municípios apresentam maior risco educacional? | Modelo B (probabilidade de não atingir 2025) × volume (`criancas_nao_alfabetizadas`) × `ic95` | `reports/ranking_risco_municipios.csv`, mapa/tabela no README |
| Quais regiões possuem padrões semelhantes? | clusters sobre forma da distribuição + contexto | `images/clusters_*.png`, perfis descritos |
| Como prever municípios que podem não atingir metas futuras? | Modelo B aplicado às features 2024 → estimativa 2025 vs `meta_2025` | mesmo ranking, coluna `gap_previsto_2025` |
| Quais variáveis possuem maior influência nos modelos? | importâncias dos modelos A e B lado a lado; produção × diagnóstico | tabela comparativa no README |

---

## 6. Estrutura do repositório (proposta)

> **Superada pela decisão do repositório novo:** a árvore abaixo mapeava a estrutura mínima do PDF sobre o repositório *existente* da Fase 2 ("sem apagar nada da Fase 2"). Como o projeto passou a viver em `predicao-alfabetiza-brasil` (repo próprio, sem as pastas `01_bronze`/`02_silver`/`03_gold`/`terraform`/`.github` da Fase 2), vale a "Estrutura de arquivos" do `plano-implementacao.md`.

Mapeia a estrutura mínima do PDF sobre o que já existe, sem apagar nada da Fase 2:

```
├── README.md                     # reescrito: Fase 3 como assunto principal (ver seção 10)
├── requirements.txt              # + scikit-learn, shap, seaborn, joblib
├── .gitignore                    # + models/
├── data/                         # data lake local (existente; fora do Git)
├── docs/
│   ├── dicionario_dados_gold.md  # + base_modelagem_aluno, base_modelagem_municipio, entidades externas
│   ├── arquitetura*.png/.drawio  # diagrama ganha o bloco "Fase 3: feature store → modelos"
│   └── (Fase 2: demais docs)
├── notebooks/
│   ├── exploracao_bronze.ipynb, laboratorio_silver.ipynb,
│   ├── laboratorio_gold.ipynb, analise_gold.ipynb        # Fase 2, intocados
│   ├── eda_modelagem.ipynb                                # R4: hipóteses H1–H9 → decisões
│   ├── modelagem_alfabetizacao.ipynb                      # R1: Modelo A, produção × diagnóstico, SHAP
│   ├── risco_municipio.ipynb                              # Modelo B + ranking 2025
│   └── clusters_vulnerabilidade.ipynb                     # clusters
├── src/
│   ├── utils/, streaming/, 01_bronze/, 02_silver/         # Fase 2 (+ novas ENTITIES na Bronze, agregações na Silver)
│   ├── 03_gold/
│   │   ├── metricas_gold.py                               # Fase 2
│   │   └── base_modelagem.py                              # novas tabelas Gold de modelagem (LOO, lags, joins externos)
│   ├── preprocessing/
│   │   ├── features.py                                    # listas de colunas por regime, colunas proibidas, LOO
│   │   └── pipeline.py                                    # monta o ColumnTransformer + Pipeline
│   ├── modeling/
│   │   ├── train.py                                       # ponta a ponta, reproduzível
│   │   ├── tune.py                                        # busca de hiperparâmetros
│   │   └── predict.py                                     # pontua base nova com o joblib
│   ├── evaluation/
│   │   ├── metrics.py                                     # métricas ponderadas/não ponderadas, recortes
│   │   └── interpret.py                                   # permutation importance, SHAP
│   └── visualization/
│       └── plots.py                                       # estilo único das figuras (README + vídeo)
├── models/                       # joblib gerados (fora do Git)
├── reports/
│   ├── relatorio_tecnico.md      # documentação técnica (R16)
│   ├── metricas_producao.json, metricas_diagnostico.json
│   └── ranking_risco_municipios.csv
├── images/                       # figuras referenciadas no README
├── tests/                        # + test_features.py, test_pipeline.py, test_base_modelagem.py
├── scripts/, terraform/, .github/ # Fase 2, intocados
└── logs/                         # relatórios de DQ (fora do Git)
```

Os `src/preprocessing`, `modeling`, `evaluation`, `visualization` **não** levam prefixo numérico: são importados normalmente (`from src.modeling import train`), o que também evita a pegadinha do `importlib` da Fase 2.

---

## 7. Plano de execução (branches e PRs)

Uma branch por frente, integrada por PR em `develop` com descrição justificando as decisões; `develop` → `main` ao fim. Ordem sugerida, com o que cada PR precisa entregar para ser mergeável:

| # | Branch | Entrega | Critério de pronto |
|---|---|---|---|
| 1 | `feature/enriquecimento-externo` | novas `ENTITIES` (Censo 2022, PIB, população, Censo Escolar 2023 filtrado, indicadores educacionais, IDEB, diretórios; FUNDEB opcional); Silver com agregação por município; DQ | Bronze/Silver rodam de ponta a ponta local; cobertura ≥ 99% dos municípios avaliados; dicionário atualizado |
| 2 | `feature/gold-base-modelagem` | `base_modelagem.py`: tabelas aluno (2024, LOO, lags) e município (2023 e 2024); testes de LOO e de colunas proibidas | tabela gerada, DQ passa, testes verdes, dicionário com contrato |
| 3 | `feature/eda` | `eda_modelagem.ipynb` executado; H1–H9 com decisão explícita; figuras em `images/` | cada hipótese fecha com "decisão de modelagem" |
| 4 | `feature/modelo-aluno` | `preprocessing/`, `modeling/`, `evaluation/`; notebook de modelagem; `train.py` reproduzível; SHAP; métricas em `reports/` | `python src/modeling/train.py` reproduz métricas; testes de leakage e split verdes |
| 5 | `feature/risco-municipio` | Modelo B + ranking 2025 + clusters (dois notebooks curtos) | ranking gerado; perfis dos clusters descritos |
| 6 | `docs/readme-fase3` | README com as 11 seções, `relatorio_tecnico.md`, diagrama atualizado, requirements | checklist da seção 8 completo; links dos notebooks funcionam no GitHub |
| 7 | — | vídeo executivo, link no README | ≤ 5 min |

Esforço: 1 e 2 são engenharia (reuso alto, meio risco na agregação do Censo Escolar); 3 e 4 são o coração e consomem mais tempo; 5 é curto se a feature store estiver boa; 6 e 7 fecham.

---

## 8. README — as 11 seções exigidas (R15)

| Seção do PDF | Conteúdo previsto |
|---|---|
| Contexto do problema | Compromisso Nacional, corte 743, Indicador; por que prever e não só medir (reaproveita Fase 2, condensado) |
| Objetivo analítico | classificar aluno alfabetizado; antecipar risco por município; identificar fatores e padrões |
| Descrição da base | Gold da Fase 2 + enriquecimento (tabela da seção 4.2), universo, alvo, balanceamento, o que a base não permite |
| Etapas de modelagem | EDA → feature store → pipeline → split por grupo → tuning → avaliação → interpretação; regimes produção/diagnóstico |
| Escolha do algoritmo | por que HGB (volume, não linearidade, NaN nativo) contra logística (baseline) e o que não foi usado |
| Métricas de avaliação | tabela produção × diagnóstico × baseline; ponderadas/não; recortes UF/rede; calibração |
| Interpretação dos resultados | SHAP, importâncias, sinais; o que a escola adiciona |
| Insights encontrados | 4–6 achados quantificados (formato dos "5 achados" da Fase 2) |
| Limitações | teto intra-escola, uma onda treinável, pseudônimos, rede privada, regressão à média, amostragem e `ic95`, municípios sem histórico |
| Aplicação prática para políticas públicas | ranking por risco × volume, clusters como carteiras de intervenção, uso do limiar |
| Possíveis evoluções futuras | terceira onda (2025) para validação temporal real, dados de aluno (Censo Escolar por matrícula, se o INEP liberar a chave), modelo hierárquico, deploy do `predict.py` |

Mais: estrutura do repositório, como executar (ordem exata dos comandos), versionamento, autoria e link do vídeo.

---

## 9. Vídeo executivo (≤ 5 min)

Simula reunião com gestores públicos. Roteiro proposto (tempo aproximado):

1. **O problema** (45 s): 40% das crianças não alfabetizadas ao fim do 2º ano; meta 2030; gestor precisa antecipar, não só medir.
2. **A base e o método** (45 s): Gold da Fase 2 + IBGE/INEP; modelo que lê o contexto antes da prova.
3. **Os insights** (2 min): 3 achados com número: fatores dominantes (SHAP), municípios em risco para 2025 com volume de crianças, perfis de vulnerabilidade.
4. **O valor estratégico** (1 min): priorização por risco × volume, carteiras de intervenção por cluster, o que o número não sustenta (`ic95`).
5. **Próximos passos** (30 s): validação com a onda 2025, integração com a esteira.

Slides saem das figuras de `images/`; tom executivo, sem código.

---

## 10. Decisões em aberto

1. ~~**Onde fica o texto da Fase 2 no README.**~~ **Resolvida pela decisão do repositório novo:** não há texto da Fase 2 para mover — o README deste repo já nasce só com a Fase 3 e um link para `pipeline-dados-alfabetiza-brasil`.
2. **`caderno` como feature.** Por padrão fora (artefato de instrumento). Vale um teste na EDA: se a taxa varia por caderno, é sinal de não equivalência dos cadernos, o que é achado, não feature.
3. **`peso_aluno` no treino.** Reportar métricas ponderadas é obrigatório; treinar com `sample_weight` é opcional. Recomendação: treinar sem, avaliar com e sem, e comentar.
4. **FUNDEB.** Entra só se o mapeamento de `codigo_indicador` for direto; senão fica em "evoluções futuras".
5. **Clusters: escopo.** Manter enxuto (um notebook, KMeans + PCA). Não virar um quarto modelo.
6. **Publicar `base_modelagem_*` na AWS.** Só se houver sessão do Learner Lab sobrando; não é requisito.

---

## 11. Riscos

| Risco | Mitigação |
|---|---|
| Métrica "baixa" (AUC ~0,7) ser lida como modelo ruim | enquadrar o teto intra-escola antes da métrica; regime diagnóstico mostra o ganho da escola; comparar com baseline |
| Agregação do Censo Escolar por município consumir tempo | começar pelas colunas de infraestrutura mais citadas na literatura; o resto entra se sobrar tempo |
| Tempo de tuning em 1,8 mi de linhas | subamostra para busca, refit único; `early_stopping`; salvar resultados intermediários |
| Vazamento sutil pelo split | teste unitário que verifica ausência de escola em dois lados; `GroupKFold` por município como segunda prova |
| Municípios sem histórico 2023 (~650) | flag explícita + imputação; medir se o modelo é pior nesse grupo |
| Fontes externas com anos diferentes | dicionário registra ano e data de publicação de cada feature; regra t−1 aplicada por coluna |
