# Relatório técnico — predicao-alfabetiza-brasil

Documento técnico complementar ao `README.md`, voltado a um leitor avaliador do Tech Challenge (Fase 3, pós-graduação em Data Analytics). Todo número citado aqui vem de um arquivo real em `reports/*.json`, `reports/*.csv`, `src/config.py`, `requirements.txt` ou de uma execução de comando registrada nas seções 3 e 7 — nenhum é estimado de memória.

## 1. Arquitetura da solução

O pipeline é uma sequência linear de módulos, cada um consumindo a saída commitada ou gerada localmente do anterior, sem laços de realimentação:

```
Silver (aluno) + Gold (indicador/distribuição, t−1) + 8 externas (defasadas)
        │
        ▼
  feature store (data/processed/, gerada localmente)
        │
        ▼
  split por id_escola, 70/15/15 (GroupShuffleSplit)
        │
        ▼
  pipeline sklearn único (imputação + encoding + modelo)
        │
        ▼
  tuning (HalvingRandomSearchCV, amostra por escola, StratifiedGroupKFold)
        │
        ▼
  teste único (escolas nunca vistas) → interpretação (permutation importance, SHAP)
```

**Leitura (`src/preprocessing/carregar.py`).** Lê as cinco tabelas Gold da Fase 2 (`indicador_municipio`, `distribuicao_proficiencia`, `meta_vs_resultado`, `perfil_escola`, `evolucao_temporal`), a fatia Silver committed (`metas.parquet`, `resultados_municipio.parquet`) e a Silver de alunos (`data/silver/alunos/`, 3,87 milhões de linhas, não commitada — obtida via `scripts/baixar_dados.py`). Nenhuma transformação de negócio acontece aqui; é só I/O tipado.

**Contexto externo (`src/preprocessing/externas.py`).** Normaliza as 8 fontes externas já pré-agregadas por município em `data/external/` (estrutural: `diretorios_municipio`, `censo2022_municipio`; defasadas: `pib_municipio`, `populacao_municipio`, `ideb_municipio`, `indicadores_municipio`, `censo_escolar_municipio`, `bolsa_familia_municipio`), cada uma com sua regra própria de defasagem em relação ao ano-alvo 2024 (detalhada na seção 2).

**Contexto t−1 e LOO (`src/preprocessing/contexto.py`).** Produz o contexto municipal defasado (`contexto_municipal_defasado`, sempre ano t−1 em relação ao alvo), a meta pactuada do ano-alvo, e o contexto de escola leave-one-out (`contexto_escola_loo`, fórmula `(soma − valor_do_aluno)/(n − 1)`) usado só no regime diagnóstico. Também calcula a decomposição de variância (H9 da EDA).

**Feature store (`src/preprocessing/feature_store.py`).** Monta as duas bases de modelagem: `base_modelagem_aluno` (grão aluno, 1.851.828 × 103 colunas) e `base_modelagem_municipio` (grão município × ano, 10.276 × 84 colunas). É o único ponto que grava em `data/processed/` (git-ignored, regenerado localmente).

**Features e guarda de vazamento (`src/preprocessing/features.py`).** Define as colunas por regime (produção/diagnóstico), roda `verificar_leakage` contra `COLUNAS_PROIBIDAS` (`src/config.py`), calcula VIF/correlação para diagnosticar colinearidade, e expõe `separar_xy` (retorna `X, y, grupos=id_escola, pesos=peso_aluno`).

**Pipeline sklearn (`src/preprocessing/pipeline.py`).** Um único `ColumnTransformer` + `Pipeline` (imputação de numéricas por mediana, one-hot de categóricas, modelo final), ajustado inteiramente dentro do fit de treino — nunca em separado sobre a base inteira.

**Split (`src/modeling/split.py`).** `GroupShuffleSplit` por `id_escola`, 70/15/15, seed 42 (`dividir_por_escola`); `conferir_sem_vazamento` confirma que nenhuma escola aparece em duas partições; `StratifiedGroupKFold` e `GroupKFold` para CV por escola/município.

**Treino e tuning (`src/modeling/train.py`, `src/modeling/tune.py`).** `train.py` treina dummy/logística/HGB nos regimes produção/diagnóstico e grava métricas; `tune.py` roda `HalvingRandomSearchCV` numa amostra por escola do treino (~300 mil alunos), com early stopping nativo do HGB, e o refit final acontece na base completa de treino.

**Métricas (`src/evaluation/metrics.py`).** Métricas ponderadas e não ponderadas (por `peso_aluno`), por recorte (UF, rede), calibração em 10 faixas, e a CV por município (`cv_municipio`) quando solicitada.

**Interpretação (`src/evaluation/interpret.py`).** Permutation importance, valores SHAP (`shap` 0.46.0) e coeficientes da regressão logística como conferência.

**Predict (`src/modeling/predict.py`).** Escora uma base nova (ou só o teste, via `--so-teste`) com um pipeline já treinado e salvo, gravando `reports/predicoes_*.csv`.

**Modelo B (`src/modeling/risco_municipio.py`).** Regressão (taxa do ano seguinte) e classificação (não atingir a meta) no grão município, usando a base `base_modelagem_municipio`.

**Clusters (`src/modeling/clusters.py`).** Agrupa municípios por perfil de vulnerabilidade a partir da forma da distribuição de proficiência e contexto socioeconômico/territorial.

**Figuras (`src/visualization/plots.py`).** Estilo único (cores, eixos, legendas) reutilizado pelos 4 notebooks e por este relatório/README; produz os 35 arquivos PNG em `images/`.

## 2. Feature store

**Grãos.** `base_modelagem_aluno`: 1.851.828 linhas × 103 colunas, grão aluno-ano (universo: presentes com nota, rede pública, ano-alvo 2024). `base_modelagem_municipio`: 10.276 linhas × 84 colunas, grão município-ano (todas as transições ano→ano+1 observáveis na Gold/Silver, não só 2024).

**Regra temporal por fonte (ano de referência para o alvo 2024).**

| Fonte | Defasagem | Ano de referência |
|---|---|---|
| `diretorios_municipio` | estrutural | — |
| `censo2022_municipio` | estrutural | 2022 (Censo) |
| `pib_municipio` | 2 | 2022 |
| `populacao_municipio` | 1 | 2023 |
| `ideb_municipio` | 1 | 2023 |
| `indicadores_municipio` | 1 | 2023 |
| `censo_escolar_municipio` | 1 | 2023 |
| `bolsa_familia_municipio` | 1 | 2023 (dez/2023) |
| Contexto municipal (Gold) | 1 | 2023 |

Cada fonte é filtrada pela sua própria regra antes do join — nunca um join ingênuo por `id_municipio` sozinho, porque `data/external/` guarda linhas de vários anos por município.

**LOO da escola.** `contexto_escola_loo` calcula, para cada aluno, o agregado da própria escola no mesmo ano **excluindo o próprio aluno**: `(soma_escola − valor_do_aluno) / (n_alunos_escola − 1)`. Aplica-se a `taxa_escola_loo`, `prof_media_escola_loo` e `taxa_participacao_escola_loo`. Escolas com um único aluno avaliado (`n_alunos_escola == 1`) recebem `NaN` no LOO (não há colega para calcular a média sem o próprio aluno) — coberto por `test_contexto.py::test_loo_exclui_o_proprio_aluno_e_da_nan_para_escola_solitaria`. Essas colunas existem só no regime diagnóstico.

**Flags.** `sem_historico`: marca os 676 municípios (23,1% dos alunos) sem resultado de 2023, cujo contexto municipal t−1 fica ausente e é imputado pelo pipeline (mediana, dentro do `ColumnTransformer`). Indicadores de faltante: o pipeline usa imputação por mediana em todas as numéricas via `SimpleImputer`, e o próprio HGB tem suporte nativo a `NaN` (não precisa de indicador binário adicional para o modelo de árvore) — a logística, por rodar sobre a mesma matriz imputada, herda a mediana como valor de fallback nas mesmas colunas.

**Os dois gaps conhecidos de dados e como cada um é tratado:**

- **`pct_va_agropecuaria`, `pct_va_industria`, `pct_va_servicos`, `pct_va_adespss` (composição setorial do PIB) ficam 100% faltantes para o ano-alvo 2024**, porque a fonte do IBGE usada só publica a quebra setorial do PIB até 2021. Tratamento: as quatro colunas continuam na lista de features numéricas (visíveis em `colunas.numericas` dos quatro `reports/metricas_*.json`), mas como são 100% `NaN` no alvo, o `SimpleImputer` de mediana não tem nenhum valor observado para imputar — o `predict.py` confirma isso na prática (`UserWarning: Skipping features without any observed values: ['pct_va_agropecuaria' 'pct_va_industria' 'pct_va_servicos' 'pct_va_adespss' 'projecao_ideb_ai']`, reproduzido na seção 7; a mesma lista inclui `projecao_ideb_ai`, também 100% ausente no alvo por razão análoga, embora essa coluna não faça parte dos dois gaps documentados aqui — ver dicionário de dados). Na prática, essas colunas não contribuem sinal ao modelo de produção — permanecem na matriz por completude do esquema, não por utilidade preditiva real neste ano-alvo.
- **Bolsa Família 2023 é a única onda coberta**: a feature `pct_familias_bolsa_familia` usa dezembro de 2023 (defasagem 1 para o alvo 2024); não há uma segunda onda anterior no pipeline para checar estabilidade dentro deste repositório (a estabilidade dez/2023→dez/2024, ρ = 0,99, foi verificada na análise externa `docs/details/analise-cadunico-bolsa-familia.md`, mas essa segunda onda não é uma feature do modelo — é só evidência de que a coluna não é ruído de um mês isolado).

**Como cada gap é tratado em A (aluno) e em B (município):** as quatro colunas de PIB setorial e `pct_familias_bolsa_familia` aparecem na lista de `colunas.numericas` do Modelo A (`reports/metricas_producao_hgb.json`), mas **não** aparecem na lista `features` do Modelo B (`reports/metricas_modelo_b.json`) — decisão discutida na seção 8 (distribuição treino×aplicação).

## 3. Inventário de vazamentos e os testes que o garantem

| Vazamento | Tratamento | Teste(s) que garantem |
|---|---|---|
| `proficiencia` e gates de presença/nota (`presente`, `sem_nota`, `presenca`) definem o próprio alvo | removidos da matriz de features via `COLUNAS_PROIBIDAS` (`src/config.py`) | `tests/test_features.py::test_verificar_leakage_barra_colunas_proibidas` |
| Agregados de escola do mesmo ano incluiriam o próprio aluno | calculados leave-one-out, restritos ao regime diagnóstico | `tests/test_contexto.py::test_loo_exclui_o_proprio_aluno_e_da_nan_para_escola_solitaria`, `tests/test_contexto.py::test_participacao_loo_desconta_o_aluno` |
| Gold do mesmo ano-alvo revelaria o próprio resultado do município | contexto municipal sempre defasado em t−1 | `tests/test_contexto.py::test_contexto_municipal_usa_apenas_o_ano_anterior`, `tests/test_feature_store.py::test_base_aluno_nao_carrega_gold_do_proprio_ano` |
| `gap`/`atingiu_meta`/`situacao_meta` descrevem o resultado, não o precedem | fora de `COLUNAS_PROIBIDAS`, nunca entram como feature; meta usada é a pactuada (`meta_alvo`/`meta_prox`), não o resultado | `tests/test_feature_store.py::test_base_municipio_alvo_do_ano_seguinte`, `tests/test_contexto.py::test_meta_pactuada_do_ano_alvo` |
| `id_escola`/`id_municipio`/`id_aluno` como categoria vazariam identidade, não contexto | usados só como chave de agrupamento (split/LOO), nunca como coluna categórica do modelo | `tests/test_features.py::test_separar_xy_devolve_x_limpo_grupos_e_pesos` |
| Split aleatório por aluno vazaria contexto de escola entre treino/teste | split sempre por `id_escola` (grupo), `GroupShuffleSplit` | `tests/test_split.py::test_nenhuma_escola_em_duas_partes_e_proporcoes`, `tests/test_split.py::test_conferir_detecta_vazamento`, `tests/test_split.py::test_split_e_deterministico` |
| CV que misture escolas do mesmo grupo entre folds vazaria contexto | `StratifiedGroupKFold`/`GroupKFold` por grupo | `tests/test_split.py::test_cv_por_grupo_nao_mistura_escolas` |
| Imputação/encoding ajustados na base inteira vazariam estatísticas do teste para o treino | tudo dentro de um único `Pipeline` sklearn, ajustado só no `fit` de treino | (garantido estruturalmente por `src/preprocessing/pipeline.py`; não há teste dedicado além da suíte de `train`/`tune`, que só passa porque o `Pipeline` é reajustado a cada fold) |
| Colinearidade extrema entre features poderia mascarar causas de vazamento indiretas | cálculo de correlação/VIF disponível para inspeção | `tests/test_features.py::test_correlacao_alta_e_vif` |
| Colunas por regime poderiam vazar features "só de diagnóstico" para produção | listas de colunas explicitamente segregadas por regime | `tests/test_features.py::test_colunas_por_regime` |
| Base do Modelo B poderia carregar múltiplas linhas por município-ano ou colunas do próprio ano-alvo agregado | uma linha por par ano→ano+1, alvo é o ano seguinte | `tests/test_feature_store.py::test_base_aluno_tem_uma_linha_por_aluno_com_nota_e_flags`, `tests/test_feature_store.py::test_cli_grava_as_duas_bases` |

Lista obtida diretamente de `.venv/Scripts/python.exe -m pytest --collect-only -q` (saída completa na seção 7); os 16 testes acima cobrem os 4 arquivos citados no brief (`test_features.py`, `test_split.py`, `test_feature_store.py`, `test_contexto.py`) — nenhum nome foi inventado.

## 4. Protocolo de validação

**Split.** `GroupShuffleSplit` por `id_escola`, proporção 70/15/15 (treino/validação/teste), `random_state=SEED=42`. Nunca um split aleatório por aluno ou estratificado simples, porque alunos da mesma escola compartilham contexto e vazariam informação entre partições — os testes de `test_split.py` confirmam que nenhuma escola aparece em duas partições e que o split é determinístico.

**Tuning.** `StratifiedGroupKFold(5)` dentro do treino, agrupado por `id_escola` e estratificado pelo alvo, usado como CV do `HalvingRandomSearchCV`. A busca roda numa amostra por escola (~300 mil alunos, preservando escolas inteiras) para caber em tempo razoável; o refit final do melhor conjunto de hiperparâmetros acontece na base de treino completa (1.301.972 linhas). O HGB tem early stopping nativo (baseado em validação interna do próprio scikit-learn), o que reduz o número de árvores desnecessárias sem precisar de um terceiro conjunto de validação dedicado ao tuning.

**Hiperparâmetros finais** (`reports/melhores_params_producao_hgb.json`, também replicado em `params` de `reports/metricas_producao_hgb.json` e `reports/metricas_diagnostico_hgb.json`):

| Hiperparâmetro | Valor |
|---|---|
| `learning_rate` | 0,12345899515187 |
| `max_depth` | 6 |
| `max_leaf_nodes` | 70 |
| `min_samples_leaf` | 142 |
| `l2_regularization` | 0,00929439415564 |

**Teste tocado uma vez.** As 278.228 linhas de teste (6.350 escolas nunca vistas no treino nem na validação) só são avaliadas depois de o limiar de decisão já estar fixado na validação — não há retuning nem reescolha de limiar depois de olhar o teste.

**Generalização por município (só no HGB de produção).** `GroupKFold(5)` por `id_municipio`, treinado inteiramente dentro do conjunto de treino (nunca toca teste/validação): ROC-AUC médio **0,6605776... ≈ 0,661**, desvio-padrão **0,0046348... ≈ 0,005** (`reports/metricas_producao_hgb.json`, campo `cv_municipio`). Esse número é consistente com o ROC-AUC do teste único por escola (0,6627), o que sugere que a fronteira mais dura — município nunca visto, não só escola nunca vista — não degrada a performance de forma relevante.

**Por que essa CV por município não foi repetida para logística/diagnóstico.** O campo `cv_municipio` existe nos quatro `reports/metricas_*.json`, mas só carrega o valor calculado em `reports/metricas_producao_hgb.json` — em `reports/metricas_diagnostico_hgb.json`, `reports/metricas_producao_logistica.json` e `reports/metricas_producao_dummy.json` o campo existe, mas fica `null` (a CV não foi executada para esses três). A decisão foi deliberada, não uma omissão: essa CV é cara (5 refits completos do HGB, cada um em ~1 milhão de linhas) e a pergunta que ela responde — "o modelo entregue generaliza para um município que nunca apareceu no treino?" — já está respondida pelo HGB de produção, que é o modelo que de fato vai a produção. Rodar a mesma CV no regime diagnóstico não mudaria a decisão de entrega (diagnóstico nunca é o modelo entregue, é só para interpretação) e rodar na logística serviria apenas de segunda opinião sobre uma pergunta já respondida pelo modelo principal — o retorno marginal não justificou o custo computacional dentro do escopo desta entrega.

## 5. Resultados completos

Teste único, 278.228 alunos, 6.350 escolas nunca vistas (`n_treino` = 1.301.972, `n_validacao` = 271.628 em todos os quatro `reports/metricas_*.json`).

### 5.1 Validação (limiar escolhido aqui, recall-alvo 0,80 na classe "não alfabetizado")

| Modelo | Limiar | ROC-AUC | PR-AUC | F1 (não alf.) | Recall (não alf.) | Precisão (não alf.) | Bal. acc. | Acurácia | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Dummy | 0,5974 | 0,500 | 0,4009 | 0,5723 | 1,000 | 0,4009 | 0,500 | 0,4009 | 0,2402 |
| Logística | 0,6499 | 0,6685 | 0,5588 | 0,5970 | 0,8077 | 0,4735 | 0,6034 | 0,5629 | 0,2197 |
| HGB (produção) | 0,6646 | 0,6707 | 0,5572 | 0,5989 | 0,8013 | 0,4782 | 0,6081 | 0,5698 | 0,2196 |
| HGB (diagnóstico) | 0,6660 | 0,6905 | 0,5729 | 0,6151 | 0,8000 | 0,4996 | 0,6319 | 0,5986 | 0,2154 |

### 5.2 Teste (não ponderado)

| Modelo | ROC-AUC | PR-AUC | F1 (não alf.) | Recall (não alf.) | Precisão (não alf.) | Bal. acc. | Acurácia | Brier |
|---|---|---|---|---|---|---|---|---|
| Dummy | 0,500 | 0,4014 | 0,5728 | 1,000 | 0,4014 | 0,500 | 0,4014 | 0,2403 |
| Logística | 0,6615 | 0,5489 | 0,5949 | 0,8009 | 0,4732 | 0,6015 | 0,5622 | 0,2216 |
| HGB (produção) | 0,6627 | 0,5460 | 0,5964 | 0,8007 | 0,4752 | 0,6039 | 0,5650 | 0,2218 |
| HGB (diagnóstico) | 0,6848 | 0,5802 | 0,6092 | 0,8004 | 0,4918 | 0,6229 | 0,5879 | 0,2164 |

Matrizes de confusão do teste (`[[VN, FP], [FN, VP]]`, classe positiva = não alfabetizado):

- Dummy: `[[111670, 0], [166558, 0]]`
- Logística: `[[89440, 22230], [99589, 66969]]`
- HGB (produção): `[[89411, 22259], [98760, 67798]]`
- HGB (diagnóstico): `[[89385, 22285], [92380, 74178]]`

### 5.3 Teste ponderado por `peso_aluno`

| Modelo | ROC-AUC | PR-AUC | F1 (não alf.) | Recall (não alf.) | Precisão (não alf.) | Bal. acc. | Acurácia | Brier |
|---|---|---|---|---|---|---|---|---|
| Dummy | 0,500 | 0,4071 | 0,5787 | 1,000 | 0,4071 | 0,500 | 0,4071 | 0,2414 |
| Logística | 0,6592 | 0,5525 | 0,5988 | 0,8061 | 0,4763 | 0,5987 | 0,5602 | 0,2230 |
| HGB (produção) | 0,6605 | 0,5496 | 0,6004 | 0,8047 | 0,4788 | 0,6016 | 0,5638 | 0,2232 |
| HGB (diagnóstico) | 0,6828 | 0,5847 | 0,6131 | 0,8063 | 0,4946 | 0,6203 | 0,5857 | 0,2177 |

### 5.4 Por UF — HGB de produção (26 linhas, ordenado por ROC-AUC decrescente)

| UF | n | Prevalência não alf. | ROC-AUC | PR-AUC | F1 | Recall | Precisão | Bal. acc. | Acurácia | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| CE | 12.673 | 0,1637 | 0,7172 | 0,2874 | 0,1880 | 0,1258 | 0,3718 | 0,5421 | 0,8221 | 0,1263 |
| PI | 4.870 | 0,3848 | 0,6599 | 0,5035 | 0,5965 | 0,8084 | 0,4726 | 0,6220 | 0,5791 | 0,2216 |
| AL | 3.635 | 0,5032 | 0,6570 | 0,6142 | 0,7091 | 0,9535 | 0,5644 | 0,6041 | 0,6063 | 0,2241 |
| AP | 1.014 | 0,5099 | 0,6563 | 0,6498 | 0,6866 | 1,0000 | 0,5228 | 0,5252 | 0,5345 | 0,2338 |
| MA | 9.459 | 0,3884 | 0,6314 | 0,4992 | 0,5750 | 0,8397 | 0,4372 | 0,5766 | 0,5179 | 0,2254 |
| RJ | 18.137 | 0,4425 | 0,6284 | 0,5573 | 0,6100 | 0,9346 | 0,4527 | 0,5190 | 0,4712 | 0,2332 |
| GO | 11.172 | 0,2655 | 0,6284 | 0,3560 | 0,3803 | 0,3793 | 0,3814 | 0,5785 | 0,6719 | 0,1879 |
| PB | 4.844 | 0,4335 | 0,6196 | 0,5316 | 0,6228 | 0,9229 | 0,4699 | 0,5631 | 0,5153 | 0,2344 |
| SE | 2.755 | 0,6167 | 0,6108 | 0,6941 | 0,7669 | 0,9594 | 0,6387 | 0,5431 | 0,6403 | 0,2271 |
| PE | 12.048 | 0,3826 | 0,6104 | 0,4684 | 0,5677 | 0,8074 | 0,4378 | 0,5824 | 0,5295 | 0,2261 |
| ES | 6.567 | 0,2672 | 0,6088 | 0,3585 | 0,3531 | 0,3379 | 0,3697 | 0,5639 | 0,6691 | 0,1897 |
| RO | 3.833 | 0,3585 | 0,6055 | 0,4445 | 0,5003 | 0,6492 | 0,4069 | 0,5603 | 0,5351 | 0,2225 |
| PR | 17.482 | 0,2868 | 0,6050 | 0,3689 | 0,3485 | 0,3137 | 0,3919 | 0,5590 | 0,6635 | 0,1994 |
| SP | 58.948 | 0,4211 | 0,6002 | 0,4965 | 0,5954 | 0,8961 | 0,4457 | 0,5428 | 0,4870 | 0,2366 |
| MG | 26.435 | 0,2842 | 0,5978 | 0,3566 | 0,3526 | 0,3300 | 0,3785 | 0,5574 | 0,6556 | 0,1994 |
| MS | 5.394 | 0,4201 | 0,5941 | 0,4866 | 0,5990 | 0,8120 | 0,4745 | 0,5802 | 0,5432 | 0,2380 |
| RS | 13.407 | 0,5359 | 0,5913 | 0,6249 | 0,6929 | 0,9744 | 0,5375 | 0,5032 | 0,5370 | 0,2431 |
| MT | 7.707 | 0,4035 | 0,5870 | 0,4720 | 0,5537 | 0,7331 | 0,4449 | 0,5571 | 0,5232 | 0,2400 |
| AM | 8.001 | 0,5002 | 0,5860 | 0,5696 | 0,6679 | 0,9608 | 0,5119 | 0,5220 | 0,5222 | 0,2440 |
| PA | 13.558 | 0,5031 | 0,5840 | 0,5615 | 0,6685 | 0,9830 | 0,5065 | 0,5066 | 0,5096 | 0,2456 |
| SC | 10.016 | 0,3608 | 0,5837 | 0,4285 | 0,4958 | 0,6223 | 0,4120 | 0,5604 | 0,5432 | 0,2270 |
| BA | 14.280 | 0,6322 | 0,5593 | 0,6708 | 0,7740 | 0,9970 | 0,6325 | 0,5006 | 0,6319 | 0,2315 |
| TO | 2.830 | 0,4915 | 0,5574 | 0,5560 | 0,6465 | 0,9590 | 0,4876 | 0,4924 | 0,4845 | 0,2514 |
| AC | 2.444 | 0,4243 | 0,5432 | 0,4491 | 0,5969 | 0,9682 | 0,4315 | 0,5139 | 0,4452 | 0,2533 |
| RN | 3.413 | 0,5913 | 0,5323 | 0,5981 | 0,7411 | 0,9822 | 0,5950 | 0,5076 | 0,5942 | 0,2466 |
| DF | 3.306 | 0,4083 | 0,5000 | 0,4083 | 0,5799 | 1,0000 | 0,4083 | 0,5000 | 0,4083 | 0,2416 |

O DF (ROC-AUC 0,50) é caso especial: é UF de município único, então nenhuma feature de contexto municipal varia dentro do recorte. O CE tem o melhor ROC-AUC (0,7172) mas o pior recall (0,1258), porque sua prevalência local (16,4%) é muito abaixo da prevalência nacional (~40%) usada para calibrar o limiar único do modelo.

### 5.5 Por rede — HGB de produção

| Rede | n | Prevalência não alf. | ROC-AUC | PR-AUC | F1 | Recall | Precisão | Bal. acc. | Acurácia | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| Municipal | 241.322 | 0,4065 | 0,6681 | 0,5543 | 0,6046 | 0,8009 | 0,4855 | 0,6098 | 0,5741 | 0,2214 |
| Estadual | 36.906 | 0,3676 | 0,6142 | 0,4691 | 0,5432 | 0,7990 | 0,4114 | 0,5674 | 0,5060 | 0,2243 |

### 5.6 Calibração — HGB de produção (10 faixas de probabilidade prevista, `reports/metricas_producao_hgb.json`)

| Faixa | Prob. média | Taxa observada | n |
|---|---|---|---|
| (0,00, 0,10] | 0,0876 | 0,1765 | 34 |
| (0,10, 0,20] | 0,1833 | 0,3422 | 716 |
| (0,20, 0,30] | 0,2683 | 0,3305 | 6.212 |
| (0,30, 0,40] | 0,3602 | 0,3791 | 24.456 |
| (0,40, 0,50] | 0,4601 | 0,4810 | 47.974 |
| (0,50, 0,60] | 0,5512 | 0,5506 | 60.207 |
| (0,60, 0,70] | 0,6493 | 0,6495 | 70.328 |
| (0,70, 0,80] | 0,7458 | 0,7355 | 41.412 |
| (0,80, 0,90] | 0,8420 | 0,8075 | 19.028 |
| (0,90, 1,00] | 0,9480 | 0,9230 | 7.861 |

O modelo está bem calibrado no miolo da distribuição (faixas 0,3–0,8, onde está a maior parte da massa de dados), e ligeiramente super/sub-calibrado nos extremos de baixa contagem (n < 1.000).

### 5.7 Predições de teste

Cada regime/modelo com HGB grava `reports/predicoes_producao_hgb.csv` e (logística) `reports/predicoes_producao_logistica.csv`, 278.228 linhas cada, contendo probabilidade prevista e classe no limiar de produção — usados para auditoria externa das métricas acima sem precisar reexecutar o pipeline.

## 6. Modelo B e clusters

### 6.1 Modelo B — avaliação completa (`reports/metricas_modelo_b.json`)

Contagens: `n_treino` = 4.775 pares ano→ano+1 com meta pactuada nos dois lados, `n_com_meta` = 4.775, `n_indistinguivel_2024` = 2.040 (municípios cuja diferença entre taxa prevista e meta cai dentro da margem de incerteza — não são contados como "não atingiu" nem "atingiu" com confiança).

| Tarefa | Modelo | Métrica | Média | Desvio-padrão |
|---|---|---|---|---|
| Regressão (`taxa_prox`) | persistência (repete a taxa do ano) | MAE | 12,8639 | 0,0000 |
| Regressão | Ridge | MAE | 8,7041 | 0,2347 |
| Regressão | Ridge | RMSE | 11,6565 | 0,2646 |
| Regressão | Ridge | R² | 0,6566 | 0,0169 |
| Regressão | HGB | MAE | 8,6010 | 0,2406 |
| Regressão | HGB | RMSE | 11,6712 | 0,3055 |
| Regressão | HGB | R² | 0,6557 | 0,0181 |
| Classificação (`nao_atingiu_prox`) | Logística | ROC-AUC | 0,8428 | 0,0105 |
| Classificação | Logística | PR-AUC | 0,6304 | 0,0226 |
| Classificação | Logística | Brier | 0,1255 | 0,0054 |
| Classificação | HGB | ROC-AUC | 0,8363 | 0,0118 |
| Classificação | HGB | PR-AUC | 0,6361 | 0,0320 |
| Classificação | HGB | Brier | 0,1320 | 0,0063 |

Os dois regressores batem a persistência por larga margem (MAE cai de ~12,86 para ~8,6–8,7 pontos percentuais). Na classificação, a logística supera ligeiramente o HGB (ROC-AUC 0,8428 vs. 0,8363) — com poucas linhas por fold (4.775 no total), o modelo mais simples generaliza melhor; o modelo de classificação final entregue é o HGB, escolhido por consistência com o Modelo A, ainda que a logística tenha ROC-AUC marginalmente maior.

**Features usadas** (70 no total, `reports/metricas_modelo_b.json` campo `features`): agregados do próprio ano corrente do município (`taxa_alfabetizacao`, `ic95`, `taxa_participacao`, `proficiencia_media`, `criancas_nao_alfabetizadas`, distribuição por nível `pct_nivel_0`…`pct_nivel_8`, `pct_critico/atencao/quase_la`), a meta do ano seguinte (`meta_prox`), estrutural/territorial (`capital_uf`, `amazonia_legal`, `latitude`, `longitude`, Censo 2022), IDEB/SAEB/Censo Escolar (defasados como no Modelo A) e categóricas `sigla_uf`, `regiao`.

**Features excluídas e motivo:** `rede` não entra porque a base de município já é agregada só pela rede municipal (não há variação de rede dentro da linha); `pct_familias_bolsa_familia` e as quatro colunas `pct_va_*` ficam fora do Modelo B (motivo detalhado na seção 8, item "Bolsa Família e `pct_va_*` fora do B").

**Aplicação em produção:** o HGB de classificação é aplicado às `n_aplicacao_2024` = 5.452 linhas de 2024 para estimar o risco de não atingir a meta de 2025. `n_sem_meta_2025` = 100 municípios ficam fora dessa aplicação por não terem meta pactuada para 2025. `n_acima_da_margem` = 1.267 municípios (23,2% das 5.452 aplicações) têm probabilidade prevista de não atingir a meta além da margem de `ic95` — não só levemente abaixo por ruído estatístico. `prob_media_nao_atingir_2025` = 0,2075 (probabilidade média entre as 5.452 aplicações).

### 6.2 Escolha de k (`reports/avaliacao_k.csv`)

| k | Inércia | Silhueta |
|---|---|---|
| 2 | 66.117,04 | 0,1923 |
| 3 | 58.353,21 | **0,2026** |
| 4 | 52.791,33 | 0,1669 |
| 5 | 49.412,58 | 0,1489 |
| 6 | 45.555,05 | 0,1528 |
| 7 | 43.125,07 | 0,1426 |
| 8 | 41.345,60 | 0,1369 |

**k = 3** foi escolhido porque tem a maior silhueta (0,2026) entre todos os valores testados (2 a 8) — inércia decresce monotonicamente com k (como esperado), mas a silhueta pico em k=3 indica a melhor separação relativa de clusters nesse ponto, não em k maiores que apenas fragmentam grupos coesos.

### 6.3 Perfis dos três clusters (`reports/perfis_clusters.csv`)

| Cluster | Nome | n | Taxa média (%) | Crianças não alfabetizadas | Região dominante | % região dominante | pct_nivel_0 | pct_nivel_8 | log_pib_per_capita | log_populacao |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **Cauda crítica** | 2.041 | 43,17 | 443.830 | Nordeste | 56,74% | 3,93 | 1,26 | 9,92 | 9,68 |
| 1 | **Alto desempenho** | 413 | 91,00 | 7.598 | Nordeste | 73,61% | 0,43 | 29,74 | 9,78 | 9,25 |
| 2 | **Intermediário urbano** | 2.996 | 72,26 | 301.523 | Sudeste | 43,49% | 1,26 | 3,53 | 10,47 | 9,34 |

Notável: os dois clusters extremos ("cauda crítica" e "alto desempenho") têm ambos o Nordeste como região dominante, o que mostra que a variação dentro de uma mesma região regional pode ser maior que a variação entre regiões — reforça por que os clusters usam forma de distribuição de proficiência e contexto, e não a região geográfica diretamente, como base de agrupamento.

### 6.4 Disclosure de qualidade de dados — ~166 municípios sem meta tratados como "atingiu a meta"

**Achado, registrado sem esconder:** em `data/processed/base_modelagem_municipio.parquet`, aproximadamente **166 municípios** com `situacao_meta_prox == "sem_meta"` (ou seja, sem meta pactuada para o ano seguinte) aparecem com `nao_atingiu_prox == 0.0` em vez de `NaN`. É um defeito upstream em `src/preprocessing/contexto.py`/`feature_store.py`, introduzido numa task já mergeada antes desta branch (PR #4, "feature/feature-store") e **fora do escopo desta entrega** — não foi corrigido aqui.

**Efeito prático:** esses ~166 municípios (cerca de 3,5% do `n_treino` = 4.775 usado na classificação do Modelo B) entram no treino da classificação como "atingiu a meta" quando, na verdade, não tinham meta pactuada e deveriam ter sido excluídos dessa tarefa. O efeito estimado é pequeno — a maior parte do sinal de treino vem dos ~4.600 municípios com meta genuína — mas é real. Fica registrado aqui e no README (seção "Limitações do projeto") como limitação conhecida, não corrigida nesta entrega por estar fora do escopo do plano de tasks em vigor.

## 7. Reprodutibilidade

**Versões** (`requirements.txt`):

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

google-cloud-bigquery==3.42.2   # só extração de fontes externas
db-dtypes==1.3.0

pytest==8.3.3                    # desenvolvimento
ipykernel==6.29.5
nbconvert==7.16.4
```

Python 3.11 especificamente (o `python` padrão desta máquina é 3.14; o venv precisa ser criado a partir de um interpretador 3.11). Seed única em todo o projeto: `SEED = 42` (`src/config.py`), usada em `GroupShuffleSplit`, `StratifiedGroupKFold`, `GroupKFold`, `HalvingRandomSearchCV`, treino do HGB e da logística, e clusterização.

**Comandos, na ordem do README ("Como executar"):**

```bash
git clone https://github.com/tuanyfortunato/predicao-alfabetiza-brasil.git && cd predicao-alfabetiza-brasil
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt   # Python 3.11
cp .env.example .env
.venv/Scripts/python.exe -m scripts.baixar_dados
.venv/Scripts/python.exe -m src.preprocessing.feature_store
.venv/Scripts/python.exe -m src.modeling.train --regime producao --modelo dummy
.venv/Scripts/python.exe -m src.modeling.train --regime producao --modelo logistica
.venv/Scripts/python.exe -m src.modeling.train --regime producao --modelo hgb --params reports/melhores_params_producao_hgb.json
.venv/Scripts/python.exe -m src.modeling.train --regime diagnostico --modelo hgb --params reports/melhores_params_producao_hgb.json
.venv/Scripts/python.exe -m src.modeling.predict --regime producao --modelo hgb --so-teste
.venv/Scripts/python.exe -m src.modeling.risco_municipio
.venv/Scripts/python.exe -m src.modeling.clusters
.venv/Scripts/python.exe -m pytest
```

**Tempos medidos ou estimados nesta sessão:**

- `predict.py --regime producao --modelo hgb --so-teste`: medido nesta sessão com `time` no Git Bash, **8,16 s reais** (278.228 linhas escoradas). Confirma que a etapa de escoragem é rápida — segundos, não minutos.
- `pytest -q` (74 testes, suíte inteira): medido, **10,27 s**.
- `feature_store.py`, `train.py` (dummy/logística/HGB), `risco_municipio.py`, `clusters.py`: não foram re-executados do zero nesta sessão (envolveriam reler/reprocessar 1,85 milhão de linhas de aluno e treinar o HGB completo em 1,3 milhão de linhas); com base na ordem de grandeza de `n_treino` e na natureza do HGB por histograma, ficam na faixa de segundos a poucos minutos cada — não foi medido com precisão para não repetir uma execução multi-minuto só para cronometrar.
- `HalvingRandomSearchCV` (`tune.py`) é a etapa mais cara do pipeline por natureza (várias rodadas de halving sobre uma amostra por escola, cada rodada treinando múltiplas configurações de HGB); não foi re-executada nesta sessão.
- Notebook `02_modelagem_aluno.ipynb` (SHAP + permutation importance): segundo os relatos de tasks anteriores desta mesma linha de desenvolvimento (histórico de commits/relatórios de task), essa célula levou **aproximadamente 30 minutos** para rodar durante o desenvolvimento — é a etapa mais lenta do projeto porque SHAP em ~280 mil linhas de teste com um HGB de até 70 folhas por árvore é computacionalmente pesado mesmo com amostragem.

**O que fica fora do Git e como obter:**

- `data/silver/alunos/` (Silver de alunos, 3,87 milhões de linhas, ~125 MB): obtida via `scripts/baixar_dados.py`, que copia do lake local da Fase 2 (variável `FASE2_LAKE_PATH` no `.env`) — requer ter rodado a Fase 2 antes ou ter acesso ao lake local.
- `data/processed/` (feature store): gerada localmente por `src.preprocessing.feature_store`, a partir da Silver de alunos + Gold + externas.
- `models/` (pipelines `.joblib` treinados e partições de split): gerados por `src.modeling.train`.
- Credenciais GCP para reextrair as 8 fontes externas via `scripts/extrair_externas.py`/`scripts/baixar_bolsa_familia.py`: só necessárias para regenerar `data/external/` do zero; o repositório já traz essas 8 fontes commitadas e pré-agregadas, então clonar e rodar não exige credencial alguma, só a Silver de alunos.

## 8. Decisões e alternativas descartadas

**FUNDEB fora.** A tabela disponível usa códigos de indicador que exigiriam uma camada própria de decodificação, sem retorno claro de sinal para esta entrega dentro do prazo — fica registrado como possível evolução futura (README, "Possíveis evoluções futuras").

**INSE fora.** Só existe para 2014–2015, defasagem grande demais em relação ao ano-alvo 2024 para ser tratado como contexto socioeconômico corrente.

**CadÚnico testado e descartado, Bolsa Família entra.** Análise completa em `docs/details/analise-cadunico-bolsa-familia.md`. As duas fontes medem quase a mesma dimensão (`p_cad × p_brc`, correlação **ρ = 0,94**; `p_brc × p_bf_fam`, ρ = 0,98), mas o CadÚnico satura — mediana de 62% da população cadastrada e **83 municípios acima de 100%** (cadastros desatualizados) — o que reduz seu poder de discriminar entre municípios pobres. O Bolsa Família não satura (mediana `BRC/população` = 31%) e, medido dentro da UF (resíduo após tirar a média da UF), tem sinal que o CadÚnico não tem: correlação residual de −0,04 (CadÚnico) contra −0,13 a −0,20 (variantes do Bolsa Família). Decisão: uma única feature de pobreza (`pct_familias_bolsa_familia`), do Bolsa Família.

**`caderno` fora do modelo (H10 da EDA).** Testado só como hipótese exploratória (`eda_h10_caderno.png`); decisão: fica fora por ser artefato do instrumento de aplicação da prova (qual caderno de questões o aluno recebeu), não contexto educacional real — e já está em `COLUNAS_PROIBIDAS` por precaução.

**`peso_aluno` não entra no treino.** É um peso amostral, não uma feature preditiva — usado só via `sample_weight` no treino ponderado e nas métricas ponderadas (`teste_ponderado` em todos os `reports/metricas_*.json`), nunca como coluna de `X`. Está em `COLUNAS_PROIBIDAS`.

**Split por município como principal foi descartado; reportado como CV secundária.** Um split principal por município (em vez de por escola) seria ainda mais rigoroso — testaria a generalização para município nunca visto, não só escola nunca vista — mas reduziria drasticamente o volume de treino disponível dado que municípios concentram múltiplas escolas, e complicaria a arquitetura do split (a maioria dos alunos de um mesmo município compartilha quase todo o contexto municipal, então a fronteira relevante para vazamento continua sendo a escola). Solução adotada: split principal por `id_escola`, com `GroupKFold` por `id_municipio` como CV secundária de checagem (`cv_municipio` = 0,661 ± 0,005, seção 4) — suficiente para responder "o modelo generaliza para município novo?" sem pagar o custo de reestruturar o split principal.

**Reamostragem descartada.** O desbalanceamento de classes é moderado (~60/40, não um cenário de classe rara), então a decisão registrada na EDA foi não reamostrar e, em vez disso, reportar métricas específicas da classe minoritária (recall/precisão/F1 de "não alfabetizado") e escolher o limiar de decisão pelo recall mínimo dessa classe — reamostragem (SMOTE, undersampling) traria custo de distorcer a distribuição real sem benefício claro nesse grau de desbalanceamento.

**RandomForest descartado por custo, não por desempenho.** A escolha do HGB (HistGradientBoosting) sobre RandomForest foi guiada por dois fatores práticos do dataset: volume de treino (1,3 milhão de linhas) e presença nativa de `NaN` em várias features externas (`had_ai` falta em ~27,2% dos municípios; `ideb_ai`/`nota_saeb_lp_ai`/`nota_saeb_mat_ai` faltam em ~2,5%). O HGB, por histograma, tem suporte nativo a ausentes e treina em tempo sublinear ao volume de árvores profundas que um RandomForest exigiria para volume e dimensionalidade comparáveis, sem o custo de imputação prévia adicional que o RandomForest exigiria. Redes neurais foram descartadas por não se justificarem para um problema tabular de pouco mais de 80 colunas.

**Bolsa Família e `pct_va_*` fora do Modelo B — distribuição treino × aplicação.** O Modelo B treina em 4.775 pares ano→ano+1 e é aplicado a 5.452 linhas de 2024; ambas as colunas têm padrões de cobertura temporal que diferem entre o período de treino histórico e o ano de aplicação (`pct_va_*` fica 100% ausente já no alvo 2024, como descrito na seção 2; `pct_familias_bolsa_familia` só está disponível a partir de 2023) — incluí-las no Modelo B arriscaria treinar sobre uma distribuição de completude diferente da que o modelo veria na aplicação real. Por isso ficaram de fora da lista de `features` do Modelo B, mesmo constando na lista `colunas.numericas` do Modelo A (que treina e aplica sobre o mesmo ano-alvo 2024, sem esse descompasso).

**Predições só do teste ficam no Git.** `reports/predicoes_producao_hgb.csv` e `reports/predicoes_producao_logistica.csv` guardam só as 278.228 linhas do conjunto de teste (não a base inteira de 1,85 milhão de alunos), para manter `reports/` num tamanho committável e por ser o recorte suficiente para auditoria externa das métricas reportadas — a base completa pode ser reescorada localmente com `predict.py` sem o argumento `--so-teste`.
