# predicao-alfabetiza-brasil

Modelo supervisionado que estima se uma criança será considerada alfabetizada ao final do 2º ano do Ensino Fundamental, a partir do contexto educacional, territorial e socioeconômico em que ela estuda. Fase 3 de um Tech Challenge individual (pós-graduação em Data Analytics); consome a camada Gold da [Fase 2](https://github.com/tuanyfortunato/pipeline-dados-alfabetiza-brasil).

## Contexto do problema

O Compromisso Nacional Criança Alfabetizada usa um corte de 743 pontos no Saeb do 2º ano para definir o Indicador Criança Alfabetizada, e cada município tem metas pactuadas até 2030. A Fase 2 deste projeto mediu esse indicador a partir dos microdados oficiais; esta fase tenta antecipá-lo — o gestor público precisa saber onde o risco de não alfabetização está concentrado antes da prova acontecer, não só depois.

Um fato molda todo o desenho da solução: o aluno, na base de origem, não carrega nenhum atributo próprio — sem sexo, idade, raça ou nível socioeconômico individual. Tudo que o modelo enxerga é contexto: escola, rede, município e fontes externas agregadas por município. Isso impõe um teto de acerto por construção, não por falha de modelagem — a Fase 2 já havia medido que 75% da variância da proficiência está dentro da própria escola (intra-escola), e esta entrega trata esse teto como parte do resultado a ser comunicado, não como algo a esconder atrás de uma métrica bonita.

## Objetivo analítico

Três entregas, todas alimentadas pela mesma feature store:

- **Modelo A (obrigatório):** classifica cada aluno como alfabetizado ou não, usando apenas features conhecíveis antes da prova (regime **produção**). É o modelo entregue.
- **Modelo B:** estima o risco de um município não atingir a meta de alfabetização pactuada para 2025.
- **Clusters:** agrupa municípios em perfis de vulnerabilidade a partir da forma da distribuição de proficiência e do contexto socioeconômico/territorial.

As cinco perguntas de negócio do enunciado e onde cada uma é respondida:

| Pergunta do enunciado | Resposta vem de | Onde |
|---|---|---|
| Quais fatores mais impactam a alfabetização? | SHAP e permutation importance do Modelo A; coeficientes da logística como conferência | seção "Interpretação dos resultados"; `images/shap_beeswarm_producao.png`, `images/importancia_permutacao_producao.png` |
| Quais municípios apresentam maior risco educacional? | Modelo B (probabilidade de não atingir 2025) cruzado com volume de crianças e `ic95` | `reports/ranking_risco_municipios.csv`, `images/ranking_risco_top20.png` |
| Quais regiões possuem padrões semelhantes? | Clusters sobre forma da distribuição de proficiência + contexto | `images/clusters_perfis_niveis.png`, `images/clusters_por_regiao.png` |
| Como prever municípios que podem não atingir metas futuras? | Modelo B aplicado às features de 2024 para estimar a taxa de 2025 vs. a meta pactuada | `reports/ranking_risco_municipios.csv` (coluna `gap_previsto_2025`) |
| Quais variáveis possuem maior influência nos modelos? | Importâncias do Modelo A e features do Modelo B lado a lado | seção "Interpretação dos resultados" |

## Descrição da base utilizada

**Gold e Silver da Fase 2 usadas:**

| Tabela | Uso |
|---|---|
| `gold/indicador_municipio` | taxa de alfabetização, `ic95`, participação e proficiência média do município, ano t−1 |
| `gold/distribuicao_proficiencia` | percentual de alunos por nível de proficiência (0 a 8) e faixas críticas do município, ano t−1 |
| `gold/meta_vs_resultado` | resultado do próprio ano e alvo do ano seguinte, grão da base de modelagem por município (Modelo B) |
| `gold/perfil_escola` | taxa de participação da escola, usada no contexto leave-one-out (regime diagnóstico) |
| `gold/evolucao_temporal` | referência auxiliar de série histórica por município |
| `silver/metas.parquet` | meta de alfabetização pactuada por município (2024 e 2025) |
| `silver/resultados_municipio.parquet` | resultado por município usado para conferência cruzada com a Gold |
| `silver/alunos/` | microdados do aluno (presença, nota, rede, escola) — não commitada, 3,87 milhões de linhas |

**Oito fontes externas por município**, cada uma com sua própria regra de defasagem em relação ao ano-alvo 2024:

| Fonte | Principais colunas | Ano de referência (alvo 2024) | Publicação |
|---|---|---|---|
| `diretorios_municipio` (estrutural) | `regiao`, `capital_uf`, `amazonia_legal`, `latitude`, `longitude` | estrutural | — |
| `censo2022_municipio` (estrutural) | `pop_2022`, `domicilios_2022`, `area_km2`, `taxa_alfabetizacao_adultos`, `idade_mediana`, `pct_pop_indigena`, `pct_pop_quilombola` | 2022 | Censo 2022, IBGE |
| `pib_municipio` (defasagem 2) | `pib_per_capita`, `pct_va_agropecuaria/industria/servicos/adespss` | 2022 | dez/2024 |
| `populacao_municipio` (defasagem 1) | `populacao` | 2023 | ago/2023 |
| `ideb_municipio` (defasagem 1) | `ideb_ai`, `nota_saeb_lp_ai`, `nota_saeb_mat_ai`, `taxa_aprovacao_ideb_ai` | 2023 | ago/2024 |
| `indicadores_municipio` (defasagem 1) | `tdi_ai`, `atu_ai`, `had_ai`, `taxa_aprovacao/reprovacao/abandono_ai`, `dsu_ai` | 2023 | 2024 |
| `censo_escolar_municipio` (defasagem 1) | `pct_escolas_internet/biblioteca/esgoto_rede/agua_potavel/energia_rede`, `matriculas_ai`, `docentes_ai` | 2023 | mar/2024 |
| `bolsa_familia_municipio` (defasagem 1) | `pct_familias_bolsa_familia` — famílias com benefício em dezembro de t−1 sobre domicílios do Censo 2022 | 2023 | jan/2024 (dados.gov.br/MDS, download manual) |

Três fontes candidatas foram testadas e descartadas: **FUNDEB** fica de fora porque a tabela disponível usa códigos de indicador que exigiriam decodificação própria, sem retorno claro para esta entrega; **INSE** fica de fora porque só existe para 2014–2015, longe demais do ano-alvo; **CadÚnico** foi testado par a par com o Bolsa Família (mesma cobertura, mesmo custo) e descartado porque satura — mais da metade dos municípios tem mais de 60% da população cadastrada e 83 municípios passam de 100% (cadastros desatualizados) — enquanto o Bolsa Família não satura e ainda carrega sinal dentro da própria UF que o CadÚnico não tem.

**Universo:** alunos presentes com nota, rede pública, ano-alvo 2024 — 1.851.828 alunos, 5.517 municípios, 42.327 escolas; 59,8% classificados como alfabetizados. 676 municípios (23,1% dos alunos) não têm histórico de 2023 e entram com a flag `sem_historico`. Todas as taxas da Gold estão em pontos percentuais (0–100); note-se que várias features externas (Censo, Censo Escolar, Bolsa Família) vêm em fração 0–1 — a escala de cada coluna está documentada no dicionário.

**O que a base não permite:** os identificadores de escola são pseudônimos que não se mantêm estáveis entre anos, o que impede reconstruir uma série temporal por escola; não há nenhum atributo individual do aluno; a rede privada (24 alunos em 2024 na Silver) fica fora porque nem sequer existe na Gold/`perfil_escola`.

Dicionário completo, coluna a coluna, com origem, ano de referência, escala e regime: [`docs/dicionario_base_modelagem.md`](docs/dicionario_base_modelagem.md).

## Etapas de modelagem

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

**Regimes:** produção usa só features conhecíveis antes da prova — contexto municipal t−1, externas defasadas, meta pactuada, `rede` e `sigla_uf`. Diagnóstico soma a isso o contexto da própria escola no mesmo ano, calculado leave-one-out (excluindo o próprio aluno) — usado só para interpretação, nunca para o modelo entregue, porque um agregado do mesmo ano que inclui a turma do aluno é, em espírito, informação vazada do próprio resultado da escola.

**Inventário de vazamentos:**

| Vazamento | Tratamento | Onde |
|---|---|---|
| `proficiencia` e gates de presença/nota definem o próprio alvo | removidos da matriz de features via `COLUNAS_PROIBIDAS` | `src/config.py`, `tests/test_features.py` |
| Agregados de escola do mesmo ano incluiriam o próprio aluno | calculados leave-one-out e restritos ao regime diagnóstico | `src/preprocessing/contexto.py` (`contexto_escola_loo`) |
| Gold do mesmo ano-alvo revelaria o próprio resultado | contexto municipal sempre em t−1 | `src/preprocessing/contexto.py` (`contexto_municipal_defasado`) |
| `gap`/`situacao_meta` descrevem o resultado, não o precedem | fora de `COLUNAS_PROIBIDAS`, nunca entram como feature | `src/config.py` |
| `id_escola`/`id_municipio` como categoria vazariam identidade, não contexto | usados só como chave de agrupamento (split/LOO), nunca como coluna categórica do modelo | `src/preprocessing/features.py` (`verificar_leakage`) |
| Split aleatório por aluno vazaria contexto de escola entre treino/teste | split sempre por `id_escola` (grupo) | `src/modeling/split.py` (`dividir_por_escola`, `conferir_sem_vazamento`) |
| Imputação/encoding ajustados na base inteira vazariam estatísticas do teste | tudo dentro de um único `Pipeline` sklearn, ajustado só no treino | `src/preprocessing/pipeline.py` |

## Escolha do algoritmo

A comparação segue uma escada deliberada: **Dummy** (piso, sempre prevê a classe majoritária) → **regressão logística** (baseline interpretável, referência para saber quanto do sinal é linear) → **HistGradientBoosting** (modelo principal). O HGB foi escolhido em vez de RandomForest ou de uma rede neural porque o volume de treino (1,3 milhão de linhas) e a presença nativa de `NaN` em várias features externas (ex.: `had_ai`, `nota_saeb_mat_ai`) favorecem um boosting por histograma com suporte nativo a ausentes, sem custo de imputação prévia nem a lentidão de florestas profundas nesse volume; deep learning não se justifica para um problema tabular com pouco mais de 80 colunas. Tudo em scikit-learn puro.

Hiperparâmetros finais do HGB de produção, obtidos por `HalvingRandomSearchCV` em amostra por escola (`reports/melhores_params_producao_hgb.json`):

| Hiperparâmetro | Valor |
|---|---|
| `learning_rate` | 0,1235 |
| `max_depth` | 6 |
| `max_leaf_nodes` | 70 |
| `min_samples_leaf` | 142 |
| `l2_regularization` | 0,0093 |

O limiar de decisão não é 0,5: foi escolhido na validação para garantir recall mínimo de 0,80 na classe "não alfabetizado" — o objetivo do modelo é não deixar passar quem está em risco, então o limiar sacrifica precisão de propósito (limiar final de produção: 0,6646).

## Métricas de avaliação

Teste único, escolas nunca vistas no treino (278.228 alunos, 6.350 escolas):

| Modelo | ROC-AUC | PR-AUC | Recall (não alf.) | Precisão (não alf.) | F1 (não alf.) | Bal. accuracy | Brier |
|---|---|---|---|---|---|---|---|
| Dummy | 0,500 | 0,401 | 1,000 | 0,401 | 0,573 | 0,500 | 0,240 |
| Logística | 0,662 | 0,549 | 0,801 | 0,473 | 0,595 | 0,602 | 0,222 |
| HGB (produção) | 0,663 | 0,546 | 0,801 | 0,475 | 0,596 | 0,604 | 0,222 |
| HGB (diagnóstico) | 0,685 | 0,580 | 0,800 | 0,492 | 0,609 | 0,623 | 0,216 |

Ponderado por `peso_aluno` (mesmo teste):

| Modelo | ROC-AUC | PR-AUC | Recall (não alf.) | Precisão (não alf.) | F1 (não alf.) | Bal. accuracy | Brier |
|---|---|---|---|---|---|---|---|
| Dummy | 0,500 | 0,407 | 1,000 | 0,407 | 0,579 | 0,500 | 0,241 |
| Logística | 0,659 | 0,552 | 0,806 | 0,476 | 0,599 | 0,599 | 0,223 |
| HGB (produção) | 0,660 | 0,550 | 0,805 | 0,479 | 0,600 | 0,602 | 0,223 |
| HGB (diagnóstico) | 0,683 | 0,585 | 0,806 | 0,495 | 0,613 | 0,620 | 0,218 |

A logística chega a um ROC-AUC quase idêntico ao do HGB de produção (0,662 vs. 0,663), o que sugere que a maior parte do sinal disponível é aproximadamente linear — o boosting ganha em precisão/recall combinados (F1, Brier), mas não muda de patamar sozinho. O regime diagnóstico, com o contexto da própria escola, sobe o AUC para 0,685 — um ganho de 0,022 sobre a produção, que reforça o teto estrutural: mesmo enxergando o desempenho da turma no próprio ano, o modelo não passa de ~0,68–0,69 de ROC-AUC porque falta o atributo individual do aluno.

**Generalização por município:** GroupKFold com 5 folds por `id_municipio`, treinado só no conjunto de treino — ROC-AUC 0,661 ± 0,005, consistente com o número do teste único.

Figuras: [`images/modelos_comparacao.png`](images/modelos_comparacao.png), [`images/roc_pr_modelos.png`](images/roc_pr_modelos.png), [`images/calibracao_producao.png`](images/calibracao_producao.png), [`images/matriz_confusao_producao.png`](images/matriz_confusao_producao.png).

**Recortes por UF e por rede** ([`images/metricas_por_uf.png`](images/metricas_por_uf.png), [`images/metricas_por_rede.png`](images/metricas_por_rede.png)) escondem heterogeneidade relevante: no **DF**, o ROC-AUC do HGB de produção cai para 0,50 — o Distrito Federal é UF de município único, então não há variação de contexto municipal para o modelo explorar dentro dela. No **CE**, o recall da classe "não alfabetizado" despenca para 0,13 porque a prevalência local é de só 16% (0,164) — bem abaixo da prevalência nacional de 40% usada para calibrar o limiar único do modelo; um limiar global sub-representa risco em UFs onde a base de não alfabetizados é pequena.

## Interpretação dos resultados

Top 10 features por permutation importance ([`reports/importancia_permutacao_producao.csv`](reports/importancia_permutacao_producao.csv), queda média de ROC-AUC ao embaralhar a coluna) e por |SHAP| médio ([`reports/shap_resumo_producao.csv`](reports/shap_resumo_producao.csv)):

| Ranking | Permutation importance | |SHAP| médio |
|---|---|---|
| 1 | `meta_alvo` (0,0128) | `meta_alvo` (0,122) |
| 2 | `sigla_uf` (0,0120) | `proficiencia_media_mun_t1` (0,120) |
| 3 | `rede` (0,0085) | `taxa_limite_inferior_mun_t1` (0,111) |
| 4 | `taxa_limite_inferior_mun_t1` (0,0067) | `rede` (0,061) |
| 5 | `proficiencia_media_mun_t1` (0,0051) | `nota_saeb_mat_ai` (0,051) |
| 6 | `nota_saeb_mat_ai` (0,0031) | `ideb_ai` (0,047) |
| 7 | `ideb_ai` (0,0015) | `sigla_uf_RS` (0,044) |
| 8 | `dsu_ai` (0,0010) | `nota_saeb_lp_ai` (0,039) |
| 9 | `nota_saeb_lp_ai` (0,0009) | `sigla_uf_MG` (0,038) |
| 10 | `pct_nivel_7_mun_t1` (0,0008) | `latitude` (0,033) |

As duas técnicas concordam no núcleo do sinal: a meta pactuada (`meta_alvo`), a proficiência municipal defasada (`proficiencia_media_mun_t1`, `taxa_limite_inferior_mun_t1`) e a rede de ensino (`rede`) dominam ambos os rankings — figuras [`images/importancia_permutacao_producao.png`](images/importancia_permutacao_producao.png), [`images/shap_beeswarm_producao.png`](images/shap_beeswarm_producao.png), dependências individuais em [`images/shap_dependence_meta_alvo.png`](images/shap_dependence_meta_alvo.png), [`images/shap_dependence_proficiencia_media_mun_t1.png`](images/shap_dependence_proficiencia_media_mun_t1.png), [`images/shap_dependence_taxa_limite_inferior_mun_t1.png`](images/shap_dependence_taxa_limite_inferior_mun_t1.png), [`images/shap_dependence_rede.png`](images/shap_dependence_rede.png), [`images/shap_dependence_nota_saeb_mat_ai.png`](images/shap_dependence_nota_saeb_mat_ai.png) e o efeito regional em [`images/shap_por_regiao.png`](images/shap_por_regiao.png).

**Sinais da logística como conferência** (coeficientes, regime produção): `proficiencia_media_mun_t1` entra com coeficiente positivo (+0,551) — quanto maior a proficiência municipal do ano anterior, maior a probabilidade prevista de o aluno ser alfabetizado —, enquanto UFs como `sigla_uf_RS` (−0,695) e `sigla_uf_BA` (−0,448) puxam a probabilidade para baixo e `sigla_uf_CE` (+0,629) para cima, na mesma direção qualitativa do peso de `sigla_uf` nos dois rankings de importância acima.

**O que o diagnóstico acrescenta:** ao incluir o contexto da própria escola no mesmo ano (leave-one-out), o ROC-AUC sobe de 0,663 para 0,685 (+0,022) e `taxa_escola_loo` — a taxa de alfabetização dos colegas de turma, excluindo o próprio aluno — passa a dominar o beeswarm ([`images/shap_beeswarm_diagnostico.png`](images/shap_beeswarm_diagnostico.png)), à frente das variáveis municipais. É um sinal forte, mas é exatamente o sinal que não pode ser usado em produção: no momento da previsão (antes da prova), a taxa de alfabetização da própria turma naquele ano ainda não existe.

**Modelo B** — avaliação por validação cruzada (`reports/metricas_modelo_b.json`, 4.775 pares ano→ano+1 com meta pactuada nos dois lados):

| Tarefa | Modelo | Métrica | Média | Desvio-padrão |
|---|---|---|---|---|
| Regressão (`taxa_prox`) | persistência (repete a taxa do ano) | MAE | 12,864 | — |
| Regressão | Ridge | MAE / RMSE / R² | 8,704 / 11,656 / 0,657 | 0,235 / 0,265 / 0,017 |
| Regressão | HGB | MAE / RMSE / R² | 8,601 / 11,671 / 0,656 | 0,241 / 0,305 / 0,018 |
| Classificação (`nao_atingiu_prox`) | Logística | ROC-AUC / PR-AUC / Brier | 0,843 / 0,630 / 0,125 | 0,010 / 0,023 / 0,005 |
| Classificação | HGB | ROC-AUC / PR-AUC / Brier | 0,836 / 0,636 / 0,132 | 0,012 / 0,032 / 0,006 |

Os dois regressores batem a persistência por larga margem (MAE cai de 12,86 para ~8,6–8,7 pontos percentuais), mostrando que o contexto do próprio ano carrega informação real sobre a taxa do ano seguinte, além da simples inércia. Na classificação, a logística supera ligeiramente o HGB (ROC-AUC 0,843 vs. 0,836) — com poucas linhas de treino por fold, o modelo mais simples generaliza melhor.

O modelo final (HGB de classificação) é aplicado às 5.452 linhas de 2024 para estimar o risco de não atingir a meta de 2025 (`n_aplicacao_2024`); outros 100 municípios ficam de fora dessa aplicação por não terem meta pactuada para 2025 (`n_sem_meta_2025`), e por isso não entram no ranking de risco.

**Comparação A × B (pergunta 5):** o Modelo A pesa fatores no grão do aluno — rede de ensino, meta pactuada e a distribuição de proficiência municipal do ano anterior — porque sua unidade de decisão é a criança dentro de uma escola específica. O Modelo B, no grão do município, não tem `rede` como feature (a base de município já é só rede municipal) nem contexto de escola; em compensação, usa a meta do ano seguinte (`meta_prox`) e o resultado do próprio ano corrente diretamente, porque está prevendo a transição de um ano para o outro, não um exame que ainda vai acontecer. Em resumo: A explica risco individual a partir de contexto de vizinhança (escola/município); B explica risco coletivo a partir de trajetória (município no tempo).

## Insights encontrados

- **A variância intra-escola domina de novo.** Decompondo a variância da proficiência (rede pública, 2024, dados com nota): 75,03% está dentro da escola, 9,01% entre escolas e 15,96% entre municípios — reproduzindo quase exatamente o achado 16/9/75 da Fase 2, agora com a base de modelagem construída do zero. É a confirmação numérica de por que um modelo sem atributo de aluno tem teto baixo por construção.
- **A rede estadual supera a municipal, mas só onde as duas competem.** Em 1.018 municípios que têm as duas redes (municipal e estadual) com alunos avaliados em 2024, a rede estadual supera a municipal em 4,5 pontos percentuais de diferença mediana na taxa de alfabetização.
- **Bolsa Família tem sinal, mas ele muda de direção pelo Brasil.** A correlação de Spearman nacional entre `pct_familias_bolsa_familia` (dez/2023) e a taxa de alfabetização de 2024 é de −0,26. Olhando por UF (municípios com ao menos 30 observações), o sinal vai de −0,48 em Rondônia a +0,05 em Alagoas — passando por positivo também em Pernambuco (+0,04) —, sinal de que parte do efeito nacional é regional (Norte/Nordeste concentram tanto mais pobreza quanto taxas menores) e não intrínseco ao programa.
- **Um em cada quatro municípios avaliados está fora da margem de erro da meta de 2025.** Das 5.452 aplicações do Modelo B em 2024, 1.267 municípios (23,2%) têm probabilidade prevista de não atingir a meta de 2025 além da margem de `ic95` (`acima_da_margem`), e o risco é concentrado: os 20 municípios de maior `prioridade` (risco × volume) somam 118.475 crianças previstas como não alfabetizadas, 15,7% do total nacional de 752.951 estimado pelo ranking.
- **Os três perfis de vulnerabilidade não seguem só o mapa regional.** O cluster "cauda crítica" (2.041 municípios, taxa média de alfabetização 43,2%) e o cluster "alto desempenho" (413 municípios, taxa média 91,0%) têm ambos o Nordeste como região dominante — 56,7% e 73,6% dos seus municípios, respectivamente —, o que mostra que a variação dentro de uma mesma região pode ser tão grande quanto a variação entre regiões. O terceiro cluster, "intermediário urbano" (2.996 municípios, taxa média 72,3%), tem o Sudeste como região dominante (43,5%).

## Limitações do projeto

- **Teto intra-escola.** Sem atributo individual do aluno, 75% da variância de proficiência (medida na Fase 2 e reconfirmada aqui) está fora do alcance de qualquer feature de contexto — o ROC-AUC de ~0,66–0,69 é o resultado esperado desse desenho, não uma falha de tuning.
- **Uma única onda treinável.** O Modelo A treina, valida e testa inteiramente sobre 2024 (não há uma segunda onda de exame para validação temporal real); o Modelo B tem uma única transição observável, 2023 → 2024, para aprender a dinâmica ano a ano.
- **Pseudônimos de escola.** `id_escola` não é estável entre anos da Fase 2, o que impede reconstruir séries históricas por escola e limita o contexto de escola ao próprio ano-alvo.
- **Rede privada fora.** Só 24 alunos em 2024 na Silver, e nem sequer presente na Gold/`perfil_escola` — a conclusão do projeto vale para rede pública.
- **Regressão à média.** Municípios com poucos alunos avaliados têm taxas mais instáveis ano a ano; o `ic95` tenta capturar essa incerteza, mas municípios pequenos aparecem sistematicamente nos extremos da distribuição por motivos estatísticos, não só substantivos.
- **Municípios sem histórico.** 676 municípios (23,1% dos alunos) não tinham resultado em 2023 e entram com contexto municipal ausente (`sem_historico=1`), imputado pelo pipeline.
- **Tuning com early stopping e halving.** A busca de hiperparâmetros roda em amostra por escola com halving, o que é eficiente mas não garante ter varrido o ótimo global do espaço de busca.
- **Faltantes reais em fontes externas.** `ideb_ai`/`nota_saeb_lp_ai`/`nota_saeb_mat_ai` faltam em ~2,5% dos municípios; `had_ai` falta em ~27,2%; e as quatro colunas de composição setorial do PIB (`pct_va_agropecuaria/industria/servicos/adespss`) ficam 100% faltantes para o alvo 2024, porque o IBGE só publicou a quebra setorial do PIB até 2021 na fonte usada.
- **UF de município único.** O Distrito Federal não tem variação de contexto municipal (é um único município), e o ROC-AUC do modelo cai para 0,50 nesse recorte — o modelo não tem o que aprender ali além do que já está nas features de escola/rede.
- **Limiar global penaliza UFs de baixa prevalência.** Calibrado para 80% de recall na prevalência nacional (~40%), o limiar único faz o recall cair para 0,13 no Ceará, onde a prevalência local de não alfabetização é de só 16%.
- **Defeito conhecido, não corrigido nesta entrega: ~166 municípios sem meta tratados como "atingiram a meta".** Em `data/processed/base_modelagem_municipio.parquet`, aproximadamente 166 municípios com `situacao_meta_prox == "sem_meta"` (ou seja, sem meta pactuada para o ano seguinte) aparecem com `nao_atingiu_prox == 0.0` em vez de `NaN`. É um defeito upstream em `src/preprocessing/contexto.py`/`feature_store.py`, introduzido numa task já mergeada antes desta branch (PR #4, "feature/feature-store") e fora do escopo desta entrega. O efeito prático é que esses ~166 municípios (cerca de 3,5% do treino de classificação do Modelo B) entram no treino como "atingiu a meta" quando, na verdade, não tinham meta pactuada e deveriam ter sido excluídos da classificação. O efeito estimado é pequeno — a maior parte do sinal de treino vem dos ~4.600 municípios com meta genuína — mas é real, e fica registrado aqui como limitação conhecida, não escondida.

## Aplicação prática para políticas públicas

**Ranking de risco × volume.** [`reports/ranking_risco_municipios.csv`](reports/ranking_risco_municipios.csv) cruza a probabilidade de não atingir a meta de 2025 com o volume de crianças não alfabetizadas e o `ic95` numa coluna `prioridade`; a coluna `acima_da_margem` sinaliza os municípios cujo risco previsto está além da margem de erro, não só levemente abaixo da meta por ruído estatístico. Uso recomendado: priorizar visitas técnicas e reforço de recursos pelos municípios de maior `prioridade` — que concentram simultaneamente alto risco e muitas crianças —, e não só pela probabilidade isolada, que favoreceria municípios pequenos e de alta variância. Figura: [`images/ranking_risco_top20.png`](images/ranking_risco_top20.png).

**Clusters como carteiras de intervenção.** Os três perfis nomeados — "cauda crítica" (2.041 municípios, taxa média 43,2%), "intermediário urbano" (2.996 municípios, taxa média 72,3%) e "alto desempenho" (413 municípios, taxa média 91,0%) — funcionam como carteiras de política pública com necessidades distintas: a "cauda crítica" pede intervenção estrutural (infraestrutura escolar, formação docente); o "intermediário urbano" pede otimização incremental; o "alto desempenho" pede manutenção e uso como referência. Figuras: [`images/clusters_perfis_niveis.png`](images/clusters_perfis_niveis.png), [`images/clusters_por_regiao.png`](images/clusters_por_regiao.png).

**O limiar do Modelo A serve para triagem de contexto, nunca para decisão individual.** O modelo prevê a probabilidade de um perfil de contexto (escola, rede, município) estar associado a não alfabetização — ele nunca deve ser usado para rotular uma criança específica como "em risco" sem avaliação pedagógica direta, porque não enxerga nada sobre a própria criança. O uso legítimo é agregado: identificar escolas/municípios cujo contexto historicamente se associa a risco, para direcionar recursos ali.

**O que o número não sustenta.** Nem o ranking, nem os clusters, nem o Modelo A permitem afirmar causalidade (por exemplo, que aumentar `pct_familias_bolsa_familia` reduziria a taxa de alfabetização — a correlação nacional é negativa, mas ela é regional, como mostra o insight de Bolsa Família acima) nem prever o resultado de uma criança específica; eles apontam onde o risco de contexto está concentrado, o que é suficiente para priorizar política pública, mas não para decisões individuais nem para conclusões de causa e efeito.

## Possíveis evoluções futuras

- Incorporar a onda de 2025 assim que publicada, permitindo validação temporal real (treinar em 2023→2024, testar em 2024→2025) em vez de só validação cruzada dentro de uma única onda.
- Reavaliar o FUNDEB como fonte externa, decodificando os códigos de indicador.
- Se o INEP liberar uma chave estável de aluno entre anos, agregar atributos individuais e reavaliar o teto de 75% de variância intra-escola.
- Modelo hierárquico (aluno dentro de escola dentro de município) em vez de features de contexto "achatadas" num único vetor.
- Limiar de decisão por UF em vez de um limiar global, para não penalizar recall em UFs de baixa prevalência como o Ceará.
- Publicar `src/modeling/predict.py` como serviço de escoragem, em vez de execução manual via linha de comando.
- Publicar a feature store (`data/processed/`) na esteira AWS da própria Fase 2, para escoragem automatizada a cada nova onda do Saeb.
- Criar uma Release da Silver de alunos no repositório da Fase 2, com o parquet zipado disponível por URL, para que clonar este repositório não dependa de ter o lake local da Fase 2.

## Estrutura do repositório

```
src/
├── config.py              # caminhos, SEED, CORTE_ALFABETIZACAO, ANO_ALVO, COLUNAS_PROIBIDAS
├── preprocessing/
│   ├── carregar.py         # leitura de Gold/Silver/externas
│   ├── externas.py         # extração/normalização das 8 fontes externas
│   ├── contexto.py         # contexto municipal t−1, meta pactuada, LOO de escola
│   ├── feature_store.py    # monta base_modelagem_aluno e base_modelagem_municipio
│   ├── features.py         # colunas por regime, guarda de leakage, VIF, correlação
│   └── pipeline.py         # ColumnTransformer + Pipeline sklearn único
├── modeling/
│   ├── split.py             # split e CV por id_escola/id_municipio
│   ├── train.py             # treino ponta a ponta (dummy/logística/HGB, produção/diagnóstico)
│   ├── tune.py               # HalvingRandomSearchCV
│   ├── predict.py            # escoragem de base nova com modelo salvo
│   ├── risco_municipio.py    # Modelo B (regressão + classificação de risco de meta)
│   └── clusters.py           # clusterização de municípios por perfil de vulnerabilidade
├── evaluation/
│   ├── metrics.py           # métricas ponderadas/não ponderadas, por recorte, calibração
│   └── interpret.py         # permutation importance, SHAP, coeficientes da logística
└── visualization/
    └── plots.py              # estilo único das figuras usadas no README e nos notebooks
scripts/
├── baixar_dados.py           # copia Gold/Silver do lake local da Fase 2
├── extrair_externas.py       # 7 fontes via BigQuery/Base dos Dados
└── baixar_bolsa_familia.py   # Bolsa Família via dados.gov.br/MDS
data/
├── gold/                     # 5 tabelas da Fase 2 (commitado)
├── silver/                   # metas.parquet, resultados_municipio.parquet (commitados); alunos/ fora do Git
├── external/                 # 8 fontes agregadas por município (commitado)
└── processed/                # feature store gerada localmente (fora do Git)
models/                       # pipelines treinados (.joblib) e partições de split (fora do Git)
reports/                      # métricas, importâncias, ranking de risco, perfis de cluster (commitado)
notebooks/
├── 01_eda.ipynb               # hipóteses H1–H11, decisões de modelagem
├── 02_modelagem_aluno.ipynb   # Modelo A: produção, diagnóstico, interpretação
├── 03_risco_municipio.ipynb   # Modelo B
└── 04_clusters.ipynb          # clusters de vulnerabilidade
images/                       # 35 arquivos; 31 figuras distintas usadas neste README e nos notebooks
docs/
└── dicionario_base_modelagem.md   # dicionário completo das duas bases de modelagem
tests/                        # 74 testes, nenhum lê data/ real (fixtures em tests/conftest.py)
```

## Como executar

Requer Python 3.11 especificamente.

```bash
git clone https://github.com/tuanyfortunato/predicao-alfabetiza-brasil.git && cd predicao-alfabetiza-brasil
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt   # Python 3.11
cp .env.example .env                                    # FASE2_LAKE_PATH (lake local da Fase 2); credencial GCP e BOLSA_FAMILIA_PATH só para reextrair
.venv/Scripts/python.exe -m scripts.baixar_dados        # Silver de alunos (124 MB; Gold e as 8 externas já vêm no repo)
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

Observação: a Silver de alunos não está no Git nem em Release; é preciso o lake local da Fase 2 (`FASE2_LAKE_PATH`) ou rodar a Fase 2 antes.

## Testes

74 testes (`pytest -q`), nenhum lê `data/` real — fixtures hand-built em `tests/conftest.py`, isoladas de qualquer caminho de `src/config.py` via a fixture `lake_tmp`.

## Versionamento

Uma branch por frente de trabalho, cortada de `develop`, com PR de volta para `develop`; `develop` vai para `main` ao final da entrega.

| PR | Branch | Conteúdo |
|---|---|---|
| [#1](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/1) | `feature/setup-e-dados` | esqueleto do repositório, `config.py`, fixtures de teste |
| [#3](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/3) | `feature/enriquecimento-externo` | Gold/Silver da Fase 2 e as 8 fontes externas em `data/` |
| [#4](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/4) | `feature/feature-store` | contexto municipal/LOO, feature store, dicionário de dados |
| [#5](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/5), [#6](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/6) | `feature/pipeline-modelo` | pipeline sklearn, split por escola, treino ponta a ponta, tuning |
| [#7](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/7) | `feature/interpretacao` | métricas ponderadas/por recorte, permutation importance, SHAP |
| [#9](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/9) | `feature/figuras-e-predict` | `src/visualization/plots.py`, `predict.py` |
| [#10](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/10) | `feature/risco-e-clusters` | Modelo B e clusters de vulnerabilidade |
| [#11](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/11) | `feature/notebooks` | os 4 notebooks executados |
| desta entrega | `docs/readme` | este README |

## Autoria

Tuany Fortunato do Carmo.

## Vídeo

Link do vídeo executivo (até 5 minutos) — placeholder até a gravação.
