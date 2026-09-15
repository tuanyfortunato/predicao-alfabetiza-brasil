# predicao-alfabetiza-brasil

Projeto que prevê se uma criança será considerada alfabetizada ao fim do 2º ano do Ensino Fundamental, usando apenas o contexto em que ela estuda (escola, rede de ensino, município, perfil socioeconômico da região) — nunca informação da própria criança, porque a base de dados não contém esse tipo de atributo individual. É a Fase 3 de um Tech Challenge individual (pós-graduação em Data Analytics) e consome os dados já organizados pela [Fase 2](https://github.com/tuanyfortunato/pipeline-dados-alfabetiza-brasil) deste mesmo trabalho.

> **Utilizamos o modelo supervisionado:** um modelo que aprende observando exemplos cuja resposta certa já é conhecida. Para cada aluno do passado já se sabe se ele foi considerado alfabetizado ou não; o modelo estuda esses casos, encontra o padrão, e aplica esse padrão para prever alunos novos.

## Contexto do problema

O Compromisso Nacional Criança Alfabetizada é uma política do governo que definiu uma régua: se a criança tira 743 pontos ou mais numa prova nacional (o Saeb do 2º ano), ela é considerada "alfabetizada". Cada município do Brasil tem uma meta pactuada — um compromisso de melhorar esse número até 2030.

A Fase 2 deste projeto (o trabalho anterior) mediu esse indicador a partir dos dados oficiais: processou a prova de quem já a realizou e calculou quem ultrapassou o corte. Esta Fase 3 vai um passo além — tenta **antecipar** esse resultado. Em vez de só apurar depois quem foi alfabetizado, a proposta é prever, antes da prova, quais escolas e municípios estão em risco, dando tempo para o gestor público agir.

Um aspecto molda todo o desenho deste projeto: **o aluno, na base de dados, não carrega nenhum atributo individual** — sem sexo, idade, raça, nem nível socioeconômico próprio. Tudo que o modelo enxerga é o contexto ao redor: em que escola ele estuda, qual é a rede (municipal ou estadual), em que município, e como é esse município (mais pobre, mais rural, etc.). Isso limita o quanto o modelo consegue acertar, e não é falha de construção do modelo — é uma limitação da própria base de dados. A Fase 2 já havia medido isso: **75% da diferença de nota entre um aluno e outro acontece dentro da mesma escola** (ou seja, colegas de turma às vezes têm resultados bem diferentes um do outro, por motivos que nenhuma variável de contexto consegue captar). Esse teto é assumido como parte honesta do resultado, documentado explicitamente em vez de escondido atrás de uma métrica favorável.

## Objetivo analítico

O projeto entrega três produtos, todos alimentados pela mesma feature store — a tabela final, já pronta, com todas as colunas que os modelos usam para prever:

- **Modelo A (o obrigatório):** classifica cada aluno como "vai ser alfabetizado" ou "não vai", usando apenas informação que já existia **antes** de a prova acontecer (regime **produção** — o modelo de fato entregue, o que seria usado na prática).
- **Modelo B:** estima o risco de um município não atingir a meta de alfabetização pactuada para 2025.
- **Clusters (agrupamentos):** reúne municípios parecidos em grupos, olhando para o formato da distribuição de notas e para o contexto socioeconômico/territorial — uma forma de separar municípios em "perfis" de vulnerabilidade.

> **feature store:** tabela final, já pronta para o modelo, com uma linha por aluno (ou por município) e uma coluna para cada variável usada na previsão. Centraliza o que antes estava espalhado em várias tabelas de origem.

O enunciado do trabalho pede a resposta a cinco perguntas de negócio. A tabela abaixo mostra onde cada uma é respondida:

| Pergunta do enunciado | Resposta vem de | Onde |
|---|---|---|
| Quais fatores mais impactam a alfabetização? | As técnicas SHAP e permutation importance (explicadas mais abaixo) aplicadas no Modelo A; os coeficientes da regressão logística como conferência | seção "Interpretação dos resultados"; `images/shap_beeswarm_producao.png`, `images/importancia_permutacao_producao.png` |
| Quais municípios apresentam maior risco educacional? | O Modelo B (probabilidade de não atingir a meta de 2025) cruzado com o volume de crianças não alfabetizadas e a margem de erro (`ic95`) | `reports/ranking_risco_municipios.csv`, `images/ranking_risco_top20.png` |
| Quais regiões possuem padrões semelhantes? | Os clusters, feitos a partir do formato da distribuição de notas + contexto | `images/clusters_perfis_niveis.png`, `images/clusters_por_regiao.png` |
| Como prever municípios que podem não atingir metas futuras? | O Modelo B aplicado nos dados de 2024 para estimar a taxa de 2025 e comparar com a meta pactuada | `reports/ranking_risco_municipios.csv` (coluna `gap_previsto_2025`) |
| Quais variáveis possuem maior influência nos modelos? | As importâncias do Modelo A e as features do Modelo B, lado a lado | seção "Interpretação dos resultados" |

## Descrição da base utilizada

A Fase 2 organizou os dados em "camadas". A **Silver** é o dado já limpo, mas ainda no nível de detalhe original (um aluno por linha). A **Gold** é o dado já resumido e pronto para análise (um município por linha, com taxas e estatísticas já calculadas). Esta Fase 3 usa as duas.

**Tabelas usadas da Fase 2:**

| Tabela | Uso |
|---|---|
| `gold/indicador_municipio` | taxa de alfabetização, margem de erro (`ic95`), participação e nota média do município, sempre do ano **anterior** ao que está sendo previsto |
| `gold/distribuicao_proficiencia` | percentual de alunos do município em cada faixa de nota (de 0 a 8), do ano anterior |
| `gold/meta_vs_resultado` | o resultado do próprio ano e o alvo do ano seguinte — a tabela que origina a base de treino do Modelo B |
| `gold/perfil_escola` | quantos alunos da escola participaram da prova; usado para calcular o contexto da escola sem incorrer em vazamento de dado (detalhado na próxima seção) |
| `gold/evolucao_temporal` | histórico por município, usado como referência auxiliar |
| `silver/metas.parquet` | a meta de alfabetização pactuada por cada município (2024 e 2025) |
| `silver/resultados_municipio.parquet` | resultado por município, usado só para conferir a consistência com a Gold |
| `silver/alunos/` | o dado bruto por aluno (se fez a prova, a nota, a rede, a escola) — não é versionado no Git por ser grande (3,87 milhões de linhas) |

**Oito fontes de dados externas**, uma por município, cada uma com sua própria "regra de atraso" em relação ao ano previsto (2024). Essa regra existe porque usar um dado do mesmo ano ou de um ano futuro equivaleria a fornecer uma resposta que, na vida real, ainda não existiria no momento da previsão — um vazamento de dados (ver caixa de recordação e a seção dedicada mais abaixo):

| Fonte | Principais colunas | Ano usado (para prever 2024) | Quando foi publicado |
|---|---|---|---|
| `diretorios_municipio` (não muda com o tempo) | região, se é capital, se é Amazônia Legal, latitude/longitude | — | — |
| `censo2022_municipio` (não muda com o tempo) | população, domicílios, área, % de adultos alfabetizados, idade mediana, % população indígena/quilombola | 2022 | Censo 2022, IBGE |
| `pib_municipio` (usa o dado de 2 anos atrás) | PIB per capita, composição da economia (agropecuária/indústria/serviços) | 2022 | dez/2024 |
| `populacao_municipio` (usa o dado do ano anterior) | população | 2023 | ago/2023 |
| `ideb_municipio` (usa o dado do ano anterior) | nota do IDEB, nota do Saeb em português e matemática, taxa de aprovação | 2023 | ago/2024 |
| `indicadores_municipio` (usa o dado do ano anterior) | distorção idade-série, alunos por turma, horas-aula, taxas de aprovação/reprovação/abandono | 2023 | 2024 |
| `censo_escolar_municipio` (usa o dado do ano anterior) | % de escolas com internet/biblioteca/esgoto/água/energia, matrículas, docentes | 2023 | mar/2024 |
| `bolsa_familia_municipio` (usa o dado do ano anterior) | % de famílias que recebem Bolsa Família (em dezembro do ano anterior, dividido pelo total de domicílios do Censo 2022) | 2023 | jan/2024 (baixado à mão do dados.gov.br/MDS) |

Três fontes candidatas foram testadas e descartadas: o **FUNDEB**, porque a tabela usa códigos que exigiriam um trabalho extra de "tradução" sem garantia de valer o esforço; o **INSE** (nível socioeconômico), porque só existe para 2014–2015, distante demais do período analisado; e o **CadÚnico**, porque, comparado ao Bolsa Família, satura rapidamente — mais da metade dos municípios já tem mais de 60% da população cadastrada, e em 83 municípios o número ultrapassa 100% (provavelmente cadastro desatualizado) —, o que reduz sua capacidade de diferenciar município rico de município pobre. O Bolsa Família não tem esse problema e ainda carrega informação relevante mesmo comparando apenas dentro do mesmo estado.

**Universo do modelo:** alunos que fizeram a prova e obtiveram nota, de escola pública, no ano de 2024. São 1.851.828 alunos, em 5.517 municípios e 42.327 escolas; 59,8% deles foram classificados como alfabetizados. 676 municípios (23,1% dos alunos) não tinham resultado do ano anterior — para esses, o modelo usa uma marcação especial (`sem_historico`) e o pipeline preenche o que falta. Todas as taxas vindas da Gold estão em pontos percentuais (de 0 a 100, como em "62,8%"); atenção, porém — várias colunas externas (Censo, Censo Escolar, Bolsa Família) vêm em fração (de 0 a 1, como em "0,628"). Cada coluna tem a escala documentada no dicionário de dados.

**O que a base não permite:** o código de cada escola muda de um ano para o outro (não é um identificador fixo e estável), então não é possível acompanhar a mesma escola ao longo do tempo. Também não existe nenhuma informação individual do aluno. E a rede privada (apenas 24 alunos em 2024) fica de fora, pois nem aparece nas tabelas Gold utilizadas.

Há um dicionário completo, coluna por coluna, com origem, ano de referência, escala e onde cada uma é usada: [`docs/dicionario_base_modelagem.md`](docs/dicionario_base_modelagem.md).

## Etapas de modelagem

Caminho percorrido pelo dado, do bruto até a previsão final:

```
Silver (aluno) + Gold (indicador/distribuição, ano anterior) + 8 fontes externas (com atraso)
        │
        ▼
  feature store (tabela final pronta para o modelo, gerada localmente em data/processed/)
        │
        ▼
  separa treino/validação/teste, 70/15/15 — sempre por escola inteira, nunca por aluno
        │
        ▼
  um único "pipeline" do scikit-learn (preenche o que falta + transforma os dados + o modelo, tudo junto)
        │
        ▼
  busca dos melhores parâmetros do modelo (tuning), testando várias combinações numa amostra menor
        │
        ▼
  teste final, com escolas que o modelo nunca viu → depois, interpretação de por que ele decide o que decide
```

**Por que separar por escola, e não por aluno:** se dois alunos da mesma escola caíssem um no treino e outro no teste, o modelo estaria, na prática, "colando" — os dois compartilham o mesmo contexto de escola. Separando por escola inteira, garante-se que o teste mede de fato se o modelo generaliza para escolas novas.

> **Para recordar — pipeline (scikit-learn):** objeto que encadeia várias etapas de transformação do dado (preencher faltantes, escalar, codificar categorias) e o modelo final, tudo dentro de um único objeto treinado de uma vez. Isso garante que toda estatística usada nas transformações (como a mediana usada para preencher faltantes) é calculada só com o treino, nunca "espiando" o teste.

**Os dois "regimes" do modelo:** o regime **produção** usa só informação que já era conhecida **antes** de a prova acontecer — o contexto do município no ano anterior, as fontes externas com defasagem, a meta pactuada, a rede de ensino e o estado. É o modelo efetivamente entregue. Já o regime **diagnóstico** soma a isso o resultado da própria escola no **mesmo** ano da prova (calculado excluindo sempre o próprio aluno da conta — técnica chamada *leave-one-out*, "deixa ele de fora"). Esse regime serve só para entender melhor o problema, nunca para prever de verdade, porque no momento em que a previsão precisaria ser feita, o resultado da turma naquele ano ainda não existe.

**Vazamento de dados — o cuidado central desta seção:** ocorre quando uma informação que só existe DEPOIS do resultado acaba entrando no modelo como se fosse conhecida ANTES. Equivale a estudar para a prova já sabendo o gabarito. A tabela abaixo lista o que foi tratado, e como:

| O que poderia vazar | Como foi resolvido | Onde no código |
|---|---|---|
| A própria nota e as marcações de presença/participação definem o resultado que queremos prever | tiradas da lista de colunas permitidas (`COLUNAS_PROIBIDAS`) | `src/config.py`, `tests/test_features.py` |
| O resultado da escola no mesmo ano incluiria o próprio aluno na conta | calculado excluindo o próprio aluno (leave-one-out), e só usado no regime diagnóstico | `src/preprocessing/contexto.py` (`contexto_escola_loo`) |
| O resultado do município no mesmo ano já seria a resposta | o contexto do município usado é sempre do ano **anterior** | `src/preprocessing/contexto.py` (`contexto_municipal_defasado`) |
| Colunas que descrevem se o município bateu a meta são, na prática, o próprio resultado | ficam fora, nunca viram feature | `src/config.py` |
| Usar o código da escola ou do município como categoria vazaria a "identidade" deles, não o contexto | usados só como chave para agrupar (na hora de separar treino/teste), nunca como coluna do modelo | `src/preprocessing/features.py` (`verificar_leakage`) |
| Separar aluno por aluno (em vez de por escola) vazaria contexto da escola entre treino e teste | separação sempre pela escola inteira | `src/modeling/split.py` (`dividir_por_escola`, `conferir_sem_vazamento`) |
| Calcular médias/transformações olhando para a base toda vazaria informação do teste para o treino | tudo dentro de um único `Pipeline` do scikit-learn, que só aprende com o treino | `src/preprocessing/pipeline.py` |

## Escolha do algoritmo

A comparação foi montada como uma escada, do mais simples ao mais sofisticado:

1. **Dummy** — um "modelo" que não aprende nada, apenas prevê sempre a classe mais comum. Serve de linha de base: se o modelo real não superar isso, o esforço de modelagem não se justificou.
2. **Regressão logística** — um modelo simples e fácil de explicar, que soma pesos de cada variável de forma direta. Serve para descobrir quanto do problema é "linear" (relação simples, direta) e quanto exige algo mais complexo.
3. **HistGradientBoosting (HGB)** — o modelo efetivamente escolhido. Combina várias árvores de decisão pequenas, cada uma corrigindo o erro da anterior (técnica de *gradient boosting*), e lida nativamente com dado faltante (`NaN`), sem exigir imputação prévia.

> **Para recordar — gradient boosting / HGB:** família de modelos que soma várias árvores de decisão treinadas em sequência, cada uma corrigindo o erro deixado pela anterior — daí "gradiente" no nome, porque cada árvore segue a direção que mais reduz o erro.

O HGB foi escolhido em vez de um RandomForest ou de uma rede neural pelos seguintes motivos: o volume de dados é grande (1,3 milhão de linhas de treino), várias colunas externas têm valores realmente faltantes (como `had_ai`, `nota_saeb_mat_ai`), e o HGB lida bem com as duas coisas ao mesmo tempo, sem perda relevante de velocidade. Uma rede neural (deep learning) não se justifica aqui — o problema é uma tabela com pouco mais de 80 colunas, não imagem nem texto, domínio onde redes neurais de fato se destacam. Toda a modelagem foi feita com scikit-learn puro.

Os melhores parâmetros do HGB de produção foram encontrados testando várias combinações numa amostra menor (técnica `HalvingRandomSearchCV` — descarta progressivamente as combinações piores, para não gastar tempo testando tudo até o fim), salvos em `reports/melhores_params_producao_hgb.json`:

> **Para recordar — busca de hiperparâmetros (tuning):** hiperparâmetros são as configurações do modelo definidas antes do treino (ex.: quão fundo uma árvore pode crescer). A busca testa várias combinações e escolhe a de melhor desempenho na validação — nunca no teste, para não viciar a avaliação final.

| Parâmetro | O que ele controla | Valor escolhido |
|---|---|---|
| `learning_rate` | o "tamanho do passo" que o modelo dá a cada correção | 0,1235 |
| `max_depth` | quão "fundo" cada árvore pode crescer | 6 |
| `max_leaf_nodes` | quantas "pontas finais" (folhas) cada árvore pode ter | 70 |
| `min_samples_leaf` | quantos exemplos, no mínimo, cada folha precisa ter para valer | 142 |
| `l2_regularization` | um freio para evitar que o modelo "decore" demais o treino | 0,0093 |

O ponto de corte para decidir "alfabetizado ou não" normalmente seria 0,5 (50% de chance). Aqui não é: o corte foi ajustado (na validação, nunca no teste) para garantir que pelo menos 80% dos alunos que **realmente não** estão alfabetizados sejam identificados pelo modelo (recall) — mesmo ao custo de mais falsos alarmes. Essa escolha reflete o objetivo de não deixar passar quem precisa de atenção. O corte final ficou em 0,6646.

## Métricas de avaliação

> **Para recordar — o que cada métrica mede:**
> - **ROC-AUC**: toma um aluno alfabetizado e um não alfabetizado ao acaso e mede se o modelo consegue apontar corretamente qual é qual. 0,5 equivale a um chute aleatório; 1,0 é separação perfeita.
> - **PR-AUC**: semelhante ao ROC-AUC, mas dá mais peso à classe minoritária (aqui, "não alfabetizado") — mais informativa quando as classes são desbalanceadas.
> - **Recall (classe "não alfabetizado")**: de todos os alunos que **realmente não** estão alfabetizados, qual % o modelo identificou corretamente.
> - **Precisão (classe "não alfabetizado")**: de todos os alunos que o modelo **apontou** como não alfabetizados, qual % realmente era.
> - **F1**: uma média entre recall e precisão, para não julgar o modelo por apenas um dos dois lados.
> - **Balanced accuracy (acurácia balanceada)**: a taxa de acerto geral, ajustada para não favorecer a classe com mais exemplos.
> - **Brier score**: mede o quão bem calibrada é a probabilidade dada pelo modelo — quanto menor, melhor.

O teste foi feito uma única vez, com 278.228 alunos de 6.350 escolas que o modelo nunca tinha visto:

| Modelo | ROC-AUC | PR-AUC | Recall (não alf.) | Precisão (não alf.) | F1 (não alf.) | Bal. accuracy | Brier |
|---|---|---|---|---|---|---|---|
| Dummy | 0,500 | 0,401 | 1,000 | 0,401 | 0,573 | 0,500 | 0,240 |
| Logística | 0,662 | 0,549 | 0,801 | 0,473 | 0,595 | 0,602 | 0,222 |
| HGB (produção) | 0,663 | 0,546 | 0,801 | 0,475 | 0,596 | 0,604 | 0,222 |
| HGB (diagnóstico) | 0,685 | 0,580 | 0,800 | 0,492 | 0,609 | 0,623 | 0,216 |

Cada aluno tem um "peso" (`peso_aluno`) que representa quantas crianças reais ele representa na amostra. A tabela abaixo repete a mesma conta, mas ponderando por esse peso — uma estimativa mais próxima do comportamento na população total, não apenas na amostra:

| Modelo | ROC-AUC | PR-AUC | Recall (não alf.) | Precisão (não alf.) | F1 (não alf.) | Bal. accuracy | Brier |
|---|---|---|---|---|---|---|---|
| Dummy | 0,500 | 0,407 | 1,000 | 0,407 | 0,579 | 0,500 | 0,241 |
| Logística | 0,659 | 0,552 | 0,806 | 0,476 | 0,599 | 0,599 | 0,223 |
| HGB (produção) | 0,660 | 0,550 | 0,805 | 0,479 | 0,600 | 0,602 | 0,223 |
| HGB (diagnóstico) | 0,683 | 0,585 | 0,806 | 0,495 | 0,613 | 0,620 | 0,218 |

Nota-se que a logística (o modelo simples) chega a um ROC-AUC quase igual ao do HGB (0,662 vs. 0,663) — sinal de que boa parte do que é previsível aqui tem natureza razoavelmente linear, e o modelo mais sofisticado ganha mais em precisão/recall combinados do que em separação bruta das classes. Já o regime diagnóstico, que usa o resultado da própria escola no mesmo ano, eleva o ROC-AUC para 0,685 — um ganho de 0,022. Isso reforça o teto já discutido: mesmo enxergando o desempenho da turma, o modelo não ultrapassa ~0,68–0,69, porque falta a informação individual do aluno.

**O modelo apenas memorizou os municípios do treino?** Para verificar, foi feito um teste à parte: separação por município (não por escola) em 5 grupos, testando em municípios totalmente diferentes a cada rodada. Resultado: ROC-AUC 0,661 ± 0,005 — muito próximo do número do teste normal, o que é um bom sinal: o modelo generaliza, não memorizou.

Figuras: [`images/modelos_comparacao.png`](images/modelos_comparacao.png), [`images/roc_pr_modelos.png`](images/roc_pr_modelos.png), [`images/calibracao_producao.png`](images/calibracao_producao.png), [`images/matriz_confusao_producao.png`](images/matriz_confusao_producao.png).

**Olhando por estado e por rede** ([`images/metricas_por_uf.png`](images/metricas_por_uf.png), [`images/metricas_por_rede.png`](images/metricas_por_rede.png)), aparecem diferenças que a média nacional esconde: no **Distrito Federal**, o ROC-AUC despenca para 0,50 — o que faz sentido, já que o DF é um único município, e portanto não existe variação de contexto municipal ali dentro para o modelo aprender. No **Ceará**, o recall da classe "não alfabetizado" cai para 0,13, porque lá apenas 16% dos alunos não são alfabetizados (bem abaixo da média nacional de 40%) — como o corte de decisão foi calibrado na média nacional, ele subestima o risco em estados onde esse grupo é proporcionalmente menor.

## Interpretação dos resultados

Duas técnicas diferentes foram usadas para identificar quais variáveis mais pesam na decisão do modelo:

> **Para recordar:**
> - **Permutation importance**: embaralha os valores de uma coluna (sem alterar as demais) e mede o quanto o desempenho do modelo piora. Se a piora for grande, a coluna era de fato importante.
> - **SHAP**: calcula, aluno por aluno, o quanto cada variável empurrou a probabilidade prevista para cima ou para baixo.

Top 10 de cada uma ([`reports/importancia_permutacao_producao.csv`](reports/importancia_permutacao_producao.csv) e [`reports/shap_resumo_producao.csv`](reports/shap_resumo_producao.csv)):

| Ranking | Permutation importance | \|SHAP\| médio |
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

As duas técnicas concordam no essencial: a meta pactuada pelo município (`meta_alvo`), a nota média do município no ano anterior (`proficiencia_media_mun_t1`, `taxa_limite_inferior_mun_t1`) e a rede de ensino (`rede`) são as informações que mais pesam nos dois rankings. Figuras: [`images/importancia_permutacao_producao.png`](images/importancia_permutacao_producao.png), [`images/shap_beeswarm_producao.png`](images/shap_beeswarm_producao.png), e o efeito individual de cada uma em [`images/shap_dependence_meta_alvo.png`](images/shap_dependence_meta_alvo.png), [`images/shap_dependence_proficiencia_media_mun_t1.png`](images/shap_dependence_proficiencia_media_mun_t1.png), [`images/shap_dependence_taxa_limite_inferior_mun_t1.png`](images/shap_dependence_taxa_limite_inferior_mun_t1.png), [`images/shap_dependence_rede.png`](images/shap_dependence_rede.png), [`images/shap_dependence_nota_saeb_mat_ai.png`](images/shap_dependence_nota_saeb_mat_ai.png) e a diferença por região em [`images/shap_por_regiao.png`](images/shap_por_regiao.png).

**Conferência com a regressão logística** (mais fácil de interpretar o sinal de cada variável): quanto maior a nota média do município no ano anterior, maior a chance prevista de o aluno estar alfabetizado (coeficiente +0,551) — resultado esperado. Já residir no Rio Grande do Sul (−0,695) ou na Bahia (−0,448) reduz a probabilidade prevista, e residir no Ceará (+0,629) a aumenta — na mesma direção que o peso do estado (`sigla_uf`) já indicava nos dois rankings acima.

**O que o regime diagnóstico acrescenta:** ao incluir o resultado da própria escola no mesmo ano (sem contar o próprio aluno), o ROC-AUC sobe de 0,663 para 0,685 (+0,022), e a taxa de alfabetização dos colegas de turma (`taxa_escola_loo`) passa a dominar o gráfico de importância ([`images/shap_beeswarm_diagnostico.png`](images/shap_beeswarm_diagnostico.png)), superando até as variáveis do município. É um sinal forte — mas é justamente o sinal que não pode ser usado de fato: no momento em que a previsão precisa ser feita (antes da prova), o resultado da turma naquele ano ainda não existe.

**Modelo B** — avaliado com validação cruzada (o mesmo teste repetido várias vezes, com divisões diferentes do dado a cada rodada, para gerar mais confiança no número). Usa 4.775 pares de município ano a ano com meta pactuada (`reports/metricas_modelo_b.json`):

> **Para recordar — validação cruzada:** técnica de avaliação que divide os dados em vários blocos, treina em uns e testa nos demais, repetindo o processo com divisões diferentes — dá uma medida de desempenho mais confiável do que um único treino/teste isolado.

| Tarefa | Modelo | Métrica | Média | Desvio-padrão |
|---|---|---|---|---|
| Prever a taxa do ano seguinte | "chuta a mesma taxa do ano atual" (comparação mais simples possível) | erro médio | 12,864 pontos percentuais | — |
| Prever a taxa do ano seguinte | Ridge (regressão linear) | erro médio / erro quadrático / R² | 8,704 / 11,656 / 0,657 | 0,235 / 0,265 / 0,017 |
| Prever a taxa do ano seguinte | HGB | erro médio / erro quadrático / R² | 8,601 / 11,671 / 0,656 | 0,241 / 0,305 / 0,018 |
| Prever se vai atingir a meta | Logística | ROC-AUC / PR-AUC / Brier | 0,843 / 0,630 / 0,125 | 0,010 / 0,023 / 0,005 |
| Prever se vai atingir a meta | HGB | ROC-AUC / PR-AUC / Brier | 0,836 / 0,636 / 0,132 | 0,012 / 0,032 / 0,006 |

Os dois modelos efetivos (Ridge e HGB) erram bem menos do que simplesmente "repetir a taxa do ano anterior" (o erro cai de 12,86 para cerca de 8,6–8,7 pontos percentuais) — ou seja, o contexto do ano atual carrega informação real sobre o ano seguinte, não é apenas inércia. Na tarefa de classificação, a logística teve desempenho levemente superior ao HGB (ROC-AUC 0,843 vs. 0,836) — com poucas linhas de treino em cada rodada, o modelo mais simples generaliza melhor.

O modelo escolhido é aplicado nas 5.452 linhas de 2024 para estimar o risco de 2025 (`n_aplicacao_2024`), e todas elas aparecem no ranking final. Desses 5.452, 100 municípios não tinham meta pactuada para 2025 (`n_sem_meta_2025`) — para esses, não é possível calcular se estão "acima da margem", e essas colunas ficam vazias, mas eles ainda recebem uma probabilidade de risco e permanecem no ranking.

**Comparação entre o Modelo A e o Modelo B:** o Modelo A analisa o aluno individual, dentro de uma escola específica — por isso usa rede de ensino, meta e nota do município do ano anterior. O Modelo B analisa o município como um todo, sem informação de escola nem de rede (sua base já é restrita à rede municipal); em compensação, usa a meta do ano seguinte e o resultado do próprio ano corrente, pois está prevendo uma transição de um ano para o outro, não uma prova ainda por acontecer. Em resumo: A explica o risco individual a partir da vizinhança (escola/município); B explica o risco coletivo a partir da trajetória do município no tempo.

## Insights encontrados

- **A maior diferença acontece dentro da mesma escola, novamente.** Decompondo a origem da diferença de nota entre alunos (rede pública, 2024): 75,03% da diferença está dentro da própria escola, 9,01% entre escolas diferentes, e apenas 15,96% entre municípios diferentes. Esse resultado repete quase exatamente o achado da Fase 2 (o "16/9/75"), agora com a base construída do zero — e confirma numericamente por que um modelo sem informação do aluno tem um teto baixo por construção.
- **A rede estadual é melhor que a municipal, mas só onde as duas competem de verdade.** Em 1.018 municípios que têm as duas redes com alunos avaliados em 2024, a rede estadual bate a municipal por 4,5 pontos percentuais de diferença (na mediana).
- **O Bolsa Família tem relação com a alfabetização, mas essa relação muda de direção pelo Brasil.** No Brasil todo, quanto maior o % de famílias que recebem Bolsa Família num município, menor tende a ser a taxa de alfabetização (correlação de −0,26 — uma forma de medir se duas coisas "andam juntas", sem precisar ser uma relação perfeitamente reta). Só que olhando estado por estado, isso varia muito: de −0,48 em Rondônia até +0,05 em Alagoas (e também positivo no Pernambuco, +0,04). Isso é sinal de que parte desse efeito nacional é, na verdade, um efeito regional — Norte e Nordeste concentram tanto mais pobreza quanto taxas menores, mas não é o programa em si que "causa" essa relação.
- **Um em cada quatro municípios avaliados está fora da margem de erro da meta de 2025.** Das 5.452 aplicações do Modelo B, 1.267 municípios (23,2%) têm risco previsto de não atingir a meta de 2025 além do que a margem de erro explicaria por acaso; considerando apenas os 5.352 municípios que de fato tinham meta para 2025, essa proporção sobe para 23,7%. E esse risco é bem concentrado: os 20 municípios de maior prioridade (risco vezes volume de crianças) somam 118.475 crianças previstas como não alfabetizadas — 15,7% do total nacional estimado (752.951).
- **Os três grupos de municípios não seguem só o mapa das regiões do Brasil.** O grupo "cauda crítica" (2.041 municípios, taxa média 43,2%) e o grupo "alto desempenho" (413 municípios, taxa média 91,0%) têm os dois o Nordeste como região mais comum (56,7% e 73,6% dos municípios, respectivamente) — mostrando que dentro de uma mesma região a variação pode ser tão grande quanto entre regiões diferentes. O terceiro grupo, "intermediário urbano" (2.996 municípios, taxa média 72,3%), tem o Sudeste como região mais comum (43,5%).

## Limitações do projeto

- **O teto de acerto vem da própria base, não do modelo.** Sem nenhuma informação individual do aluno, 75% da diferença de nota está fora do alcance de qualquer informação de contexto — um ROC-AUC de ~0,66–0,69 é o esperado desse desenho, não uma falha de ajuste.
- **Só uma "onda" de dados para treinar.** O Modelo A treina, valida e testa tudo em cima de 2024 (ainda não existe uma segunda prova para validar de verdade "ao longo do tempo"); o Modelo B tem só uma transição observada (2023 → 2024) para aprender como a dinâmica funciona ano a ano.
- **O código da escola muda de um ano para o outro.** Isso impede reconstruir o histórico de uma mesma escola e limita o contexto de escola apenas ao ano analisado.
- **Rede privada ficou de fora.** Só 24 alunos em 2024, e nem aparece nas tabelas usadas — a conclusão deste projeto vale para a rede pública.
- **Municípios pequenos têm números mais instáveis de um ano para o outro** (fenômeno chamado de "regressão à média" — quando um número muito alto ou muito baixo tende a se aproximar do normal no ano seguinte, só por acaso estatístico). A margem de erro (`ic95`) tenta captar essa incerteza, mas municípios pequenos ainda aparecem em excesso nos extremos por motivo estatístico, não só por serem de fato os melhores/piores.
- **676 municípios (23,1% dos alunos) não tinham resultado do ano anterior**, e entram com o contexto municipal marcado como ausente, preenchido pelo pipeline.
- **A busca de parâmetros do modelo foi feita numa amostra, de forma mais rápida (halving)** — eficiente, mas sem garantia de ter testado literalmente toda combinação possível.
- **Faltam dados reais em algumas fontes externas.** As notas do IDEB/Saeb faltam em ~2,5% dos municípios; a distorção idade-série (`had_ai`) falta em ~27,2%; e as quatro colunas de composição da economia do PIB ficam 100% vazias para o ano de 2024, porque o IBGE só publicou essa quebra até 2021 na fonte usada.
- **O Distrito Federal é um caso especial.** Por ser um único município, não tem variação de contexto municipal, e o ROC-AUC ali cai para 0,50 — o modelo simplesmente não tem o que aprender além do que já vem de escola/rede.
- **O corte de decisão único prejudica estados com poucos casos de risco.** Calibrado para capturar 80% dos casos na média nacional (~40% de prevalência), esse mesmo corte faz o recall cair para 0,13 no Ceará, onde só 16% dos alunos não são alfabetizados.
- **A cobertura de uma coluna (`had_ai`) cai de 2023 para 2024, e isso não foi corrigido nesta entrega.** Ela está 100% preenchida nas linhas de treino (2023) do Modelo B, mas só 73,4% preenchida nas linhas de aplicação (2024) — ou seja, 26,6% das linhas de 2024 recebem um valor "inventado" (a mediana) para uma coluna que, no treino, o modelo sempre via preenchida de verdade. É o mesmo tipo de problema que já levou a excluir outras duas colunas do Modelo B (explicado acima), só que de menor impacto.
- **Defeito conhecido, não corrigido nesta entrega: ~166 municípios sem meta aparecem como se tivessem atingido a meta.** Na base processada, esses municípios (que na verdade não tinham meta pactuada para o ano seguinte) foram marcados como "atingiu a meta" em vez de ficarem de fora do cálculo. É um erro que já existia numa etapa anterior do projeto (antes desta entrega) e ficou fora do escopo consertar agora. O efeito é pequeno — cerca de 3,5% do treino de classificação do Modelo B — mas é real, e está sendo registrado aqui abertamente em vez de escondido.

## Aplicação prática para políticas públicas

**Ranking de risco cruzado com volume.** O arquivo [`reports/ranking_risco_municipios.csv`](reports/ranking_risco_municipios.csv) combina a probabilidade de não atingir a meta de 2025 com o número de crianças não alfabetizadas e a margem de erro, numa coluna chamada `prioridade`. A coluna `acima_da_margem` marca os municípios cujo risco é maior do que a própria margem de erro explicaria por acaso. Uso recomendado: priorizar visita técnica e reforço de recursos pelos municípios de maior `prioridade` — que combinam alto risco **e** muitas crianças —, em vez de usar apenas a probabilidade isolada (que favoreceria municípios pequenos, onde os números variam mais). Figura: [`images/ranking_risco_top20.png`](images/ranking_risco_top20.png).

**Os grupos de municípios como "carteiras" de política diferentes.** Os três perfis — "cauda crítica" (2.041 municípios, taxa média 43,2%), "intermediário urbano" (2.996 municípios, taxa média 72,3%) e "alto desempenho" (413 municípios, taxa média 91,0%) — pedem ações diferentes: a "cauda crítica" precisa de intervenção mais estrutural (infraestrutura escolar, formação de professores); o "intermediário urbano" precisa de ajustes incrementais; o "alto desempenho" precisa de manutenção e pode servir de referência para os demais. Figuras: [`images/clusters_perfis_niveis.png`](images/clusters_perfis_niveis.png), [`images/clusters_por_regiao.png`](images/clusters_por_regiao.png).

**O corte de decisão do Modelo A serve para triagem de contexto, nunca para decisão sobre uma criança específica.** O modelo prevê a probabilidade de um **contexto** (escola, rede, município) estar associado a não alfabetização — ele não enxerga nada sobre a criança em si, e por isso nunca deveria ser usado isoladamente para rotular uma criança como "em risco" sem avaliação pedagógica adequada. O uso correto é olhar para o agregado: identificar escolas/municípios cujo contexto historicamente aparece ligado a risco, para direcionar recursos ali.

**O que o número não sustenta.** Nem o ranking, nem os grupos, nem o Modelo A permitem afirmar que uma coisa "causa" a outra — por exemplo, não é possível afirmar que aumentar o Bolsa Família reduziria a alfabetização, mesmo com a correlação nacional sendo negativa, porque esse efeito é regional (como já mostrado no insight sobre Bolsa Família). Também não é possível prever o resultado de uma criança específica. O que é possível fazer é apontar onde o risco de contexto está concentrado — suficiente para ajudar a priorizar política pública, mas não para decisões individuais nem para conclusões de causa e efeito.

## Possíveis evoluções futuras

- Incluir a prova de 2025 assim que sair, para validar de verdade "ao longo do tempo" (treinar com 2023→2024 e testar de verdade com 2024→2025), em vez de só validação cruzada dentro de uma única onda.
- Reavaliar o FUNDEB como fonte externa, fazendo o trabalho de decodificar os códigos de indicador.
- Se o INEP liberar uma forma de identificar o mesmo aluno entre anos diferentes, será possível agregar atributos individuais e reavaliar esse teto de 75% de variação dentro da escola.
- Um modelo "hierárquico" (que entende aluno-dentro-de-escola-dentro-de-município como uma estrutura, em vez de jogar tudo achatado num vetor só).
- Um corte de decisão diferente por estado, em vez de um único corte nacional, para não prejudicar o recall em estados com pouca prevalência (como o Ceará).
- Transformar `src/modeling/predict.py` num serviço automático de escoragem, em vez de rodar manualmente pela linha de comando.
- Publicar a feature store (`data/processed/`) na esteira da AWS que já existe na Fase 2, para escorar automaticamente a cada nova prova.
- Criar uma forma de baixar a Silver de alunos direto por link (Release no GitHub da Fase 2), para quem clonar este repositório não precisar ter o "lake" local da Fase 2 configurado.

## Estrutura do repositório

```
src/
├── config.py              # caminhos, SEED (a "semente" que garante resultado repetível), corte de 743 pontos, ano-alvo, colunas proibidas
├── preprocessing/          # tudo que prepara o dado antes do modelo
│   ├── carregar.py         # lê Gold/Silver/externas
│   ├── externas.py         # busca e organiza as 8 fontes externas
│   ├── contexto.py         # contexto do município (ano anterior), meta pactuada, contexto da escola (leave-one-out)
│   ├── feature_store.py    # monta as duas tabelas finais: base do aluno e base do município
│   ├── features.py         # decide quais colunas entram por regime, guarda contra vazamento, VIF/correlação
│   └── pipeline.py         # monta o Pipeline único do scikit-learn
├── modeling/                # treino e uso dos modelos
│   ├── split.py             # separa treino/validação/teste por escola/município
│   ├── train.py             # treina do início ao fim (dummy/logística/HGB)
│   ├── tune.py               # busca os melhores parâmetros
│   ├── predict.py            # usa um modelo já treinado para prever em dado novo
│   ├── risco_municipio.py    # Modelo B
│   └── clusters.py           # os grupos de municípios
├── evaluation/               # como medir se o modelo é bom
│   ├── metrics.py           # as métricas (ponderadas, por recorte, calibração)
│   └── interpret.py         # permutation importance, SHAP, coeficientes da logística
└── visualization/
    └── plots.py              # um estilo único de gráfico, usado no README e nos notebooks
scripts/                     # scripts que só rodam uma vez para buscar/preparar dado
├── baixar_dados.py           # copia Gold/Silver do lake local da Fase 2
├── extrair_externas.py       # busca 7 das fontes externas via BigQuery
└── baixar_bolsa_familia.py   # busca o Bolsa Família
data/
├── gold/                     # as 5 tabelas da Fase 2 (versionado no Git)
├── silver/                   # metas e resultados (versionado no Git); a base de alunos fica fora por ser grande
├── external/                 # as 8 fontes externas já prontas (versionado no Git)
└── processed/                # a feature store final, gerada na sua máquina (fora do Git)
models/                       # os modelos já treinados (.joblib) — fora do Git, você recria rodando o código
reports/                      # métricas, importâncias, ranking de risco, perfis dos grupos (versionado no Git)
notebooks/                    # os cadernos Jupyter, já executados, com gráficos e explicação de cada decisão
├── 01_eda.ipynb               # a análise exploratória: testa hipóteses e decide o que vira feature
├── 02_modelagem_aluno.ipynb   # o Modelo A: comparação, interpretação
├── 03_risco_municipio.ipynb   # o Modelo B
└── 04_clusters.ipynb          # os grupos de municípios
images/                       # os gráficos usados aqui e nos notebooks (35 arquivos)
docs/
└── dicionario_base_modelagem.md   # dicionário completo das duas tabelas finais
tests/                        # 75 testes automáticos, nenhum lê dado real (usa dado inventado só para testar)
```

## Como executar

Precisa especificamente do Python 3.11 (não funciona com versões mais novas, por causa de alguma biblioteca usada).

```bash
git clone https://github.com/tuanyfortunato/predicao-alfabetiza-brasil.git && cd predicao-alfabetiza-brasil
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt   # Python 3.11
cp .env.example .env                                    # FASE2_LAKE_PATH (lake local da Fase 2); credencial GCP e BOLSA_FAMILIA_PATH só para reextrair os dados
.venv/Scripts/python.exe -m scripts.baixar_dados        # baixa a base de alunos (124 MB; a Gold e as 8 fontes externas já vêm no repositório)
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

Uma observação importante: a base de alunos não está no Git nem disponível para download direto (é grande demais). Para rodar o primeiro comando de fato (`baixar_dados`), é preciso ter acesso ao "lake" local da própria Fase 2, ou rodar a Fase 2 primeiro.

## Testes

São 75 testes automáticos (rode com `pytest -q`). Nenhum deles lê dado real de `data/` — todos usam "fixtures", dado construído à mão apenas para teste, sem depender de nada externo (estão em `tests/conftest.py`, com uma fixture chamada `lake_tmp` que garante isso).

> **Para recordar — fixture:** um dado ou objeto preparado especificamente para um teste, com valores conhecidos e controlados — diferente de dado real, que pode mudar e tornaria o teste instável.

## Versionamento

O trabalho foi dividido em frentes: cada uma ganhou sua própria branch (uma "cópia paralela" do código), cortada a partir da branch `develop`. Cada branch virou um Pull Request (PR — um pedido para integrar aquele pedaço de código de volta à branch principal, após revisão). Ao final, `develop` foi integrada à `main`, a versão "oficial" do projeto.

| PR | Branch | Conteúdo |
|---|---|---|
| [#1](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/1) | `feature/setup-e-dados` | esqueleto do repositório, `config.py`, fixtures de teste |
| [#3](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/3) | `feature/enriquecimento-externo` | Gold/Silver da Fase 2 e as 8 fontes externas em `data/` |
| [#4](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/4) | `feature/feature-store` | contexto municipal/leave-one-out, feature store, dicionário de dados |
| [#5](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/5), [#6](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/6) | `feature/pipeline-modelo` | pipeline sklearn, separação por escola, treino do início ao fim, tuning |
| [#7](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/7) | `feature/interpretacao` | métricas ponderadas/por recorte, permutation importance, SHAP |
| [#9](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/9) | `feature/figuras-e-predict` | `src/visualization/plots.py`, `predict.py` |
| [#10](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/10) | `feature/risco-e-clusters` | Modelo B e os grupos de municípios |
| [#11](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/11) | `feature/notebooks` | os 4 notebooks executados |
| [#12](https://github.com/tuanyfortunato/predicao-alfabetiza-brasil/pull/12) | `docs/readme` | README, relatório técnico e ajustes finais |

## Autoria

Tuany Fortunato do Carmo.

## Vídeo

Link do vídeo executivo (até 5 minutos) — ainda vai ser gravado.
