# Dicionário — bases de modelagem

Gerado a partir de `src/preprocessing/feature_store.py` (Task 7), rodado sobre os dados reais
(`data/gold`, `data/silver`, `data/external`). As duas bases não são commitadas
(`data/processed/`, git-ignored) — este dicionário é a referência pública das colunas.

## Universo e números-chave

- **Universo**: alunos presentes com nota (`presente == True and sem_nota == False`), rede
  pública (`rede_nome ∈ {municipal, estadual}` — a rede privada tem 24 alunos em 2024 na Silver,
  mas não existe na Gold/`perfil_escola` e é filtrada fora), ano alvo **2024**.
- **Corte de alfabetização**: proficiência ≥ **743** (`config.CORTE_ALFABETIZACAO`).
- **`base_modelagem_aluno`**: 1.851.828 linhas × 103 colunas (rodada real, `--ano 2024`).
  `sem_historico` (município sem linha em 2023 no `indicador_municipio`) cobre 676 municípios,
  23,1% dos alunos.
- **`base_modelagem_municipio`**: 10.276 linhas × 84 colunas — 5.452 em 2024 + 4.824 em 2023,
  grão `(ano, id_municipio)`, rede municipal.
- **Anos de referência das externas para o alvo 2024** (`anos_referencia(2024)`):
  `pib_municipio`→2022, `populacao_municipio`→2023, `ideb_municipio`→2023,
  `indicadores_municipio`→2023, `censo_escolar_municipio`→2023, `bolsa_familia_municipio`→2023.
- **Faltantes por fonte externa** (medido sobre os 5.571 municípios em `montar_contexto_externo(2024)`,
  Task 5): `pct_va_agropecuaria`/`pct_va_industria`/`pct_va_servicos`/`pct_va_adespss` = **100%**;
  `projecao_ideb_ai` = **100%**; `had_ai` ≈ 27,2%; `ideb_ai`/`nota_saeb_lp_ai`/`nota_saeb_mat_ai` ≈ 2,5%;
  `capital_uf` = 1 município em 5.571 (< 0,02%, negligenciável). Demais colunas estruturais e de
  Censo/Censo Escolar/Bolsa Família ≈ 0% de faltante.
- **Datas de publicação das fontes**: IDEB 2023 — ago/2024; Censo Escolar 2023 — mar/2024;
  PIB dos municípios 2022 — dez/2024; estimativa de população 2023 — ago/2023; indicadores
  educacionais (INEP) 2023 — 2024; Bolsa Família dez/2023 — jan/2024, dados.gov.br/MDS, download
  manual em 13/09/2026.

### Duas observações sobre gaps reais nos dados (não são bugs desta task)

1. **`pct_va_*` 100% faltante para o alvo 2024**: `data/external/pib_municipio.parquet` só tem a
   quebra setorial do PIB populada para 2019–2021; 2022 e 2023 têm `pib` total mas share setorial
   nulo (o IBGE publica o PIB total antes da quebra por setor). Como a regra de defasagem do PIB
   escolhe 2022 para o alvo 2024, essas quatro colunas saem 100% NaN — real, não corrigido aqui.
2. **`capital_uf` tem 1 NaN genuíno** em 5.571 municípios (`diretorios_municipio`) — gap pré-existente
   e negligenciável; na `base_modelagem_aluno` não aparece porque esse município não tem alunos
   avaliados em 2024 (5.517 municípios distintos na base).

### Nota de desenho: externas do `base_modelagem_municipio` são referenciadas ao próprio `ano` da linha

Cada linha `(ano, id_municipio)` carrega o resultado Gold do próprio `ano` **e** o contexto externo
do próprio `ano` (`montar_contexto_externo(ano)`, mesma regra de defasagem por fonte de sempre —
t−1 ou t−2 relativo a `ano`, nunca ao ano seguinte). Só `meta_prox` e os alvos (`taxa_prox`,
`situacao_meta_prox`, `nao_atingiu_prox`) olham para `ano + 1`, porque são literalmente o que a
linha tenta prever. Isso mantém a garantia "contexto sempre t−1 relativo ao que descreve" para
todas as seis fontes com defasagem — uma versão anterior desta task tinha trocado
`montar_contexto_externo(ano)` por `montar_contexto_externo(ano + 1)` para contornar uma fonte com
histórico curto (ver observação sobre `pct_familias_bolsa_familia` abaixo), mas isso adiantava em
um ano a referência de `indicadores_municipio`, `censo_escolar_municipio` e `populacao_municipio`
(passavam a usar dados do próprio `ano`, vazamento) e do `pib_municipio` (passava a usar t−1 em vez
de t−2). Foi revertido: `_municipio_no_ano` volta a chamar `montar_contexto_externo(ano)` puro, e o
gap real (Bolsa Família) foi corrigido na fonte (`externas.py`), não contornado aqui.

**`pct_familias_bolsa_familia` sai 100% NaN nas linhas de 2023**: a fonte `bolsa_familia_municipio`
só tem 2023/2024 commitados (foi adicionada depois das demais fontes, adendo 11); a linha de 2023
pediria uma referência ≤ 2022, que ainda não existe. `montar_contexto_externo` degrada essa coluna
para NaN nesse caso específico (só essa fonte, só quando faltar histórico) em vez de levantar
exceção — as demais 54 colunas externas da linha de 2023 seguem populadas normalmente. Deixa de ser
um gap assim que mais anos de Bolsa Família forem extraídos.

---

## `base_modelagem_aluno`

### Identificadores e alvo (regime *fora*)

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `id_municipio` | Silver | 2024 | inteiro (código IBGE) | fora | identificador — `COLUNAS_PROIBIDAS`, vazaria contexto direto do alvo |
| `id_escola` | Silver | 2024 | inteiro | fora | identificador — `COLUNAS_PROIBIDAS`, chave usada só para LOO/split |
| `id_aluno` | Silver | 2024 | inteiro | fora | identificador — `COLUNAS_PROIBIDAS` |
| `ano` | Silver | 2024 | inteiro | fora | constante (é o `ano_alvo`); sem variância como feature |
| `caderno` | Silver | 2024 | categórico | fora | `COLUNAS_PROIBIDAS` — versão do caderno da prova |
| `serie` | Silver | 2024 | inteiro | fora | `COLUNAS_PROIBIDAS` — série (constante, 2º ano) |
| `presenca` | Silver | 2024 | binário (0/1) | fora | `COLUNAS_PROIBIDAS` — gate de presença na prova |
| `presente` | Silver | 2024 | bool | fora | gate de presença — define o universo, não é feature |
| `presenca_nome` | Silver | 2024 | categórico | fora | `COLUNAS_PROIBIDAS` — redundante com `presenca` |
| `sem_nota` | Silver | 2024 | bool | fora | `COLUNAS_PROIBIDAS` — gate de elegibilidade da nota |
| `preenchimento_caderno` | Silver | 2024 | binário (0/1) | fora | `COLUNAS_PROIBIDAS` — condição de validade da nota |
| `peso_aluno` | Silver | 2024 | float (peso amostral) | fora | `COLUNAS_PROIBIDAS` — nunca é feature, só `sample_weight` |
| `alfabetizado` | Silver | 2024 | binário (0/1) | fora | **é o alvo** — proficiência ≥ 743 |
| `proficiencia` | Silver | 2024 | escala do teste (mesma do corte 743; ~590–900 observado) | fora | define o alvo — vazamento direto |
| `_row_hash` | Silver (Fase 2) | 2024 | string/uint | fora | metadado técnico de pipeline, não é domínio |
| `_ingestion_ts` | Silver (Fase 2) | 2024 | timestamp | fora | metadado técnico de pipeline |
| `_source` | Silver (Fase 2) | 2024 | categórico | fora | metadado técnico de pipeline |

### Silver do aluno — contexto mantido como feature (regime produção)

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `rede` | Silver | 2024 | código numérico (2=estadual, 3=municipal, observado) | produção | contexto de rede |
| `rede_nome` | Silver | 2024 | categórico (`municipal`/`estadual`) | produção | mesma informação que `rede`, legível |
| `sigla_uf` | Silver | 2024 | categórico (UF) | produção | contexto territorial |

### Contexto municipal t−1 (`contexto_municipal_defasado`, regime produção)

Todas vêm do `indicador_municipio`/`distribuicao_proficiencia` (Gold) do **ano 2023** — nunca do
ano-alvo, por desenho (`contexto.py`, Task 6). Sufixo `_mun_t1` em todas.

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `taxa_alfabetizacao_mun_t1` | Gold `indicador_municipio` | 2023 | pontos percentuais (0–100) | produção | taxa do município no ano anterior |
| `ic95_mun_t1` | Gold `indicador_municipio` | 2023 | pontos percentuais (0–100) | produção | margem do IC95 da taxa |
| `taxa_participacao_mun_t1` | Gold `indicador_municipio` | 2023 | pontos percentuais (0–100) | produção | participação na prova |
| `proficiencia_media_mun_t1` | Gold `indicador_municipio` | 2023 | escala do teste (mesma do corte 743) | produção | **não é 0–100** |
| `alunos_avaliados_mun_t1` | Gold `indicador_municipio` | 2023 | contagem absoluta | produção | volume, não taxa |
| `criancas_nao_alfabetizadas_mun_t1` | Gold `indicador_municipio` | 2023 | contagem absoluta | produção | volume, não taxa |
| `taxa_limite_inferior_mun_t1` | Gold `indicador_municipio` | 2023 | pontos percentuais (0–100) | produção | limite inferior do IC95 |
| `taxa_limite_superior_mun_t1` | Gold `indicador_municipio` | 2023 | pontos percentuais (0–100) | produção | limite superior do IC95 |
| `alerta_participacao_mun_t1` | Gold `indicador_municipio` | 2023 | binário (0.0/1.0) | produção | flag de participação baixa |
| `pct_nivel_0_mun_t1` … `pct_nivel_8_mun_t1` | Gold `distribuicao_proficiencia` (rede total) | 2023 | pontos percentuais (0–100) | produção | distribuição por nível de proficiência |
| `pct_critico_mun_t1` | Gold `distribuicao_proficiencia` (rede total) | 2023 | pontos percentuais (0–100) | produção | |
| `pct_atencao_mun_t1` | Gold `distribuicao_proficiencia` (rede total) | 2023 | pontos percentuais (0–100) | produção | |
| `pct_quase_la_mun_t1` | Gold `distribuicao_proficiencia` (rede total) | 2023 | pontos percentuais (0–100) | produção | |
| `sem_historico` | derivada (`taxa_alfabetizacao_mun_t1` nula) | — | bool | produção | município sem linha em 2023 (676 municípios, 23,1% dos alunos) |

### Meta pactuada (`meta_pactuada`, regime produção)

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `meta_alvo` | Silver `metas` (`meta_alfabetizacao_2024`) | 2024 (pactuada antes do resultado) | pontos percentuais (0–100) | produção | meta municipal para o ano-alvo |

### Externas por fonte (`montar_contexto_externo`, Task 5)

**Estruturais — `diretorios_municipio`** (sem ano; regime produção, exceto `nome_municipio`):

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `nome_municipio` | `diretorios_municipio` | estrutural | texto | fora | identificador textual (1:1 com `id_municipio`), não é feature |
| `regiao` | `diretorios_municipio` | estrutural | categórico | produção | |
| `capital_uf` | `diretorios_municipio` | estrutural | binário (0/1) | produção | 1 NaN em 5.571 municípios (negligenciável) |
| `amazonia_legal` | `diretorios_municipio` | estrutural | binário (0/1) | produção | |
| `latitude` | `diretorios_municipio` | estrutural | graus decimais | produção | |
| `longitude` | `diretorios_municipio` | estrutural | graus decimais | produção | |

**Estruturais — Censo 2022 (`censo2022_municipio`)**:

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `pop_2022` | Censo 2022 | estrutural (2022) | contagem absoluta | produção | |
| `domicilios_2022` | Censo 2022 | estrutural (2022) | contagem absoluta | produção | usado também como base de `pct_familias_bolsa_familia` |
| `area_km2` | Censo 2022 | estrutural (2022) | km² | produção | |
| `taxa_alfabetizacao_adultos` | Censo 2022 | estrutural (2022) | **fração 0–1** | produção | não confundir com `taxa_alfabetizacao_mun_t1` (0–100) |
| `idade_mediana` | Censo 2022 | estrutural (2022) | anos | produção | |
| `indice_envelhecimento` | Censo 2022 | estrutural (2022) | índice (por 100) | produção | |
| `razao_sexo` | Censo 2022 | estrutural (2022) | índice (por 100) | produção | |
| `densidade_demografica` | derivada (`pop_2022/area_km2`) | estrutural (2022) | hab/km² | produção | |
| `pct_pop_indigena` | derivada (Censo 2022) | estrutural (2022) | fração 0–1 | produção | |
| `pct_pop_quilombola` | derivada (Censo 2022) | estrutural (2022) | fração 0–1 | produção | |

**PIB dos municípios (`pib_municipio`, defasagem 2)**:

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `pib_per_capita` | `pib_municipio` ÷ `populacao_municipio` (mesmo ano do PIB) | 2022 | R$ correntes | produção | |
| `log_pib_per_capita` | `log1p(pib_per_capita)` | 2022 | log(R$) | produção | |
| `pct_va_agropecuaria` | `pib_municipio` | 2022 | fração 0–1 | produção | **100% NaN para o alvo 2024** — ver observação acima |
| `pct_va_industria` | `pib_municipio` | 2022 | fração 0–1 | produção | **100% NaN para o alvo 2024** — ver observação acima |
| `pct_va_servicos` | `pib_municipio` | 2022 | fração 0–1 | produção | **100% NaN para o alvo 2024** — ver observação acima |
| `pct_va_adespss` | `pib_municipio` | 2022 | fração 0–1 | produção | **100% NaN para o alvo 2024** — ver observação acima |

**População (`populacao_municipio`, defasagem 1)**:

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `populacao` | `populacao_municipio` | 2023 | contagem absoluta | produção | |
| `log_populacao` | `log1p(populacao)` | 2023 | log(hab.) | produção | |

**IDEB (`ideb_municipio`, defasagem 1)**:

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `ideb_ai` | `ideb_municipio` | 2023 | índice IDEB (0–10) | produção | ~2,5% faltante |
| `taxa_aprovacao_ideb_ai` | `ideb_municipio` | 2023 | pontos percentuais (0–100) | produção | |
| `rendimento_ideb_ai` | `ideb_municipio` | 2023 | fração 0–1 | produção | |
| `nota_saeb_lp_ai` | `ideb_municipio` | 2023 | escala Saeb | produção | ~2,5% faltante |
| `nota_saeb_mat_ai` | `ideb_municipio` | 2023 | escala Saeb | produção | ~2,5% faltante |
| `projecao_ideb_ai` | `ideb_municipio` | 2023 | índice IDEB (0–10) | produção | **100% faltante** na fonte committed |

**Indicadores educacionais INEP (`indicadores_municipio`, defasagem 1)**:

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `atu_ai` | `indicadores_municipio` | 2023 | alunos/turma | produção | |
| `had_ai` | `indicadores_municipio` | 2023 | horas-aula/dia | produção | ~27,2% faltante |
| `tdi_ai` | `indicadores_municipio` | 2023 | pontos percentuais (0–100) | produção | taxa de distorção idade-série |
| `taxa_aprovacao_ai` | `indicadores_municipio` | 2023 | pontos percentuais (0–100) | produção | |
| `taxa_reprovacao_ai` | `indicadores_municipio` | 2023 | pontos percentuais (0–100) | produção | |
| `taxa_abandono_ai` | `indicadores_municipio` | 2023 | pontos percentuais (0–100) | produção | |
| `dsu_ai` | `indicadores_municipio` | 2023 | índice (0–100) | produção | |
| `afd_ai_grupo1` | `indicadores_municipio` | 2023 | pontos percentuais (0–100) | produção | docentes adequadamente formados |
| `ied_ai_nivel1` | `indicadores_municipio` | 2023 | pontos percentuais (0–100) | produção | esforço docente nível 1 |
| `ird_baixa` | `indicadores_municipio` | 2023 | pontos percentuais (0–100) | produção | regularidade docente baixa |

**Censo Escolar (`censo_escolar_municipio`, defasagem 1)**:

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `n_escolas_ai` | `censo_escolar_municipio` | 2023 | contagem absoluta | produção | |
| `pct_escolas_rurais` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `pct_escolas_internet` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `pct_escolas_biblioteca` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `pct_escolas_esgoto_rede` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `pct_escolas_agua_potavel` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `pct_escolas_energia_rede` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `pct_escolas_lab_informatica` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `pct_escolas_quadra` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `pct_escolas_alimentacao` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `matriculas_ai` | `censo_escolar_municipio` | 2023 | contagem absoluta | produção | |
| `docentes_ai` | `censo_escolar_municipio` | 2023 | contagem absoluta | produção | |
| `pct_matriculas_integral_ai` | `censo_escolar_municipio` | 2023 | fração 0–1 | produção | |
| `alunos_por_turma_ai` | `censo_escolar_municipio` | 2023 | razão (alunos/turma) | produção | |
| `alunos_por_docente_ai` | derivada (`matriculas_ai/docentes_ai`) | 2023 | razão (alunos/docente) | produção | |

**Bolsa Família (`bolsa_familia_municipio`, defasagem 1, dez/t−1)**:

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `pct_familias_bolsa_familia` | `familias_bf (dez/2023) / domicilios_2022` | 2023 | **fração, sem cap (pode passar de 1)** | produção | mediana ~0,25; máximo observado ~1,59 em municípios pequenos; na base de município, a linha de 2023 pede referência ≤ 2022 (inexistente ainda) e fica 100% NaN — ver seção "Externas referenciadas ao próprio ano" |

### LOO da escola (`contexto_escola_loo`/`participacao_escola_loo`, regime diagnóstico)

Calculadas excluindo o próprio aluno — usadas só para interpretação/EDA, nunca no modelo
entregue (usar a nota/proficiência de colegas do próprio ano letivo, mesmo que sem o próprio
aluno, ainda é uma agregação do mesmo ano que o alvo).

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `taxa_escola_loo` | Silver `alunos` (LOO) | 2024 | fração 0–1 | diagnóstico | taxa de alfabetização dos colegas de escola, excluindo o aluno |
| `prof_media_escola_loo` | Silver `alunos` (LOO) | 2024 | escala do teste (mesma do corte 743) | diagnóstico | |
| `n_alunos_escola` | Silver `alunos` (LOO) | 2024 | contagem absoluta | diagnóstico | nº de colegas com nota, excluindo o aluno |
| `taxa_participacao_escola_loo` | Gold `perfil_escola` (LOO) | 2024 | fração 0–1 | diagnóstico | participação da escola, excluindo o aluno |

---

## `base_modelagem_municipio`

Grão `(ano, id_municipio)`, rede municipal, anos 2023 e 2024. Usada pelo modelo de risco de não
atingir a meta de alfabetização de 2030 (Task 16) — cada linha usa o resultado e o contexto externo
do próprio `ano` para prever o resultado de `ano + 1`
(`taxa_prox`/`situacao_meta_prox`/`nao_atingiu_prox`), NaN em 2024 por não haver 2025 na Gold ainda.

### Identificadores (regime *fora*)

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `ano` | derivada | — | inteiro | fora | grão da tabela, não é feature previsora do próprio ano |
| `id_municipio` | Gold `meta_vs_resultado` | ano da linha | inteiro (código IBGE) | fora | identificador |
| `nome_municipio` | `diretorios_municipio` | estrutural | texto | fora | identificador textual |

### Resultado Gold do próprio ano (regime produção — descreve o estado atual, usado para prever o próximo ano)

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `sigla_uf` | Gold `meta_vs_resultado` | ano da linha | categórico (UF) | produção | |
| `alunos_com_nota` | Gold `meta_vs_resultado` | ano da linha | contagem absoluta | produção | |
| `taxa_alfabetizacao` | Gold `meta_vs_resultado` | ano da linha | pontos percentuais (0–100) | produção | resultado do próprio ano (não é o alvo desta tabela) |
| `ic95` | Gold `meta_vs_resultado` | ano da linha | pontos percentuais (0–100) | produção | |
| `meta_ano` | Gold `meta_vs_resultado` | ano da linha | pontos percentuais (0–100) | fora | ~49% NaN (2023 inteiro é `sem_meta`); distribuição muda entre treino/aplicação (ver Task 16) |
| `gap` | Gold `meta_vs_resultado` | ano da linha | pontos percentuais (0–100) | fora | idem `meta_ano` — mesma razão |
| `situacao_meta` | Gold `meta_vs_resultado` | ano da linha | categórico | fora | idem — 2023 inteiro é `sem_meta` |
| `taxa_participacao` | Gold `evolucao_temporal` | ano da linha | pontos percentuais (0–100) | produção | |
| `proficiencia_media` | Gold `evolucao_temporal` | ano da linha | escala do teste (mesma do corte 743) | produção | **não é 0–100** |
| `criancas_nao_alfabetizadas` | Gold `evolucao_temporal` | ano da linha | contagem absoluta | produção | |
| `pct_nivel_0` … `pct_nivel_8` | Gold `distribuicao_proficiencia` (rede municipal) | ano da linha | pontos percentuais (0–100) | produção | |
| `pct_critico` | Gold `distribuicao_proficiencia` (rede municipal) | ano da linha | pontos percentuais (0–100) | produção | |
| `pct_atencao` | Gold `distribuicao_proficiencia` (rede municipal) | ano da linha | pontos percentuais (0–100) | produção | |
| `pct_quase_la` | Gold `distribuicao_proficiencia` (rede municipal) | ano da linha | pontos percentuais (0–100) | produção | |

### Externas referenciadas ao próprio `ano` da linha (mesmas fontes e escalas da base de aluno)

Mesmas 55 colunas de `diretorios_municipio`/Censo 2022/PIB/população/IDEB/indicadores/censo
escolar/Bolsa Família descritas na seção "Externas por fonte" acima — origem, escala e
observações idênticas (`regiao`, `capital_uf`, `amazonia_legal`, `latitude`, `longitude`,
`pop_2022`, `domicilios_2022`, `area_km2`, `taxa_alfabetizacao_adultos`, `idade_mediana`,
`indice_envelhecimento`, `razao_sexo`, `densidade_demografica`, `pct_pop_indigena`,
`pct_pop_quilombola`, `pib_per_capita`, `log_pib_per_capita`, `pct_va_agropecuaria`,
`pct_va_industria`, `pct_va_servicos`, `pct_va_adespss`, `populacao`, `log_populacao`, `ideb_ai`,
`taxa_aprovacao_ideb_ai`, `rendimento_ideb_ai`, `nota_saeb_lp_ai`, `nota_saeb_mat_ai`,
`projecao_ideb_ai`, `atu_ai`, `had_ai`, `tdi_ai`, `taxa_aprovacao_ai`, `taxa_reprovacao_ai`,
`taxa_abandono_ai`, `dsu_ai`, `afd_ai_grupo1`, `ied_ai_nivel1`, `ird_baixa`, `n_escolas_ai`,
`pct_escolas_rurais`, `pct_escolas_internet`, `pct_escolas_biblioteca`, `pct_escolas_esgoto_rede`,
`pct_escolas_agua_potavel`, `pct_escolas_energia_rede`, `pct_escolas_lab_informatica`,
`pct_escolas_quadra`, `pct_escolas_alimentacao`, `matriculas_ai`, `docentes_ai`,
`pct_matriculas_integral_ai`, `alunos_por_turma_ai`, `alunos_por_docente_ai`,
`pct_familias_bolsa_familia`). A diferença é o ano de referência efetivo por linha: como cada linha
usa `montar_contexto_externo(ano)` com o próprio `ano` da linha (não `ano + 1`), a linha de 2023
referencia um ano a **menos** por fonte do que a linha de 2024 (e do que a base de aluno, alvo
2024) — ex.: PIB na linha de 2023 é o de 2021 (t−2 relativo a 2023), enquanto a linha de 2024 e a
base de aluno usam 2022 (t−2 relativo a 2024).

Isso muda **onde** os dois gaps conhecidos aparecem nesta tabela (confirmado rodando
`montar_base_municipio()` de verdade e medindo `notna().mean()` por ano):
- `pct_va_*`: a linha de 2023 usa PIB **2021**, que tem a quebra setorial publicada (~100%
  populado); a linha de 2024 usa PIB **2022**, sem quebra setorial (**100% NaN**) — mesmo gap da
  base de aluno, só que aqui aparece isolado na linha de 2024, não nas duas.
- `pct_familias_bolsa_familia`: a linha de 2023 pede uma referência ≤ 2022 que a fonte não tem
  (**100% NaN**, ver observação acima); a linha de 2024 usa a referência 2023, igual à base de
  aluno (**~100% populado**).

`nome_municipio` já está no grupo "identificadores" acima e não se repete aqui.

### Meta e alvos do ano seguinte

| coluna | origem | ano de referência | escala | regime | observação |
|---|---|---|---|---|---|
| `meta_prox` | Silver `metas` (`meta_alfabetizacao_{ano+1}`) | ano da linha + 1 | pontos percentuais (0–100) | produção | pactuada antes do resultado de `ano+1` — **entra** como feature (ver Task 16) |
| `taxa_prox` | Gold `meta_vs_resultado` (ano+1) | ano da linha + 1 | pontos percentuais (0–100) | fora | **alvo** (regressão) — NaN em 2024 (sem 2025 na Gold) |
| `situacao_meta_prox` | Gold `meta_vs_resultado` (ano+1) | ano da linha + 1 | categórico | fora | usado só para derivar `nao_atingiu_prox` |
| `nao_atingiu_prox` | derivada (`situacao_meta_prox == "nao_atingiu"`) | ano da linha + 1 | binário (0.0/1.0) | fora | **alvo** (classificação) — NaN em 2024 |
