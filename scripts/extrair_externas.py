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
            WHERE ano >= {ANO_MIN} AND id_municipio IS NOT NULL
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


def registrar_metadados(destino: Path, nome: str, resumo: dict) -> None:
    meta_path = Path(destino) / "_metadados.json"
    metadados = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    metadados[nome] = {**resumo, "extraido_em": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    meta_path.write_text(json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")


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

    for nome in args.nomes:
        resumo = extrair(client, nome, args.destino)
        registrar_metadados(args.destino, nome, resumo)
        print(f"{nome:<26}{resumo['linhas']:>9,} linhas  {resumo['bytes_processados']/1e6:>8.1f} MB lidos")


if __name__ == "__main__":
    main()
