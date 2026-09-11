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
