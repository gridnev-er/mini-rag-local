from rag.config import load_config


def test_load_config_has_basic_keys():
    cfg = load_config("config.yaml")

    # Основные секции
    assert "paths" in cfg
    assert "chunking" in cfg
    assert "retriever" in cfg
    assert "generator" in cfg

    assert "data_dir" in cfg["paths"]
    assert "index_dir" in cfg["paths"]
    assert "chunk_size" in cfg["chunking"]
    assert "overlap" in cfg["chunking"]
    assert "top_k" in cfg["retriever"]
    assert "model" in cfg["generator"]
