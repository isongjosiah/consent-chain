"""Tests for fl.models.bilstm_casas — shape sanity + config respect."""

import torch

from fl.models.bilstm_casas import BiLSTMClassifier, BiLSTMConfig, build_from_yaml


def test_forward_shape_matches_num_classes():
    cfg = BiLSTMConfig(vocab_size=64, embedding_dim=8, hidden_dim=16, num_layers=1, num_classes=12)
    model = BiLSTMClassifier(cfg)
    x = torch.randint(0, cfg.vocab_size, size=(4, 25), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (4, 12)


def test_padding_index_is_zero_by_default():
    cfg = BiLSTMConfig(vocab_size=64)
    model = BiLSTMClassifier(cfg)
    # All-pad input should not raise (embedding(0) is the padding vector).
    x = torch.zeros((2, 10), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (2, cfg.num_classes)


def test_bidirectional_doubles_classifier_input_dim():
    cfg = BiLSTMConfig(vocab_size=16, hidden_dim=8, num_layers=1, bidirectional=True)
    model = BiLSTMClassifier(cfg)
    assert model.classifier.in_features == 16  # 8 * 2

    cfg_uni = BiLSTMConfig(vocab_size=16, hidden_dim=8, num_layers=1, bidirectional=False)
    model_uni = BiLSTMClassifier(cfg_uni)
    assert model_uni.classifier.in_features == 8


def test_single_layer_disables_inter_layer_dropout():
    """With num_layers=1, PyTorch warns if dropout != 0 — we should not."""
    cfg = BiLSTMConfig(vocab_size=16, hidden_dim=8, num_layers=1, dropout=0.5)
    model = BiLSTMClassifier(cfg)
    assert model.lstm.dropout == 0.0  # honoured override


def test_build_from_yaml_respects_config():
    yaml_model = {
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,
    }
    model = build_from_yaml(yaml_model, vocab_size=200)
    assert isinstance(model, BiLSTMClassifier)
    assert model.embedding.num_embeddings == 200
    assert model.embedding.embedding_dim == 32
    assert model.lstm.hidden_size == 64
    assert model.lstm.num_layers == 2
    assert model.lstm.bidirectional is True
    assert model.classifier.out_features == 12


def test_forward_works_on_cpu_with_realistic_dims():
    """Smoke check at the actual production scale (vocab=256, seq=2000, batch=4)."""
    cfg = BiLSTMConfig(vocab_size=256, embedding_dim=64, hidden_dim=128, num_layers=2)
    model = BiLSTMClassifier(cfg)
    x = torch.randint(0, cfg.vocab_size, size=(4, 2000), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (4, 12)
    assert torch.isfinite(logits).all()
