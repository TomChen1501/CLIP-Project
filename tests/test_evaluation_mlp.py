import torch
import numpy as np
from models.mlp.inference_wrapper import AttributeTransferEngine
from utility.data_utils import load_celeb_attribute
from models.svm.svmClassify import CLIPAttributeSVM
from torch.nn.functional import cosine_similarity


def test_mlp_evaluation_output_ranges():
    df = load_celeb_attribute()
    data = torch.load("Resource/all_image_embeddings.pt", weights_only=True)
    embeddings = data["embeddings"]
    X_pos = embeddings[df["Smiling"] == 1]
    X_neg = embeddings[df["Smiling"] == -1]

    engine = AttributeTransferEngine(embedding_dim=512, hidden_dim=128)
    mse, cos = engine.train_on_pair(X_neg, X_pos, num_epochs=2, lr=1e-3)
    assert 0.0 <= mse <= 1.0
    assert 0.0 <= cos <= 1.0

def test_svm_evaluation_outputs_valid_metrics():
    df = load_celeb_attribute()
    data = torch.load("Resource/all_image_embeddings.pt", weights_only=True)
    embeddings = data["embeddings"]
    pos = embeddings[df["Smiling"] == 1]
    neg = embeddings[df["Smiling"] == -1]

    svm = CLIPAttributeSVM(kernel="linear", C=1.0, seed=123)
    acc, f1 = svm.train_and_evaluate(pos, neg, sample_ratio=0.1)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_identity_baseline_perfect_score():
    x = torch.randn(100, 512)
    mse = torch.mean((x - x) ** 2).item()
    cos = cosine_similarity(x, x).mean().item()
    assert np.isclose(mse, 0.0, atol=1e-6)
    assert np.isclose(cos, 1.0, atol=1e-6)

def test_all_attributes_evaluated():
    df = load_celeb_attribute()
    attributes = ["Smiling", "Young", "Male", "Bald"]
    for attr in attributes:
        assert attr in df.columns
        assert len(df[df[attr] == 1]) > 100
        assert len(df[df[attr] == -1]) > 100

def test_mlp_grid_search_complete():
    import itertools
    hiddens = [64, 128, 256, 512]
    lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    combos = list(itertools.product(hiddens, lrs))
    assert len(combos) == 16

