import torch
from models.mlp.mlp_model import MLPModel
from models.mlp.inference_wrapper import AttributeTransferEngine
import pytest

def test_mlp_model_forward():
    embedding_dim = 512
    hidden_dim = 128
    model = MLPModel(embedding_dim, hidden_dim)

    x = torch.randn(16, embedding_dim)  # batch of 16 embeddings
    out = model(x)

    assert out.shape == x.shape, f"Expected output shape {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_transform_embedding_shape():
    embedding_dim = 512
    hidden_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine = AttributeTransferEngine(embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device)

    test_input = torch.randn(1, embedding_dim).to(device)
    transformed = engine.transform_embedding(test_input)

    assert isinstance(transformed, torch.Tensor), "Output is not a tensor"
    assert transformed.shape == test_input.shape, f"Expected shape {test_input.shape}, got {transformed.shape}"
    assert not torch.isnan(transformed).any(), "Transformed output contains NaNs"

def test_end_to_end_pipeline():
    engine = AttributeTransferEngine(embedding_dim=512, hidden_dim=128)
    X_src = torch.randn(64, 512)
    X_tgt = torch.randn(64, 512)
    mse, cos = engine.train_on_pair(X_src, X_tgt, num_epochs=3, lr=0.001)

    transformed = engine.transform_embedding(X_src)
    assert transformed.shape == X_src.shape
    assert transformed.dtype == X_src.dtype

def test_mlp_weights_change_after_training():
    engine = AttributeTransferEngine(embedding_dim=512, hidden_dim=128)
    X_src = torch.randn(64, 512)
    X_tgt = torch.randn(64, 512)

    # clone initial weights
    initial_weights = [p.clone().detach() for p in engine.model.parameters()]
    engine.train_on_pair(X_src, X_tgt, num_epochs=5, lr=1e-2)

    for init, trained in zip(initial_weights, engine.model.parameters()):
        assert not torch.equal(init, trained), "Weights did not update during training"

def test_training_loss_decreases():
    engine = AttributeTransferEngine(embedding_dim=512, hidden_dim=128)
    X_src = torch.randn(128, 512)
    X_tgt = torch.randn(128, 512)

    # Run longer training and compare loss
    losses = []
    for _ in range(3):
        mse, _ = engine.train_on_pair(X_src, X_tgt, num_epochs=1, lr=1e-2)
        losses.append(mse)

    assert losses[-1] <= losses[0] + 1e-4, f"Loss did not decrease: {losses}"

def test_deterministic_output_with_seed():
    torch.manual_seed(114514)
    engine1 = AttributeTransferEngine(embedding_dim=512, hidden_dim=128)
    X_src = torch.randn(32, 512)
    X_tgt = torch.randn(32, 512)
    mse1, cos1 = engine1.train_on_pair(X_src, X_tgt, num_epochs=5, lr=1e-3)

    torch.manual_seed(114514)
    engine2 = AttributeTransferEngine(embedding_dim=512, hidden_dim=128)
    X_src2 = torch.randn(32, 512)
    X_tgt2 = torch.randn(32, 512)
    mse2, cos2 = engine2.train_on_pair(X_src2, X_tgt2, num_epochs=5, lr=1e-3)

    assert abs(mse1 - mse2) < 1e-6
    assert abs(cos1 - cos2) < 1e-6

def test_transform_wrong_shape_raises():
    engine = AttributeTransferEngine(embedding_dim=512, hidden_dim=128)
    with pytest.raises(RuntimeError):
        bad_input = torch.randn(1, 256)  # wrong dim
        engine.transform_embedding(bad_input)

def test_zero_input_stability():
    model = MLPModel(embedding_dim=512, hidden_dim=128)
    zero_input = torch.zeros(16, 512)
    output = model(zero_input)
    assert torch.all(torch.isfinite(output)), "Zero input produced NaN or Inf"

def test_batch_independence():
    model = MLPModel(embedding_dim=512, hidden_dim=128)
    x1 = torch.randn(1, 512)
    x2 = torch.cat([x1, torch.randn(1, 512)], dim=0)
    out1 = model(x1)
    out2 = model(x2)[0]
    assert torch.allclose(out1, out2, atol=1e-6), "Single vs. batched input mismatch"

def test_output_magnitude_reasonable():
    model = MLPModel(embedding_dim=512, hidden_dim=128)
    x = torch.randn(16, 512)
    y = model(x)
    assert y.abs().max() < 100, "Output values too large; may indicate instability"

def test_inference_before_training():
    engine = AttributeTransferEngine(embedding_dim=512, hidden_dim=128)
    x = torch.randn(5, 512)
    transformed = engine.transform_embedding(x)
    assert transformed.shape == x.shape
    assert transformed.dtype == x.dtype

def test_consistent_output_cpu_gpu():
    if not torch.cuda.is_available():
        return

    model_cpu = MLPModel(512, 128)
    model_gpu = MLPModel(512, 128).to("cuda")
    model_gpu.load_state_dict(model_cpu.state_dict())  # ensure same weights

    x = torch.randn(4, 512)
    out_cpu = model_cpu(x)
    out_gpu = model_gpu(x.cuda()).cpu()
    assert torch.allclose(out_cpu, out_gpu, atol=1e-4), "Mismatch between CPU and GPU outputs"


