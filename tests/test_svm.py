import torch
import numpy as np
import os
from models.svm.svmClassify import CLIPAttributeSVM
from utility.data_utils import load_celeb_attribute
import pytest
from sklearn.exceptions import NotFittedError

def get_dummy_data():
    pos = torch.randn(50, 512)
    neg = torch.randn(50, 512)
    return pos, neg

def test_svm_initialization():
    svm = CLIPAttributeSVM(kernel="rbf", C=10.0)
    assert svm.kernel == "rbf"
    assert svm.C == 10.0

def test_prepare_data_shapes_and_labels():
    pos, neg = get_dummy_data()
    svm = CLIPAttributeSVM(test_size=0.2)
    X_train, X_test, y_train, y_test = svm.prepare_data(pos, neg, sample_ratio=1.0)

    assert X_train.shape[1] == 512
    assert X_test.shape[1] == 512
    assert len(X_train) > 0 and len(X_test) > 0
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})

def test_svm_train_and_predict():
    pos, neg = get_dummy_data()
    svm = CLIPAttributeSVM(kernel="linear", C=1.0)
    X_train, X_test, y_train, y_test = svm.prepare_data(pos, neg, sample_ratio=1.0)
    svm.train(X_train, y_train)
    preds = svm.model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)).issubset({0, 1})

def test_svm_train_and_evaluate_range():
    pos, neg = get_dummy_data()
    svm = CLIPAttributeSVM(kernel="rbf", C=1.0)
    acc, f1 = svm.train_and_evaluate(pos, neg, sample_ratio=0.5)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0

def test_svm_save_and_load(tmp_path):
    pos, neg = get_dummy_data()
    model_path = tmp_path / "svm_model_test.pkl"

    # Train and save
    svm = CLIPAttributeSVM()
    svm.train_and_evaluate(pos, neg)
    svm.save_model(model_path)
    assert os.path.exists(model_path)

    # Load into new object
    new_svm = CLIPAttributeSVM()
    new_svm.load_model(model_path)
    assert hasattr(new_svm.model, "predict")

def test_all_kernels_supported():
    pos, neg = get_dummy_data()
    for kernel in ["linear", "rbf", "poly"]:
        svm = CLIPAttributeSVM(kernel=kernel)
        acc, f1 = svm.train_and_evaluate(pos, neg, sample_ratio=0.5)
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= f1 <= 1.0

def test_svm_on_real_clip_embeddings():
    # Load real data
    df = load_celeb_attribute()
    data = torch.load("Resource/all_image_embeddings.pt", weights_only=True)
    embeddings = data["embeddings"]

    # Select an attribute and get matching indices
    attribute = "Smiling"
    pos_idx = df[df[attribute] == 1].index.tolist()
    neg_idx = df[df[attribute] == -1].index.tolist()

    X_pos = embeddings[pos_idx]
    X_neg = embeddings[neg_idx]

    # Sanity check sizes
    assert X_pos.shape[1] == 512
    assert X_neg.shape[1] == 512
    assert len(X_pos) > 100 and len(X_neg) > 100

    # Run evaluation
    svm = CLIPAttributeSVM(kernel="linear", C=1.0)
    acc, f1 = svm.train_and_evaluate(X_pos, X_neg, sample_ratio=0.1)

    print(f"[REAL] Smiling | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    assert 0.7 <= acc <= 1.0
    assert 0.7 <= f1 <= 1.0

def test_sample_ratio_one_does_not_crash():
    pos, neg = get_dummy_data()
    svm = CLIPAttributeSVM()
    acc, f1 = svm.train_and_evaluate(pos, neg, sample_ratio=1.0)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0

def test_predict_before_train_raises():
    pos, neg = get_dummy_data()
    svm = CLIPAttributeSVM()
    X_train, X_test, y_train, y_test = svm.prepare_data(pos, neg)

    with pytest.raises(NotFittedError):
        _ = svm.model.predict(X_test)

def test_invalid_kernel_raises():
    with pytest.raises(ValueError):
        _ = CLIPAttributeSVM(kernel="unsupported")

def test_very_small_dataset_still_runs():
    pos = torch.randn(5, 512)
    neg = torch.randn(5, 512)
    svm = CLIPAttributeSVM()
    acc, f1 = svm.train_and_evaluate(pos, neg, sample_ratio=1.0)
    assert 0.0 <= acc <= 1.0

def test_evaluation_is_reproducible():
    data = torch.load("Resource/encoded_tensors.pt", weights_only=True)
    smile = data["smile"]
    unsmile = data["unsmile"]

    seed = 100
    svm1 = CLIPAttributeSVM(kernel="linear", C=1.0, test_size=0.2, seed=seed)
    acc1, f1_1 = svm1.train_and_evaluate(smile, unsmile, sample_ratio=0.1)

    svm2 = CLIPAttributeSVM(kernel="linear", C=1.0, test_size=0.2, seed=seed)
    acc2, f1_2 = svm2.train_and_evaluate(smile, unsmile, sample_ratio=0.1)

    assert abs(acc1 - acc2) < 1e-2, f"Accuracy mismatch: {acc1} vs {acc2}"
    assert abs(f1_1 - f1_2) < 1e-2, f"F1 mismatch: {f1_1} vs {f1_2}"

def test_svm_zero_input_predicts_binary():
    svm = CLIPAttributeSVM()
    pos = torch.zeros(50, 512)
    neg = torch.zeros(50, 512)
    acc, f1 = svm.train_and_evaluate(pos, neg, sample_ratio=1.0)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0

def test_svm_batch_independence():
    # Positive samples: 0–4, Negative samples: 5–9
    pos = torch.arange(5).unsqueeze(1).float()
    neg = torch.arange(5, 10).unsqueeze(1).float()

    svm = CLIPAttributeSVM(test_size=0.4, seed=42)

    X_train, X_test, y_train, y_test = svm.prepare_data(pos, neg, sample_ratio=1.0)

    train_set = set(map(tuple, X_train.tolist()))
    test_set  = set(map(tuple, X_test.tolist()))
    assert train_set.isdisjoint(test_set), "Train and test share samples!"

    total = pos.shape[0] + neg.shape[0]
    assert len(X_train) + len(X_test) == total

    combined = np.concatenate([X_train, X_test])
    combined_labels = np.concatenate([y_train, y_test])
    id_to_label = {i: 1 for i in range(5)}
    id_to_label.update({i: 0 for i in range(5, 10)})
    for row, lbl in zip(combined.tolist(), combined_labels.tolist()):
        sample_id = row[0]
        assert lbl == id_to_label[sample_id]

    X2_train, X2_test, y2_train, y2_test = svm.prepare_data(pos, neg, sample_ratio=1.0)
    assert np.array_equal(X_train, X2_train)
    assert np.array_equal(X_test,  X2_test)
    assert np.array_equal(y_train, y2_train)
    assert np.array_equal(y_test,  y2_test)



