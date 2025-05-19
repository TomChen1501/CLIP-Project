import os
import tempfile
import torch
import zipfile
import pytest
from utility.data_utils import load_celeb_attribute, ensure_file_exists, unzip_file
from PIL import Image
from utility.utils import compute_embedding, find_k_nearest

def _create_dummy_image(path, mode='RGB', size=(224, 224), color=128):
    img = Image.new(mode, size, color)
    img.save(path)

def test_load_celeb_attribute_structure():
    df = load_celeb_attribute("Resource/list_attr_celeba.txt")
    assert df is not None
    assert "Filename" in df.columns
    assert "Smiling" in df.columns

def test_ensure_file_exists_creates_file(tmp_path):
    dummy_url = "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"
    test_file = tmp_path / "test.txt"
    ensure_file_exists(str(test_file), dummy_url)
    assert os.path.exists(test_file)

def test_unzip_file_extracts(tmp_path):
    zip_path = tmp_path / "dummy.zip"
    extract_dir = tmp_path / "extracted"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        dummy_file = tmp_path / "test.txt"
        dummy_file.write_text("test content")
        zipf.write(dummy_file, arcname="test.txt")

    unzip_file(zip_path, extract_dir)
    extracted_file = extract_dir / "test.txt"
    assert extracted_file.exists()
    assert extracted_file.read_text() == "test content"

def test_compute_embedding_output_shape(tmp_path, monkeypatch):
    img_path = tmp_path / "test_rgb.png"
    _create_dummy_image(img_path, mode='RGB')

    # Stub out clip.load to avoid heavy model loading
    class DummyModel:
        def encode_image(self, x): return torch.ones(1, 512)
    dummy_model = DummyModel()
    def dummy_preprocess(img):
        return torch.zeros(3, 224, 224)
    monkeypatch.setattr("clip.load", lambda *args, **kwargs: (dummy_model, dummy_preprocess))

    emb = compute_embedding(str(img_path))
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (1, 512)

def test_embedding_on_grayscale_image(tmp_path, monkeypatch):
    img_path = tmp_path / "test_gray.png"
    _create_dummy_image(img_path, mode='L')

    class DummyModel:
        def encode_image(self, x): return torch.ones(1, 512)
    dummy_model = DummyModel()
    def dummy_preprocess(img):
        return torch.zeros(3, 224, 224)
    monkeypatch.setattr("clip.load", lambda *args, **kwargs: (dummy_model, dummy_preprocess))

    emb = compute_embedding(str(img_path))
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (1, 512)

def test_find_k_nearest_outputs():
    db = torch.randn(100, 512)
    query = db[0].unsqueeze(0)
    indices = find_k_nearest(query, db, k=5)
    assert isinstance(indices, torch.Tensor)
    assert len(indices) == 5

def test_load_celeb_attribute_missing_file():
    with pytest.raises(FileNotFoundError):
        load_celeb_attribute("nonexistent_file.txt")

def test_ensure_file_exists_no_download(tmp_path):
    test_file = tmp_path / "already_there.txt"
    test_file.write_text("I'm already here")
    ensure_file_exists(test_file, "https://example.com/should_not_be_used")
    assert test_file.read_text() == "I'm already here"

def test_unzip_file_rejects_non_zip(tmp_path):
    non_zip = tmp_path / "not_a_zip.txt"
    non_zip.write_text("This is not a zip")
    with pytest.raises(zipfile.BadZipFile):
        unzip_file(non_zip, tmp_path / "out")

def test_compute_embedding_invalid_path():
    with pytest.raises(FileNotFoundError):
        _ = compute_embedding("this_file_does_not_exist.jpg")

def test_embedding_on_grayscale_image(tmp_path):
    # create a dummy grayscale image
    img_path = tmp_path / "test_gray.png"
    Image.new("L", (64, 64), color=128).save(img_path)

    emb = compute_embedding(str(img_path))
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (1, 512)

def test_k_nearest_sorted_by_distance():
    db = torch.eye(512)[:10]
    query = db[0].unsqueeze(0)  # identical to first entry
    indices = find_k_nearest(query, db, k=5)
    assert indices[0] == 0

