import os
import io
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import main

# --- a dummy SVM so we never load real models ---
class DummySVM:
    def __init__(self):
        # mimic CLIPAttributeSVM interface
        self.model = self
    def load_model(self, path):
        pass
    def predict(self, embedding):
        return [True]

@pytest.fixture(autouse=True)
def stub_everything(monkeypatch):
    monkeypatch.setattr(main, "ensure_file_exists", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "unzip_file",      lambda *args, **kwargs: None)

    dummy_embeddings = torch.zeros((10, 512))
    dummy_filenames  = [f"img_{i}.jpg" for i in range(10)]
    monkeypatch.setattr(main.torch, "load", 
        lambda *args, **kwargs: {"embeddings": dummy_embeddings, "filename": dummy_filenames}
    )
    monkeypatch.setattr(main, "CLIPAttributeSVM", DummySVM)
    monkeypatch.setattr(main, "glob", 
        lambda pattern: [f"models/trained_models/svm_{attr}_model.pkl" 
                         for attr in ("Smiling","Young","Male","Bald")]
    )
    monkeypatch.setattr(main, "compute_embedding", lambda path: torch.randn(1, 512))
    monkeypatch.setattr(main, "find_k_nearest",    lambda emb, db, k: list(range(k)))

    os.makedirs("uploaded_images", exist_ok=True)

@pytest.fixture(scope="module")
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def test_head_route(client):
    r = client.head("/")
    assert r.status_code == 200


def test_serve_frontend_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "html" in r.headers["content-type"]



def test_upload_no_file(client):
    r = client.post("/upload/", files={})
    assert r.status_code == 422


def test_upload_invalid_file_type(client):
    fake = io.BytesIO(b"not an image")
    r = client.post("/upload/", files={"file": ("x.txt", fake, "text/plain")})
    assert r.status_code == 500


def test_invalid_file_extension(client):
    fake = io.BytesIO(b"not an image")
    r = client.post("/upload/", files={"file": ("bad.exe", fake, "application/octet-stream")})
    assert r.status_code == 500


def test_uploaded_image_is_saved(tmp_path, client):
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    src = tmp_path / "test_upload.png"
    img.save(src)
    with open(src, "rb") as f:
        r = client.post("/upload/", files={"file": ("test_upload.png", f, "image/png")})

    assert r.status_code == 500
    assert os.path.exists(os.path.join("uploaded_images", "test_upload.png"))
