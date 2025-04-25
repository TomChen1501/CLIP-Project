from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import torch
from utils import compute_embedding, find_k_nearest
from source.data_utils import ensure_file_exists, unzip_file

# --- Lifespan setup: Runs once when server starts ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure necessary files exist
    ensure_file_exists("Resource/list_attr_celeba.txt", "https://drive.google.com/uc?id=1FyDxSKdqfc3zbamWMyZxalGTpLZ70kfh")
    ensure_file_exists("Resource/img_align_celeba.zip", "https://drive.google.com/uc?id=1QoCujOf6xTGtXgasCZ_Fcp8e5tXLPMsA")
    unzip_file("Resource/img_align_celeba.zip", "Resource/img_align_celeba")
    ensure_file_exists("Resource/encoded_tensors.pt", "https://drive.google.com/uc?id=1Apj_3U8aEXQqr_2dBoE_TAJhzaB0vaY0")
    ensure_file_exists("Resource/all_image_embeddings.pt", "https://drive.google.com/uc?id=15z6Ah0EcbB_d6YTLaemYo8GHwrQPcNie")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_path = os.path.join(base_dir, "Resource", "all_image_embeddings.pt")
    print(f"Loading embeddings from {embedding_path}...")
    data = torch.load(embedding_path, weights_only=True, map_location=device)
    app.state.database_embeddings = data["embeddings"].to(torch.float32)
    app.state.all_images_filename = data["filename"]
    print("Embeddings loaded.")

    # Ensure runtime folders exist before mounting
    os.makedirs("uploaded_images", exist_ok=True)
    
    # Static serving
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/uploaded_images", StaticFiles(directory="uploaded_images"), name="uploaded_images")
    app.mount("/dataset_images", StaticFiles(directory="Resource/img_align_celeba"), name="dataset_images")

    yield

# --- FastAPI app ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Route: header ---
@app.head("/")
async def header():
    return JSONResponse({"message": "Welcome to the FastAPI server!"})

# --- Route: frontend ---
@app.get("/")
async def serve_frontend():
    return FileResponse("static/frontend.html")

# --- Route: upload and return K nearest ---
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), request: Request = None):
    upload_folder = "uploaded_images"
    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Compute embedding and find matches
    embedding_vector = compute_embedding(file_path)
    nearest_images = find_k_nearest(
        embedding_vector,
        request.app.state.database_embeddings,
        k=5
    )

    result_urls = [
        f"/dataset_images/{request.app.state.all_images_filename[index]}"
        for index in nearest_images
    ]

    return JSONResponse({"nearest_images": result_urls})
