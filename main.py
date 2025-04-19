from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import torch
from utils import compute_embedding, find_k_nearest
from source.data_utils import load_database_embeddings, ensure_file_exists, unzip_file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the necessary files exist before starting the server
ensure_file_exists("Resource/list_attr_celeba.txt", "https://drive.google.com/uc?id=1FyDxSKdqfc3zbamWMyZxalGTpLZ70kfh")
ensure_file_exists("Resource/img_align_celeba.zip", "https://drive.google.com/uc?id=1QoCujOf6xTGtXgasCZ_Fcp8e5tXLPMsA")
unzip_file("Resource/img_align_celeba.zip", "Resource/img_align_celeba")
ensure_file_exists("embeddings/encoded_tensors.pt", "https://drive.google.com/uc?id=1Apj_3U8aEXQqr_2dBoE_TAJhzaB0vaY0")
ensure_file_exists("embeddings/all_image_embeddings.pt", "https://drive.google.com/uc?id=15z6Ah0EcbB_d6YTLaemYo8GHwrQPcNie")

app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/uploaded_images", StaticFiles(directory="uploaded_images"), name="uploaded_images") 
app.mount("/dataset_images", StaticFiles(directory="Resource/img_align_celeba/img_align_celeba"), name="dataset_images") 

# Pre-load your database embeddings at server start
@app.on_event("startup")
def load_embeddings():
    global database_embeddings, all_images_filename
    data = torch.load("embeddings/all_image_embeddings.pt", weights_only=True)
    all_images_filename = data['filename']
    database_embeddings = data['embeddings']

# ---------------- Route: Serve frontend HTML at root ----------------
@app.get("/")
async def serve_frontend():
    return FileResponse("static/frontend.html")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    upload_folder = "uploaded_images"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding_vector = compute_embedding(file_path)
    nearest_images = find_k_nearest(embedding_vector, database_embeddings, k=5)

    result_urls = [f"/dataset_images/{all_images_filename[index]}" for index in nearest_images]

    return JSONResponse({"nearest_images": result_urls})
