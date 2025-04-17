from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import torch
from utils import compute_embedding, find_k_nearest
from source.data_utils import load_database_embeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploaded_images", StaticFiles(directory="uploaded_images"), name="uploaded_images")  # for uploaded files
app.mount("/dataset_images", StaticFiles(directory="Resource/img_align_celeba/img_align_celeba"), name="dataset_images") 

# Pre-load your database embeddings at server start
@app.on_event("startup")
def load_embeddings():
    global database_embeddings, all_images_filename
    data = torch.load("all_image_embeddings.pt", weights_only=True)
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
