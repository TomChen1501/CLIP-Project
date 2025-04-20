import torch
import clip
from PIL import Image
import os
from source.data_utils import load_celeb_attribute
import tqdm

def compute_embedding(img_path):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features

def find_k_nearest(query, embedding_database, k=4):
    # Compute the cosine similarity between the input embedding and the database embeddings
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    query = query.to(device)
    embedding_database = embedding_database.to(device)

    # Match dtype explicitly to avoid issues
    query = query.to(embedding_database.dtype)

    query_norm = query / query.norm(dim=-1, keepdim=True)
    embedding_database_norm = embedding_database / embedding_database.norm(dim=-1, keepdim=True)

    similarities = torch.matmul(embedding_database_norm, query_norm.T).squeeze(1)
    
    # Get the indices of the top k most similar embeddings
    topk_similarities, topk_indices = torch.topk(similarities, k=k, largest=True)

    return topk_indices