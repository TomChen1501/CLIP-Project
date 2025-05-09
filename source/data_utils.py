import pandas as pd
import os
import clip
from PIL import Image
import torch
from tqdm import tqdm
import gdown
import zipfile


def load_celeb_attribute(file_path = 'Resource/list_attr_celeba.txt'):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        attributes = lines[1].strip().split()

        data = []
        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            labels = list(map(int, parts[1:]))
            data.append([filename] + labels)

        df = pd.DataFrame(data, columns=['Filename'] + attributes)
        return df
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error:{e} occurred. Please try again.")
        return None

def load_database_embeddings(image_dir='Resource/img_align_celeba/img_align_celeba', file_path='Resource/list_attr_celeba.txt'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    
    df = load_celeb_attribute(file_path)
    all_images_filename = df['Filename'].tolist()

    encoded_features = []
    for filename in tqdm(all_images_filename):
        image_path = os.path.join(image_dir, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            encoded_features.append(image_features)
    
    encoded_features = torch.cat(encoded_features)
    return all_images_filename, encoded_features

def ensure_file_exists(file_path, url):
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Downloading from {url}...")
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        gdown.download(url, file_path, quiet=False)
        print(f"File {file_path} downloaded successfully.")
    else:
        print(f"File {file_path} already exists.")

def unzip_file(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        print(f"Unzipping {zip_path} into {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Unzipping completed.")
    else:
        print(f"{extract_dir} already exists, skipping unzip.")

if __name__ == "__main__":
    df = load_celeb_attribute()
    print(df.head())
    