import os
import torch
from svmClassify import CLIPAttributeSVM
from source.data_utils import load_celeb_attribute

data = torch.load('Resource/all_image_embeddings.pt', weights_only=True)
df = load_celeb_attribute()

attributes_to_train = ['Smiling', 'Young']  # Add more attributes as needed

os.makedirs("trained_models", exist_ok=True)

for attribute in attributes_to_train:
    print(f"Training model for attribute: {attribute}")

    pos_indices = df[df[attribute] == 1].index.tolist()
    neg_indices = df[df[attribute] == -1].index.tolist()

    all_embeddings = data['embeddings']  # Assumes all data under one key
    positive_tensor = all_embeddings[pos_indices]
    negative_tensor = all_embeddings[neg_indices]

    svm = CLIPAttributeSVM(kernel="linear", C=1.0, test_size=0.2, seed=114514)
    svm.train_and_evaluate(positive_tensor, negative_tensor, sample_ratio=0.1)
    svm.save_model(path=f"trained_models/svm_{attribute.lower()}_model.pkl")
