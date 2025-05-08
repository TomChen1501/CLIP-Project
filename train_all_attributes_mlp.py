import torch
import sys
import os
from mlp.inference_wrapper import AttributeTransferEngine
from source.data_utils import load_celeb_attribute

def train_all_attributes(df, embeddings, attributes_to_train, save_dir="trained_models", hidden_dim=256, num_epochs=100, lr=0.001):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = embeddings.shape[1]

    for attribute in attributes_to_train:
        
        print(f"Training model for attribute: {attribute}")
        engine = AttributeTransferEngine(embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device)

        pos_indices = df[df[attribute] == 1].index.tolist()
        neg_indices = df[df[attribute] == -1].index.tolist()

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            print(f"Skipping {attribute}: not enough data.")
            continue

        X_pos = embeddings[pos_indices]
        X_neg = embeddings[neg_indices]

        # model from X_pos to X_neg
        engine.train_on_pair(X_pos, X_neg, num_epochs=num_epochs, lr=lr)
        engine.save_model(os.path.join(save_dir, f"mlp_{attribute}.pth"))
        print(f"Model for {attribute} saved to {save_dir}")

        # model from X_neg to X_pos
        engine = AttributeTransferEngine(embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device)
        engine.train_on_pair(X_neg, X_pos, num_epochs=num_epochs, lr=lr)
        engine.save_model(os.path.join(save_dir, f"mlp_neg_{attribute}.pth"))
        print(f"Model for negated {attribute} saved to {save_dir}")

if __name__ == "__main__":
    # Load the CelebA attributes and embeddings
    df = load_celeb_attribute()
    d = torch.load('Resource/all_image_embeddings.pt', weights_only=True)
    embeddings = d['embeddings']

    # Hardcoded attributes which we are interested in training
    attributes_to_train = ['Smiling', 'Young'] 
    # attributes_to_train = df.columns[1:]  # All attributes

    train_all_attributes(df, embeddings, attributes_to_train)
