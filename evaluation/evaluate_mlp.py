import torch
import os
from models.mlp.inference_wrapper import AttributeTransferEngine
from source.data_utils import load_celeb_attribute
from torch.nn.functional import cosine_similarity

def evaluate_baseline(source, target, baseline_type):
    with torch.no_grad():
        if baseline_type == "identity":
            predicted = source
        elif baseline_type == "mean_shift":
            mu_source = torch.mean(source, dim=0, keepdim=True)
            mu_target = torch.mean(target, dim=0, keepdim=True)
            shift = mu_target - mu_source
            predicted = source + shift
        else:
            raise ValueError("Unknown baseline")

        # Ensure predicted and target are the same length
        n = min(predicted.size(0), target.size(0))
        predicted = predicted[:n]
        actual = target[:n]

        mse = torch.mean((predicted - actual) ** 2).item()
        cos = cosine_similarity(predicted, actual, dim=1).mean().item()
        return mse, cos

def evaluate_all(df, embeddings, attributes_to_train, hidden_dim=256, num_epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = embeddings.shape[1]

    for attribute in attributes_to_train:
        print(f"\n=== Evaluating Attribute: {attribute} ===")
        engine = AttributeTransferEngine(embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device)

        pos_indices = df[df[attribute] == 1].index.tolist()
        neg_indices = df[df[attribute] == -1].index.tolist()

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            print(f"Skipping {attribute}: not enough data.")
            continue

        X_pos = embeddings[pos_indices]
        X_neg = embeddings[neg_indices]

        # MLP: Neg → Pos
        test_loss, cosine_sim = engine.train_on_pair(X_neg, X_pos, num_epochs=num_epochs, lr=lr)
        print(f"MLP (Neg → Pos)        | MSE: {test_loss:.4f} | CosSim: {cosine_sim:.4f}")

        # Baseline: Identity (Neg → Neg)
        mse_id, cos_id = evaluate_baseline(X_neg, X_neg, "identity")
        print(f"Identity Baseline      | MSE: {mse_id:.4f} | CosSim: {cos_id:.4f}")

        # Baseline: Mean Shift (Neg → Pos)
        mse_mv, cos_mv = evaluate_baseline(X_neg, X_pos, "mean_shift")
        print(f"Mean Shift Baseline    | MSE: {mse_mv:.4f} | CosSim: {cos_mv:.4f}")

if __name__ == "__main__":
    df = load_celeb_attribute()
    d = torch.load('Resource/all_image_embeddings.pt', weights_only=True)
    embeddings = d['embeddings']

    attributes_to_train = ['Smiling', 'Young', 'Male', 'Bald']
    evaluate_all(df, embeddings, attributes_to_train)
