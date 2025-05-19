import torch
import os
from models.mlp.inference_wrapper import AttributeTransferEngine
from utility.data_utils import load_celeb_attribute
from torch.nn.functional import cosine_similarity
import numpy as np


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

def evaluate_all(df, embeddings, attributes_to_train, hidden_dim=256, num_epochs=20, lr=0.001, seeds=[1, 10, 20, 30, 40]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = embeddings.shape[1]

    for attribute in attributes_to_train:
        print(f"\n=== Evaluating Attribute: {attribute} ===")

        pos_indices = df[df[attribute] == 1].index.tolist()
        neg_indices = df[df[attribute] == -1].index.tolist()

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            print(f"Skipping {attribute}: not enough data.")
            continue

        X_pos = embeddings[pos_indices]
        X_neg = embeddings[neg_indices]

        # Evaluate both directions
        def run_direction(src, tgt):
            losses, sims = [], []
            for seed in seeds:
                torch.manual_seed(seed)
                engine = AttributeTransferEngine(embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device)
                mse, cos = engine.train_on_pair(src, tgt, num_epochs=num_epochs, lr=lr)
                losses.append(mse)
                sims.append(cos)
            return np.mean(losses), np.std(losses), np.mean(sims), np.std(sims)

        mse1, std1, cos1, cstd1 = run_direction(X_neg, X_pos)
        print(f"MLP (Neg → Pos)        | MSE: {mse1:.4f} ± {std1:.4f} | CosSim: {cos1:.4f} ± {cstd1:.4f}")

        mse2, std2, cos2, cstd2 = run_direction(X_pos, X_neg)
        print(f"MLP (Pos → Neg)        | MSE: {mse2:.4f} ± {std2:.4f} | CosSim: {cos2:.4f} ± {cstd2:.4f}")


if __name__ == "__main__":
    df = load_celeb_attribute()
    d = torch.load('Resource/all_image_embeddings.pt', weights_only=True)
    embeddings = d['embeddings']

    attributes_to_train = ['Smiling', 'Young', 'Male', 'Bald']
    evaluate_all(
        df=df,
        embeddings=embeddings,
        attributes_to_train=attributes_to_train,
        hidden_dim=256,
        num_epochs=10,
        lr=0.0001,
        seeds=[1, 40, 200, 772, 114514]
    )
