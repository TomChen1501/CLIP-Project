import torch
import numpy as np
import pandas as pd
from utility.data_utils import load_celeb_attribute
from models.mlp.inference_wrapper import AttributeTransferEngine
from torch.nn.functional import cosine_similarity

df = load_celeb_attribute()
data = torch.load("Resource/all_image_embeddings.pt", weights_only=True)
embeddings = data["embeddings"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attributes = ["Smiling", "Young", "Male", "Bald"]
embedding_dim = embeddings.shape[1]
hidden_dim = 256
lr = 0.0001
num_epochs = 10
seeds = [1, 40, 200, 772, 114514]

rows = []

for attr in attributes:
    pos_idx = df[df[attr] == 1].index
    neg_idx = df[df[attr] == -1].index
    X_pos = embeddings[pos_idx]
    X_neg = embeddings[neg_idx]

    # Identity
    rows.append([attr, "Identity", "0.0000 ± 0.0000", "1.0000 ± 0.0000"])

    # Mean Vector Shift (with sampling and std)
    mses_mv, coss_mv = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        n = min(len(X_pos), len(X_neg))
        perm_pos = torch.randperm(len(X_pos))[:n]
        perm_neg = torch.randperm(len(X_neg))[:n]
        X_pos_sub = X_pos[perm_pos]
        X_neg_sub = X_neg[perm_neg]

        mean_diff = X_pos_sub.mean(dim=0) - X_neg_sub.mean(dim=0)
        shifted = X_neg_sub + mean_diff

        mse = torch.mean((shifted - X_pos_sub) ** 2).item()
        cos = cosine_similarity(shifted, X_pos_sub).mean().item()

        mses_mv.append(mse)
        coss_mv.append(cos)

    rows.append([
        attr,
        "Mean Vector Shift",
        f"{np.mean(mses_mv):.4f} ± {np.std(mses_mv):.4f}",
        f"{np.mean(coss_mv):.4f} ± {np.std(coss_mv):.4f}"
    ])

    # MLP (Neg → Pos)
    mses, cossims = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        engine = AttributeTransferEngine(embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device)
        mse, cos = engine.train_on_pair(X_neg, X_pos, num_epochs=num_epochs, lr=lr)
        mses.append(mse)
        cossims.append(cos)

    rows.append([
        attr,
        "MLP (Neg → Pos)",
        f"{np.mean(mses):.4f} ± {np.std(mses):.4f}",
        f"{np.mean(cossims):.4f} ± {np.std(cossims):.4f}"
    ])

# Format and output
df_table = pd.DataFrame(rows, columns=["Attribute", "Method", "MSE (mean ± std)", "Cosine Similarity (mean ± std)"])
print(df_table.to_latex(index=False, escape=False))  # escape=False allows ± to appear correctly in LaTeX
