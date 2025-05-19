import torch
import itertools
from models.mlp.inference_wrapper import AttributeTransferEngine
from utility.data_utils import load_celeb_attribute
from torch.nn.functional import cosine_similarity
import numpy as np

def evaluate_mlp(X_src, X_tgt, embedding_dim, hidden_dim, lr, num_epochs=100, device='cpu', seeds=[1,2,3,4,5]):
    sims = []
    sims = []
    losses = []
    for seed in seeds:
        torch.manual_seed(seed)
        engine = AttributeTransferEngine(embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device)
        mse, cos_sim = engine.train_on_pair(X_src, X_tgt, num_epochs=num_epochs, lr=lr)
        sims.append(cos_sim)
        losses.append(mse)
    return {
        "cos_mean": np.mean(sims),
        "cos_std": np.std(sims),
        "mse_mean": np.mean(losses),
        "mse_std": np.std(losses),
    }

if __name__ == "__main__":
    df = load_celeb_attribute()
    d = torch.load('Resource/all_image_embeddings.pt', weights_only=True)
    embeddings = d['embeddings']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attribute = 'Smiling'
    pos_indices = df[df[attribute] == 1].index.tolist()
    neg_indices = df[df[attribute] == -1].index.tolist()
    X_pos = embeddings[pos_indices]
    X_neg = embeddings[neg_indices]
    embedding_dim = embeddings.shape[1]

    # hidden_dims = [64, 128, 256, 512]
    hidden_dims = [128, 256]
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]

    print(f"{'Hidden':>6} | {'LR':>7} | {'MSE (mean ± std)':>20} | {'CosSim (mean ± std)':>24}")
    print("-" * 65)

    for hidden_dim, lr in itertools.product(hidden_dims, learning_rates):
        out = evaluate_mlp(X_neg, X_pos, embedding_dim, hidden_dim, lr, num_epochs=100, device=device)
        print(f"{hidden_dim:6} | {lr:<7.0e} | {out['mse_mean']:.4f} ± {out['mse_std']:.4f}      | {out['cos_mean']:.4f} ± {out['cos_std']:.4f}")
