import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from models.svm.svmClassify import CLIPAttributeSVM
from utility.data_utils import load_celeb_attribute

# Load data
data = torch.load('Resource/all_image_embeddings.pt', weights_only=True)
df = load_celeb_attribute()
attribute = "Smiling"

pos_indices = df[df[attribute] == 1].index.tolist()
neg_indices = df[df[attribute] == -1].index.tolist()
all_embeddings = data['embeddings']
positive_tensor = all_embeddings[pos_indices]
negative_tensor = all_embeddings[neg_indices]

# Grid
kernels = ["linear", "rbf", "poly"]
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
seeds = [1, 40, 200, 772, 114514]

# Storage
results = {k: {"acc_mean": [], "acc_std": [], "f1_mean": [], "f1_std": []} for k in kernels}

# Evaluate
for kernel in kernels:
    print(f"\n=== Evaluating Kernel: {kernel} ===")
    for C in C_values:
        print(f"Evaluating C={C}...")
        accs, f1s = [], []
        for seed in seeds:
            svm = CLIPAttributeSVM(kernel=kernel, C=C, test_size=0.2, seed=seed)
            X_train, X_test, y_train, y_test = svm.prepare_data(positive_tensor, negative_tensor, sample_ratio=0.1)
            svm.train(X_train, y_train)
            y_pred = svm.model.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))

        results[kernel]["acc_mean"].append(np.mean(accs))
        results[kernel]["acc_std"].append(np.std(accs))
        results[kernel]["f1_mean"].append(np.mean(f1s))
        results[kernel]["f1_std"].append(np.std(f1s))

# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for kernel in kernels:
    axs[0].errorbar(C_values, results[kernel]["acc_mean"], yerr=results[kernel]["acc_std"],
                    label=kernel, marker='o', capsize=3)
    axs[1].errorbar(C_values, results[kernel]["f1_mean"], yerr=results[kernel]["f1_std"],
                    label=kernel, marker='o', capsize=3)

for ax, title in zip(axs, ["Accuracy vs C", "F1-score vs C"]):
    ax.set_xscale('log')
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(title="Kernel")

plt.tight_layout()
plt.show()
plt.savefig("fig/svm_performance_with_errorbars.png")

