import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cosine similarity values
cosine_data = np.array([
    [0.9048, 0.9029, 0.8995, 0.8833],
    [0.9135, 0.9116, 0.9090, 0.8923],
    [0.9184, 0.9154, 0.9127, 0.8952],
    [0.9205, 0.9164, 0.9126, 0.8957],
])

# Axis labels
hidden_dims = [64, 128, 256, 512]
learning_rates = ["1e-4", "5e-4", "1e-3", "5e-3"]

# Plot heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    cosine_data,
    annot=True,
    fmt=".4f",
    xticklabels=learning_rates,
    yticklabels=hidden_dims,
    cmap="YlGnBu",
    cbar_kws={"label": "Cosine Similarity"}  # 👉 label for colorbar
)

# Axes and title
plt.xlabel("Learning Rate")
plt.ylabel("Hidden Layer Size")
# plt.title("Cosine Similarity Heatmap for MLP (Smiling)")
plt.tight_layout()
plt.savefig("fig/mlp_heatmap_cosine_right_label.png")
plt.show()
