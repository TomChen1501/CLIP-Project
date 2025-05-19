import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from tuning results
data = {
    "Hidden": [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512],
    "LR": [0.0001, 0.0005, 0.001, 0.005]*4,
    "CosSim": [
        0.9051, 0.9031, 0.8993, 0.8866,
        0.9134, 0.9114, 0.9087, 0.8926,
        0.9184, 0.9162, 0.9128, 0.8962,
        0.9206, 0.9157, 0.9124, 0.8964
    ]
}

df = pd.DataFrame(data)

# Pivot for heatmap
heatmap_data = df.pivot(index="Hidden", columns="LR", values="CosSim")

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Cosine Similarity'})
plt.ylabel("Hidden Layer Size")
plt.xlabel("Learning Rate")
plt.tight_layout()
plt.show()
