import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import pandas as pd

# Generate dummy data
np.random.seed(0)
data = np.random.rand(8, 8)
rows = [f'Row {i+1}' for i in range(8)]
cols = [f'Col {i+1}' for i in range(8)]
df = pd.DataFrame(data, index=rows, columns=cols)

# Create a clustered heatmap using seaborn
sns.set(style="white")
g = sns.clustermap(df, cmap="coolwarm", linewidths=0.5, figsize=(8, 8), annot=False)

# Improve layout and display
plt.subplots_adjust(top=0.9)
plt.suptitle("Example of a Clustered Heatmap", fontsize=14)
plt.show()
g.savefig("fig/clustered_heatmap.png", dpi=300, bbox_inches='tight')
