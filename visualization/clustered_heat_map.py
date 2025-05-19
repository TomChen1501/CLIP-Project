import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import torch

encoded_data = torch.load('Resource/encoded_tensors.pt', weights_only=True)

smile = encoded_data['smile']
unsmile = encoded_data['unsmile']
young = encoded_data['young']
old = encoded_data['old']

X = np.concatenate([smile.cpu(), unsmile.cpu()])
y = np.array([1] * len(smile) + [0] * len(unsmile))

# subsample 10% of the data
subset_size = int(len(X) * 0.0003)
indices = np.random.choice(len(X), size=subset_size, replace=False)
X = X[indices]
y = y[indices]

selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X, y)

selected_feature_indices = selector.get_support(indices=True)
feature_names = [f'feature_{i}' for i in selected_feature_indices]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

df = pd.DataFrame(X_scaled, columns=feature_names)
df['label'] = y

row_colors = df['label'].map({0: 'blue', 1: 'red'})

sns.clustermap(df.drop('label', axis=1),
               method='average',
               metric='cosine',
               cmap='vlag',
               row_colors=row_colors,
               figsize=(12, 10))

plt.show()
