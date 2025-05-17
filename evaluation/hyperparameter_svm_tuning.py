from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import torch
import numpy as np
from models.svm.svmClassify import CLIPAttributeSVM
from source.data_utils import load_celeb_attribute

data = torch.load('Resource/all_image_embeddings.pt', weights_only=True)
df = load_celeb_attribute()

attribute = "Smiling"
pos_indices = df[df[attribute] == 1].index.tolist()
neg_indices = df[df[attribute] == -1].index.tolist()

all_embeddings = data['embeddings']
positive_tensor = all_embeddings[pos_indices]
negative_tensor = all_embeddings[neg_indices]

kernels = ["linear", "rbf", "poly"]
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

results = []

for kernel in kernels:
    for C in C_values:
        print(f"Training {kernel} SVM with C={C}")
        svm = CLIPAttributeSVM(kernel=kernel, C=C, test_size=0.2, seed=114514)
        X_train, X_test, y_train, y_test = svm.prepare_data(positive_tensor, negative_tensor, sample_ratio=0.1)
        svm.train(X_train, y_train)
        y_pred = svm.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append((kernel, C, acc, f1))

# Save to LaTeX-compatible table
df_out = pd.DataFrame(results, columns=["Kernel", "C", "Accuracy", "F1-score"])
df_out.to_csv("svm_hyperparam_results.csv", index=False)
print(df_out.to_latex(index=False, float_format="%.3f"))
