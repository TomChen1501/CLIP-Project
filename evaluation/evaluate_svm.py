import os
import torch
import numpy as np
from models.svm.svmClassify import CLIPAttributeSVM
from utility.data_utils import load_celeb_attribute
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

data = torch.load('Resource/all_image_embeddings.pt', weights_only=True)
df = load_celeb_attribute()
attributes_to_train = ['Smiling', 'Young', 'Male', 'Bald']

seeds = [1, 40, 200, 772, 114514]

for attribute in attributes_to_train:
    print(f"\n=== Evaluating Attribute: {attribute} ===")

    pos_indices = df[df[attribute] == 1].index.tolist()
    neg_indices = df[df[attribute] == -1].index.tolist()
    all_embeddings = data['embeddings']
    positive_tensor = all_embeddings[pos_indices]
    negative_tensor = all_embeddings[neg_indices]

    X = torch.cat([positive_tensor, negative_tensor]).cpu().numpy()
    y = np.array([1] * len(positive_tensor) + [0] * len(negative_tensor))

    acc_random, f1_random = [], []
    acc_major, f1_major = [], []
    acc_lin, f1_lin = [], []
    acc_rbf, f1_rbf = [], []
    acc_poly, f1_poly = [], []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Random Baseline
        dummy_random = DummyClassifier(strategy="uniform", random_state=seed)
        dummy_random.fit(X_train, y_train)
        y_pred = dummy_random.predict(X_test)
        acc_random.append(accuracy_score(y_test, y_pred))
        f1_random.append(f1_score(y_test, y_pred))

        # Majority Baseline
        dummy_majority = DummyClassifier(strategy="most_frequent", random_state=seed)
        dummy_majority.fit(X_train, y_train)
        y_pred = dummy_majority.predict(X_test)
        acc_major.append(accuracy_score(y_test, y_pred))
        f1_major.append(f1_score(y_test, y_pred))

        # Linear SVM
        svm = CLIPAttributeSVM(kernel="linear", C=1.0, test_size=0.2, seed=seed)
        acc, f1 = svm.train_and_evaluate(positive_tensor, negative_tensor, sample_ratio=0.1)
        acc_lin.append(acc)
        f1_lin.append(f1)

        # RBF SVM
        svm = CLIPAttributeSVM(kernel="rbf", C=1.0, test_size=0.2, seed=seed)
        acc, f1 = svm.train_and_evaluate(positive_tensor, negative_tensor, sample_ratio=0.1)
        acc_rbf.append(acc)
        f1_rbf.append(f1)

        # Poly SVM
        svm = CLIPAttributeSVM(kernel="poly", C=1.0, test_size=0.2, seed=seed)
        acc, f1 = svm.train_and_evaluate(positive_tensor, negative_tensor, sample_ratio=0.1)
        acc_poly.append(acc)
        f1_poly.append(f1)

    def fmt(mean, std): return f"{mean:.4f} ± {std:.4f}"

    print(f"[Random] Accuracy: {fmt(np.mean(acc_random), np.std(acc_random))}, F1: {fmt(np.mean(f1_random), np.std(f1_random))}")
    print(f"[Majority] Accuracy: {fmt(np.mean(acc_major), np.std(acc_major))}, F1: {fmt(np.mean(f1_major), np.std(f1_major))}")
    print(f"[Linear SVM] Accuracy: {fmt(np.mean(acc_lin), np.std(acc_lin))}, F1: {fmt(np.mean(f1_lin), np.std(f1_lin))}")
    print(f"[RBF SVM] Accuracy: {fmt(np.mean(acc_rbf), np.std(acc_rbf))}, F1: {fmt(np.mean(f1_rbf), np.std(f1_rbf))}")
    print(f"[Poly SVM] Accuracy: {fmt(np.mean(acc_poly), np.std(acc_poly))}, F1: {fmt(np.mean(f1_poly), np.std(f1_poly))}")
