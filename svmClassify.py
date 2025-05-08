import torch
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class CLIPAttributeSVM:
    def __init__(self, kernel="linear", C=1.0, test_size=0.2, seed=114514):
        self.kernel = kernel
        self.C = C
        self.test_size = test_size
        self.seed = seed
        self.model = SVC(kernel=self.kernel, C=self.C)

    def prepare_data(self, positive_tensor, negative_tensor, sample_ratio=1.0):
        X = np.concatenate([positive_tensor.cpu(), negative_tensor.cpu()])
        y = np.array([1] * len(positive_tensor) + [0] * len(negative_tensor))

        if sample_ratio < 1.0:
            subset_size = int(len(X) * sample_ratio)
            indices = np.random.choice(len(X), size=subset_size, replace=False)
            X = X[indices]
            y = y[indices]

        return train_test_split(X, y, test_size=self.test_size, random_state=self.seed)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def train_and_evaluate(self, positive_tensor, negative_tensor, sample_ratio=0.1):
        X_train, X_test, y_train, y_test = self.prepare_data(positive_tensor, negative_tensor, sample_ratio)
        self.train(X_train, y_train)
        accuracy, report = self.evaluate(X_test, y_test)
        print("Accuracy:", accuracy)
        print(report)
        return accuracy, report
    
    def save_model(self, path=f"trained_models/svm_model.pkl"):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path=f"trained_models/svm_model.pkl"):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    data = torch.load('Resource/encoded_tensors.pt', weights_only=True)
    smile_encoded_tensor = data['smile']
    unsmile_encoded_tensor = data['unsmile']
    svm = CLIPAttributeSVM(kernel="linear", C=1.0, test_size=0.2, seed=114514)
    svm.train_and_evaluate(smile_encoded_tensor, unsmile_encoded_tensor,sample_ratio=0.1)
    svm.save_model(path="trained_models/svm_smile_model.pkl")

