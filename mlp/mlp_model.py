import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class MLPModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.layers(x)


def create_dataloaders(source_tensor, target_tensor, batch_size=64):
    source_tensor = source_tensor.float()
    target_tensor = target_tensor.float()

    num_pairs = min(source_tensor.shape[0], target_tensor.shape[0])
    indices_source = torch.randperm(source_tensor.shape[0])[:num_pairs]
    indices_target = torch.randperm(target_tensor.shape[0])[:num_pairs]

    source_subset = source_tensor[indices_source]
    target_subset = target_tensor[indices_target]

    difference_vectors = source_subset - target_subset  

    X = source_subset
    Y = difference_vectors

    num_total = X.shape[0]
    num_train = int(num_total * 0.7)
    num_val = int(num_total * 0.15)

    indices = torch.randperm(num_total)
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    test_indices = indices[num_train + num_val :]

    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    Y_train, Y_val, Y_test = Y[train_indices], Y[val_indices], Y[test_indices]

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs=1000, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, Y_val_batch in val_loader:
                X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
                val_preds = model(X_val_batch)
                val_loss = criterion(val_preds, Y_val_batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs} | Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    criterion = nn.MSELoss()
    total_test_loss = 0.0
    with torch.no_grad():
        for X_test_batch, Y_test_batch in test_loader:
            X_test_batch, Y_test_batch = X_test_batch.to(device), Y_test_batch.to(device)
            test_preds = model(X_test_batch)
            loss = criterion(test_preds, Y_test_batch)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.6f}')
    return avg_test_loss


def cosine_similarity_evaluation(model, test_loader, device='cpu'):
    model.eval()
    cos_similarities = []
    with torch.no_grad():
        for X_test_batch, Y_test_batch in test_loader:
            X_test_batch, Y_test_batch = X_test_batch.to(device), Y_test_batch.to(device)
            predicted_diffs = model(X_test_batch)
            predicted_smile_embeddings = X_test_batch + predicted_diffs
            actual_smile_embeddings = X_test_batch + Y_test_batch
            cosine_sim = F.cosine_similarity(predicted_smile_embeddings, actual_smile_embeddings, dim=1)
            cos_similarities.append(cosine_sim.cpu())

    cos_similarities = torch.cat(cos_similarities)
    mean_cos_sim = cos_similarities.mean().item()
    print(f'Mean Cosine Similarity on Test Set: {mean_cos_sim:.4f}')
    return mean_cos_sim

if __name__ == "__main__":
    data = torch.load("Resource/encoded_tensors.pt", weights_only=True)

    smile_encoded_tensor = data['smile']
    unsmile_encoded_tensor = data['unsmile']
    young_encoded_tensor = data['young']
    old_encoded_tensor = data['old']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = smile_encoded_tensor.shape[1]
    hidden_dim = 256
    train_loader, val_loader, test_loader = create_dataloaders(smile_encoded_tensor, unsmile_encoded_tensor)

    model = MLPModel(embedding_dim, hidden_dim).to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device=device)
    test_loss = evaluate_model(model, test_loader, device=device)
    cosine_similarity = cosine_similarity_evaluation(model, test_loader, device=device)
