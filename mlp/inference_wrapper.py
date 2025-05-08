import torch
from mlp.mlp_model import *

class AttributeTransferEngine:
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model = MLPModel(embedding_dim, hidden_dim).to(self.device)

    def train_on_pair(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor, num_epochs=1000, lr=0.001):
        train_loader, val_loader, test_loader = create_dataloaders(source_tensor, target_tensor)
        train_model(self.model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=self.device)
        test_loss = evaluate_model(self.model, test_loader, device=self.device)
        cosine_similarity = cosine_similarity_evaluation(self.model, test_loader, device=self.device)
        return test_loss, cosine_similarity

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def transform_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = embedding.to(self.device).float()
            diff = self.model(embedding)
            transformed = embedding + diff
            return transformed.cpu()

    def batch_transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = embeddings.to(self.device).float()
            diffs = self.model(embeddings)
            transformed = embeddings + diffs
            return transformed.cpu()

if __name__ == "__main__":
    pass