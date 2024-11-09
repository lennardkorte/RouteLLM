import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from routellm.routers.matrix_factorization.model import MODEL_IDS

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MultiModelDataset(Dataset):
    def __init__(self, data):
        self.prompt_ids = [sample["idx"] for sample in data]
        self.models = torch.tensor([MODEL_IDS[sample["model"]] for sample in data])
        self.labels = torch.tensor([MODEL_IDS[sample["winner"]] for sample in data])

    def __len__(self):
        return len(self.prompt_ids)

    def __getitem__(self, index):
        prompt_id = self.prompt_ids[index]
        model_ids = self.models[index]
        label = self.labels[index]  # Winner label (if applicable)
        
        return model_ids, prompt_id, label

    def get_dataloaders(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class MFModel_Train(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_models,
        num_prompts,
        text_dim=1536,
        use_proj=True,
        npy_path=None,
    ):
        super().__init__()
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)  # Model embedding matrix
        self.Q = torch.nn.Embedding(num_prompts, text_dim, requires_grad=False)
        embeddings = np.load(npy_path)
        self.Q.weight.data.copy_(torch.tensor(embeddings))  # Load prompt embeddings

        if self.use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)

        self.classifier = nn.Linear(dim, 1, bias=False)  # Compatibility score

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_ids, prompt_id, test=False, alpha=0.05):
        model_ids = model_ids.to(self.get_device())
        prompt_id = prompt_id.to(self.get_device())

        # Model and prompt embeddings
        model_embed = self.P(model_ids)  # Shape: [batch_size, num_models, dim]
        model_embed = F.normalize(model_embed, p=2, dim=2)
        prompt_embed = self.Q(prompt_id)  # Shape: [batch_size, text_dim]

        if not test:
            # Adding noise to stabilize training
            prompt_embed += torch.randn_like(prompt_embed) * alpha
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)  # Project prompt embedding

        prompt_embed = F.normalize(prompt_embed, p=2, dim=1).unsqueeze(1)  # Shape: [batch_size, 1, dim]
        
        # Compatibility scores for each model
        scores = self.classifier(model_embed * prompt_embed).squeeze(-1)  # Shape: [batch_size, num_models]
        
        return scores  # Output compatibility scores for each model

    @torch.no_grad()
    def predict(self, model_ids, prompt_id):
        # Calculate scores and select the highest score model per prompt
        scores = self.forward(model_ids, prompt_id, test=True)
        return scores.argmax(dim=1)  # Predicted model with highest compatibility score per prompt

def evaluator(net, test_iter, device):
    net.eval()
    correct, total_loss, num_samples = 0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    
    with torch.no_grad():
        for model_ids, prompt_id, labels in test_iter:
            model_ids = model_ids.to(device)
            prompt_id = prompt_id.to(device)
            labels = labels.to(device)

            scores = net(model_ids, prompt_id, test=True)
            loss = criterion(scores, labels)
            total_loss += loss.item()
            predictions = scores.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            num_samples += labels.size(0)
    
    return total_loss / num_samples, correct / num_samples

def train_loops(
    net,
    train_iter,
    test_iter,
    lr,
    weight_decay,
    alpha,
    num_epochs,
    device="cuda",
    evaluator=None,
):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()  # Cross-entropy for multi-class (multi-model) prediction

    def train_epoch():
        net.train()
        train_loss_sum, n = 0.0, 0
        for model_ids, prompt_id, labels in train_iter:
            model_ids, prompt_id, labels = model_ids.to(device), prompt_id.to(device), labels.to(device)
            scores = net(model_ids, prompt_id, alpha=alpha)
            loss = loss_fn(scores, labels)
            
            # Add L2 regularization on the embedding layers
            l2_lambda = 0.01
            l2_reg = l2_lambda * (net.P.weight.norm(2) + net.Q.weight.norm(2))
            loss += l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * model_ids.size(0)
            n += model_ids.size(0)
        return train_loss_sum / n

    train_losses, test_losses, test_accuracies = [], [], []
    best_test_acc = -1

    progress_bar = tqdm(total=num_epochs, desc="Training Progress", unit="epoch")

    for epoch in range(num_epochs):
        train_loss = train_epoch()
        train_losses.append(train_loss)

        # Update progress information
        info = {"train_loss": train_loss, "epoch": epoch + 1}
        if evaluator:
            test_loss, test_acc = evaluator(net, test_iter, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            info.update({"test_loss": test_loss, "test_acc": test_acc})
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                info["best_test_acc"] = best_test_acc

        # Update progress bar with epoch info
        progress_bar.set_postfix(info)
        progress_bar.update(1)

    progress_bar.close()


if __name__ == "__main__":
    # an example of training the model
    json_path = "/path/to/pairwise_data.json"
    npy_path = "/path/to/prompt/embedding.npy"

    dim = 128
    batch_size = 64
    num_epochs = 100
    alpha = 0.1
    use_proj = True
    lr = 3e-4
    weight_decay = 1e-5

    # load and filter data
    data = json.load(open(json_path, "r"))

    filtered_data = [
        sample
        for sample in data
        if sample["winner"] in ["model_a", "model_b"]
        and sample["model_a"] != sample["model_b"]
    ]

    # shuffle and prepare train test split
    data_shuffled = filtered_data.copy()
    random.shuffle(data_shuffled)
    train_data = data_shuffled[: int(len(data_shuffled) * 0.95)]
    test_data = data_shuffled[int(len(data_shuffled) * 0.95) :]

    train_data_loader = MultiModelDataset(train_data).get_dataloaders(
        batch_size=batch_size, shuffle=True
    )
    test_data_loader = MultiModelDataset(test_data).get_dataloaders(1024, shuffle=False)


    model = MFModel_Train(
        dim=dim,
        num_models=len(MODEL_IDS),
        num_prompts=len(data),
        use_proj=use_proj,
        npy_path=npy_path,
    ).to("cuda")

    train_loops(
        model,
        train_data_loader,
        test_data_loader,
        lr=lr,
        weight_decay=weight_decay,
        alpha=alpha,
        num_epochs=num_epochs,
        device="cuda",
    )
