import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def get_task_metadata():
    return {
        'task_name': 'mlp_newsgroups_tfidf',
        'description': 'MLP Text Classifier on 20 Newsgroups with TF-IDF',
        'categories': ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.guns'],
        'model': 'TextMLP',
        'framework': 'pytorch',
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_dataloaders(batch_size=64, max_features=5000):
    categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.guns']

    train_data = fetch_20newsgroups(subset='train', categories=categories)
    test_data = fetch_20newsgroups(subset='test', categories=categories)

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_data.data).toarray()
    X_test = vectorizer.transform(test_data.data).toarray()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(train_data.target)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(test_data.target)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class TextMLP(nn.Module):
    def __init__(self, input_dim=5000, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def build_model(input_dim=5000, num_classes=4, device=None):
    if device is None:
        device = get_device()
    model = TextMLP(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)
    return model


def train(model, train_loader, val_loader, epochs=20, lr=1e-3):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / num_batches
        loss_history.append(avg_train_loss)

        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_running_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_running_loss / val_batches
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    return {'loss_history': loss_history, 'val_loss_history': val_loss_history}


def evaluate(model, data_loader):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    model.eval()

    all_preds = []
    all_targets = []
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            num_batches += 1
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = float(np.mean(all_preds == all_targets))
    macro_f1 = float(f1_score(all_targets, all_preds, average='macro'))
    avg_loss = running_loss / num_batches

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'loss': avg_loss}


def predict(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        X = X.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
    return preds


def save_artifacts(model, metrics, loss_history, val_loss_history, val_loader):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_checkpoint.pt'))

    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['comp.graphics', 'rec.sport.baseball',
                                                  'sci.space', 'talk.politics.guns'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_newsgroups_cm.png'))
    plt.close()


if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device)

    history = train(model, train_loader, val_loader, epochs=20)

    print("\n--- Train Metrics ---")
    train_metrics = evaluate(model, train_loader)
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Validation Metrics ---")
    val_metrics = evaluate(model, val_loader)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    save_artifacts(model, val_metrics, history['loss_history'], history['val_loss_history'], val_loader)

    assert val_metrics['macro_f1'] > 0.75, f"macro_f1 too low: {val_metrics['macro_f1']:.4f}"
    print("OK")
    sys.exit(0)
