# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import ContextRNN
from generate_data import generate_trials
import numpy as np


def prepare_dataloader(trials, batch_size=32, use_sequences=True):
    """
    Convert generated trial data to PyTorch DataLoader.
    """
    if use_sequences and "sequence" in trials:
        # Use full sequences
        X = np.array(trials["sequence"], dtype=np.float32)  # Shape: (N, seq_len, input_dim)
    else:
        # Use single timestep (backward compatibility)
        X = np.array(trials["input"], dtype=np.float32)
        X = X[:, np.newaxis, :]  # Add sequence dimension: (B, T=1, F)
    
    y = np.array(trials["label"], dtype=np.int64)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, dataloader, num_epochs=10, lr=1e-3):
    """
    Train the RNN model using cross-entropy loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()

        acc = correct / len(dataloader.dataset)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    return model


def save_model(model, filepath="trained_model.pth"):
    """
    Save trained model state dict.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    # Generate training data with sequences
    trials = generate_trials(n_trials=1000, noise_std=0.25, structured_noise=True, seq_len=5, seed=42)
    dataloader = prepare_dataloader(trials, use_sequences=True)

    # Init and train model
    model = ContextRNN(input_size=3, hidden_size=64, output_size=2, rnn_type="gru")
    trained_model = train_model(model, dataloader, num_epochs=20)
    
    # Save the trained model
    save_model(trained_model)
