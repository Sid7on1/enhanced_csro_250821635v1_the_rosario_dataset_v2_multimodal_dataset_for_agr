import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Define constants
DATA_DIR = 'data'
MODEL_DIR = 'models'
LOG_DIR = 'logs'
CONFIG_FILE = 'config.json'

# Define logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingConfig:
    def __init__(self, batch_size: int, epochs: int, learning_rate: float, validation_split: float):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.validation_split = validation_split

class DatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # Load data from files
        X = np.load(os.path.join(self.data_dir, 'X.npy'))
        y = np.load(os.path.join(self.data_dir, 'y.npy'))
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, validation_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        return X_train, X_val, y_train, y_val

class DataScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

class CustomDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[index], self.y[index]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(128, 128)  # input layer (128) -> hidden layer (128)
        self.fc2 = nn.Linear(128, 128)  # hidden layer (128) -> hidden layer (128)
        self.fc3 = nn.Linear(128, 2)  # hidden layer (128) -> output layer (2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer:
    def __init__(self, model: Model, device: torch.device, batch_size: int, epochs: int, learning_rate: float):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')
            self.model.eval()
            self.evaluate(val_loader)

    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == target).sum().item()
        accuracy = total_correct / len(val_loader.dataset)
        logger.info(f'Validation Accuracy: {accuracy:.4f}')

def main():
    # Load configuration
    config = TrainingConfig(batch_size=32, epochs=10, learning_rate=0.001, validation_split=0.2)

    # Load data
    dataset_loader = DatasetLoader(DATA_DIR)
    X, y = dataset_loader.load_data()
    X_train, X_val, y_train, y_val = dataset_loader.split_data(X, y, config.validation_split)

    # Scale data
    scaler = DataScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # Create data loaders
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Create model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    trainer = Trainer(model, device, config.batch_size, config.epochs, config.learning_rate)

    # Train model
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()