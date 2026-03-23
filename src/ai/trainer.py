# gere tout l'entrainement du modele: charger les donnees, entrainer, evaluer, sauvegarder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

from src.ai.model import DrivingModel
from src.ai.metrics import print_metrics
from src.collector.data_collector import DataCollector


class Trainer:
    def __init__(self, hidden_size=64, lr=1e-3, batch_size=64):
        self.hidden_size = hidden_size
        self.lr = lr              # learning rate: vitesse d'apprentissage
        self.batch_size = batch_size  # nombre de samples par batch
        self.model = None
        self.history = {"train_loss": [], "val_loss": []}

    def prepare_data(self, data_dir="data", val_split=0.2):
        # charge tous les csv, separe les rays (X) des actions (y)
        # split 80% train / 20% validation
        df = DataCollector.load_all(data_dir)
        ray_cols = [c for c in df.columns if c.startswith("ray_")]
        X = df[ray_cols].values.astype(np.float32)
        y = df[["throttle", "steering"]].values.astype(np.float32)

        # normalise les observations (rays) entre 0 et 1 pour que le modele apprenne mieux
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        X = (X - self.x_min) / (self.x_max - self.x_min + 1e-8)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)

        # cree le modele avec le bon nombre d'entrees (50 rays)
        self.model = DrivingModel(input_size=X.shape[1], hidden_size=self.hidden_size)

        # transforme en tensors pytorch et cree les dataloaders
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.batch_size)
        )

    def train(self, data_dir="data", epochs=100, val_split=0.2):
        train_loader, val_loader = self.prepare_data(data_dir, val_split)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()  # mean squared error: mesure l'ecart entre prediction et realite

        for epoch in range(epochs):
            self.model.train()
            train_loss = self._run_epoch(train_loader, criterion, optimizer)
            val_loss = self._evaluate(val_loader, criterion)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

        # affiche les metriques finales (mse, mae, rmse, r2)
        self._print_final_metrics(val_loader)
        return self.history

    def _print_final_metrics(self, val_loader):
        # compare les predictions du modele avec les vraies actions sur le set de validation
        self.model.eval()
        X_all = torch.cat([X for X, _ in val_loader])
        y_all = torch.cat([y for _, y in val_loader])
        with torch.no_grad():
            preds = self.model(X_all).numpy()
        print_metrics(y_all.numpy(), preds, label="Validation")

    def _run_epoch(self, loader, criterion, optimizer):
        # une epoch = on passe sur toutes les donnees une fois
        total = 0
        for X_batch, y_batch in loader:
            loss = criterion(self.model(X_batch), y_batch)
            optimizer.zero_grad()  # reset les gradients
            loss.backward()        # calcule les gradients
            optimizer.step()       # met a jour les poids
            total += loss.item() * len(X_batch)
        return total / len(loader.dataset)

    def _evaluate(self, loader, criterion):
        # evalue le modele sur le set de validation (sans modifier les poids)
        self.model.eval()
        with torch.no_grad():
            total = sum(criterion(self.model(X), y).item() * len(X) for X, y in loader)
        return total / len(loader.dataset)

    def save_model(self, path="models/driving_model.pth"):
        # save le modele + ses parametres + normalisation pour pouvoir le recharger plus tard
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_size": self.model.net[0].in_features,
            "hidden_size": self.hidden_size,
            "x_min": self.x_min,
            "x_max": self.x_max,
        }, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path="models/driving_model.pth"):
        # charge un modele sauvegarde avec ses params de normalisation
        checkpoint = torch.load(path, weights_only=False)
        model = DrivingModel(input_size=checkpoint["input_size"], hidden_size=checkpoint["hidden_size"])
        model.load_state_dict(checkpoint["state_dict"])
        # on attache les params de normalisation au modele pour l'inference
        model.x_min = checkpoint["x_min"]
        model.x_max = checkpoint["x_max"]
        return model
