# le reseau de neurones qui apprend a conduire
# petit modele avec 3 couches: rays en entree -> decisions en sortie

import torch
import torch.nn as nn


class DrivingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        # reseau simple: entree (50 rays) -> 64 neurones -> 64 neurones -> 2 sorties (throttle + steering)
        # tanh a la fin pour que les valeurs restent entre -1 et 1
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),  # throttle + steering
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, observation):
        # utilise le modele pour predire une action a partir des raycasts
        # normalise les observations avec les memes params que l'entrainement
        self.eval()
        with torch.no_grad():
            obs = observation.copy()
            if hasattr(self, 'x_min') and hasattr(self, 'x_max'):
                obs = (obs - self.x_min) / (self.x_max - self.x_min + 1e-8)
            x = torch.FloatTensor(obs).unsqueeze(0)
            return self.net(x).numpy()
