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
        # pas besoin de calculer les gradients, c'est juste de l'inference
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(observation).unsqueeze(0) if not isinstance(observation, torch.Tensor) else observation
            return self.net(x).numpy()
